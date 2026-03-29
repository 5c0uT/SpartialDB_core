import logging
import os
import sys
from pathlib import Path

import laspy
import matplotlib
import numpy as np
import pandas as pd
from skimage import measure

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import nibabel as nib
except ImportError:
    nib = None

try:
    import pydicom
except ImportError:
    pydicom = None


project_root = Path(__file__).resolve().parent
python_path = project_root / "python"
if str(python_path) not in sys.path:
    sys.path.insert(0, str(python_path))

from spatial_db.core import SpatialConfig, SpatialDB


class LidarMedicalProcessor:
    def __init__(self, config=None):
        self.logger = self.setup_logging()
        self.db = SpatialDB(config or SpatialConfig())
        self.lidar_data = None
        self.medical_volume = None
        self.point_cloud = None

    def setup_logging(self):
        logger = logging.getLogger("LidarMedical")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def load_lidar(self, file_path):
        self.logger.info(f"Loading LiDAR data from {file_path}")
        try:
            with laspy.open(file_path) as reader:
                las = reader.read()
                self.lidar_data = pd.DataFrame(
                    {
                        "x": las.x,
                        "y": las.y,
                        "z": las.z,
                        "intensity": las.intensity,
                        "classification": las.classification,
                        "return_number": las.return_number,
                        "number_of_returns": las.number_of_returns,
                    }
                )

            z_mean = self.lidar_data["z"].mean()
            z_std = self.lidar_data["z"].std()
            self.lidar_data = self.lidar_data[
                (self.lidar_data["z"] > z_mean - 3 * z_std)
                & (self.lidar_data["z"] < z_mean + 3 * z_std)
            ]
            self.logger.info(f"Loaded {len(self.lidar_data)} LiDAR points")
            return True
        except Exception as exc:
            self.logger.error(f"Failed to load LiDAR data: {exc}")
            return False

    def load_medical_scan(self, file_path):
        self.logger.info(f"Loading medical scan from {file_path}")
        try:
            if file_path.lower().endswith(".dcm"):
                if pydicom is None:
                    raise ImportError("pydicom is not installed")
                dicom = pydicom.dcmread(file_path)
                self.medical_volume = dicom.pixel_array
            elif file_path.lower().endswith((".nii", ".nii.gz")):
                if nib is None:
                    raise ImportError("nibabel is not installed")
                nifti = nib.load(file_path)
                self.medical_volume = nifti.get_fdata()
            else:
                raise ValueError("Unsupported medical scan format")

            self.medical_volume = (self.medical_volume - self.medical_volume.min()) / (
                self.medical_volume.max() - self.medical_volume.min()
            )
            self.logger.info(f"Loaded medical scan with shape {self.medical_volume.shape}")
            return True
        except Exception as exc:
            self.logger.error(f"Failed to load medical scan: {exc}")
            return False

    def convert_medical_to_pointcloud(self, threshold=0.5, sampling_rate=0.01):
        if self.medical_volume is None:
            self.logger.error("Medical scan not loaded")
            return False

        self.logger.info("Converting medical scan to point cloud...")
        try:
            binary_volume = self.medical_volume > threshold
            verts, _faces, _normals, _values = measure.marching_cubes(binary_volume, level=0.5, spacing=(1, 1, 1))
            num_points = int(len(verts) * sampling_rate)
            if num_points <= 0:
                raise ValueError("No points generated from medical scan")
            indices = np.random.choice(len(verts), size=num_points, replace=False)
            self.point_cloud = verts[indices]
            self.logger.info(f"Generated {len(self.point_cloud)} points from medical scan")
            return True
        except Exception as exc:
            self.logger.error(f"Failed to convert medical scan: {exc}")
            return False

    def visualize_data(self, data_type="lidar", sample_size=10000, output_file=None):
        fig = plt.figure(figsize=(15, 10))

        if data_type == "lidar" and self.lidar_data is not None:
            sample = self.lidar_data.sample(min(sample_size, len(self.lidar_data)))
            ax = fig.add_subplot(111, projection="3d")
            sc = ax.scatter(sample["x"], sample["y"], sample["z"], c=sample["z"], cmap="viridis", s=1)
            plt.colorbar(sc, label="Elevation")
            ax.set_title("LiDAR Point Cloud")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
        elif data_type == "medical" and self.point_cloud is not None:
            points = self.point_cloud
            if len(points) > sample_size:
                indices = np.random.choice(len(points), size=sample_size, replace=False)
                points = points[indices]
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="plasma", s=1)
            ax.set_title("Medical Scan Point Cloud")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
        elif data_type == "volume" and self.medical_volume is not None:
            ax = fig.add_subplot(111)
            slice_idx = self.medical_volume.shape[2] // 2
            ax.imshow(self.medical_volume[:, :, slice_idx], cmap="gray")
            ax.set_title("Medical Scan Slice")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
        else:
            plt.close(fig)
            self.logger.warning("No data available for visualization")
            return None

        plt.tight_layout()
        if output_file:
            plt.savefig(output_file, dpi=150)
            self.logger.info(f"Saved visualization to {output_file}")
        plt.close(fig)
        return output_file

    def analyze_density(self, resolution=5.0):
        all_points = self._combined_points()
        if all_points is None:
            self.logger.error("No data available for density analysis")
            return None

        mins = all_points.min(axis=0)
        maxs = all_points.max(axis=0)
        bins = [max(1, int((maxs[i] - mins[i]) / resolution)) for i in range(3)]
        hist, _edges = np.histogramdd(all_points, bins=bins, range=tuple((mins[i], maxs[i]) for i in range(3)))
        self.logger.info(f"Created density histogram with shape {hist.shape}")
        return hist

    def calculate_surface_area(self, threshold=0.5):
        if self.medical_volume is None:
            self.logger.error("Medical scan not loaded")
            return None

        try:
            binary_volume = self.medical_volume > threshold
            verts, faces, _normals, _values = measure.marching_cubes(binary_volume, level=0.5)
            surface_area = measure.mesh_surface_area(verts, faces)
            self.logger.info(f"Calculated surface area: {surface_area:.2f} units^2")
            return surface_area
        except Exception as exc:
            self.logger.error(f"Failed to calculate surface area: {exc}")
            return None

    def export_point_cloud(self, output_path, format="las"):
        all_points = self._combined_points()
        if all_points is None:
            self.logger.error("No data to export")
            return False

        try:
            if format == "las":
                header = laspy.LasHeader(point_format=3, version="1.4")
                las = laspy.LasData(header)
                las.x = all_points[:, 0]
                las.y = all_points[:, 1]
                las.z = all_points[:, 2]
                las.write(output_path)
            elif format == "ply":
                with open(output_path, "w", encoding="ascii") as handle:
                    handle.write("ply\nformat ascii 1.0\n")
                    handle.write(f"element vertex {len(all_points)}\n")
                    handle.write("property float x\nproperty float y\nproperty float z\nend_header\n")
                    for point in all_points:
                        handle.write(f"{point[0]} {point[1]} {point[2]}\n")
            else:
                raise ValueError("Unsupported export format")
            self.logger.info(f"Exported {len(all_points)} points to {output_path}")
            return True
        except Exception as exc:
            self.logger.error(f"Export failed: {exc}")
            return False

    def spatial_analysis(self):
        all_points = self._combined_points()
        if all_points is None:
            self.logger.error("No data for spatial analysis")
            return None

        self.logger.info("Loading data into SpatialDB...")
        try:
            origins = all_points.copy()
            origins[:, 2] += 10
            directions = np.tile([0.0, 0.0, -1.0], (len(origins), 1))
            max_dists = np.full(len(origins), 20.0)
            hits = self.db.batch_raycast(origins, directions, max_dists)
            hit_distances = [hit.distance for hit in hits if hit.distance > 0]
            hit_ratio = len(hit_distances) / len(origins) if len(origins) else 0
            self.logger.info(f"Raycast results: {len(hit_distances)} hits ({hit_ratio:.1%} hit ratio)")
            if not hit_distances:
                return 0.0
            avg_distance = float(np.mean(hit_distances))
            self.logger.info(f"Average hit distance: {avg_distance:.2f} units")
            return avg_distance
        except Exception as exc:
            self.logger.error(f"Spatial analysis failed: {exc}")
            return None

    def _combined_points(self):
        point_sets = []
        if self.lidar_data is not None:
            point_sets.append(self.lidar_data[["x", "y", "z"]].to_numpy())
        if self.point_cloud is not None:
            point_sets.append(self.point_cloud)
        if not point_sets:
            return None
        return np.vstack(point_sets)


def create_synthetic_lidar():
    return pd.DataFrame(
        {
            "x": np.random.uniform(0, 1000, 10000),
            "y": np.random.uniform(0, 1000, 10000),
            "z": np.random.uniform(0, 100, 10000),
            "intensity": np.random.uniform(0, 255, 10000),
            "classification": np.random.randint(0, 10, 10000),
            "return_number": np.random.randint(1, 4, 10000),
            "number_of_returns": np.random.randint(1, 4, 10000),
        }
    )


def create_synthetic_medical_volume():
    volume = np.zeros((100, 100, 100))
    x, y, z = np.indices((100, 100, 100))
    sphere = (x - 50) ** 2 + (y - 50) ** 2 + (z - 50) ** 2 < 30 ** 2
    volume[sphere] = 1.0
    volume += np.random.normal(0, 0.1, volume.shape)
    return np.clip(volume, 0, 1)


def main():
    processor = LidarMedicalProcessor()

    lidar_file = "path/to/your/lidar_data.las"
    if os.path.exists(lidar_file):
        processor.load_lidar(lidar_file)
    else:
        processor.logger.warning("LiDAR file not found, creating synthetic data...")
        processor.lidar_data = create_synthetic_lidar()

    medical_file = "path_to_your_medical_scan.nii.gz"
    if os.path.exists(medical_file):
        processor.load_medical_scan(medical_file)
    else:
        processor.logger.warning("Medical file not found, creating synthetic data...")
        processor.medical_volume = create_synthetic_medical_volume()

    processor.convert_medical_to_pointcloud()
    processor.visualize_data("lidar", output_file="lidar_preview.png")
    processor.visualize_data("medical", output_file="medical_preview.png")
    processor.visualize_data("volume", output_file="medical_slice.png")

    density = processor.analyze_density()
    average_distance = processor.spatial_analysis()
    processor.export_point_cloud("combined_point_cloud.las")
    surface_area = processor.calculate_surface_area()

    stats = pd.DataFrame(
        {
            "metric": ["density_shape", "avg_distance", "surface_area"],
            "value": [
                str(density.shape if density is not None else None),
                average_distance,
                surface_area,
            ],
        }
    )
    stats.to_csv("lidar_medical_stats.csv", index=False)
    processor.logger.info("Saved lidar_medical_stats.csv")
    processor.logger.info("Processing completed successfully")


if __name__ == "__main__":
    main()
