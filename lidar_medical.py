import sys
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import laspy
import pydicom
import nibabel as nib
from scipy import ndimage
from skimage import measure
from tqdm import tqdm

# Настройка корневого пути проекта
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Путь к модулям
python_path = os.path.join(project_root, 'python')
if python_path not in sys.path:
    sys.path.insert(0, python_path)

try:
    from spatial_db.core import SpatialDB, SpatialConfig
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class LidarMedicalProcessor:
    def __init__(self, config=None):
        self.logger = self.setup_logging()
        self.db = SpatialDB(config or SpatialConfig())
        self.lidar_data = None
        self.medical_volume = None
        self.point_cloud = None

    def setup_logging(self):
        """Настройка системы логирования"""
        logger = logging.getLogger("LidarMedical")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def load_lidar(self, file_path):
        """Загрузка данных LiDAR из LAS/LAZ файла"""
        self.logger.info(f"Loading LiDAR data from {file_path}")
        try:
            with laspy.open(file_path) as reader:
                las = reader.read()

                # Создаем DataFrame с основными атрибутами
                self.lidar_data = pd.DataFrame({
                    'x': las.x,
                    'y': las.y,
                    'z': las.z,
                    'intensity': las.intensity,
                    'classification': las.classification,
                    'return_number': las.return_number,
                    'number_of_returns': las.number_of_returns
                })

                # Фильтрация выбросов (ИСПРАВЛЕННЫЙ БЛОК)
                z_mean = self.lidar_data['z'].mean()
                z_std = self.lidar_data['z'].std()

                # Правильный перенос строки с использованием скобок
                self.lidar_data = self.lidar_data[
                    (self.lidar_data['z'] > z_mean - 3 * z_std) &
                    (self.lidar_data['z'] < z_mean + 3 * z_std)
                    ]

                self.logger.info(f"Loaded {len(self.lidar_data)} LiDAR points")
                return True
        except Exception as e:
            self.logger.error(f"Failed to load LiDAR data: {e}")
            return False

    def load_medical_scan(self, file_path):
        """Загрузка медицинских сканов (DICOM или NIfTI)"""
        self.logger.info(f"Loading medical scan from {file_path}")
        try:
            if file_path.lower().endswith('.dcm'):
                # Обработка DICOM файлов
                dicom = pydicom.dcmread(file_path)
                self.medical_volume = dicom.pixel_array
                self.logger.info(f"Loaded DICOM scan with shape {self.medical_volume.shape}")
            elif file_path.lower().endswith(('.nii', '.nii.gz')):
                # Обработка NIfTI файлов
                nifti = nib.load(file_path)
                self.medical_volume = nifti.get_fdata()
                self.logger.info(f"Loaded NIfTI scan with shape {self.medical_volume.shape}")
            else:
                self.logger.error("Unsupported medical scan format")
                return False

            # Нормализация значений
            self.medical_volume = (self.medical_volume - self.medical_volume.min()) / \
                                  (self.medical_volume.max() - self.medical_volume.min())

            return True
        except Exception as e:
            self.logger.error(f"Failed to load medical scan: {e}")
            return False

    def convert_medical_to_pointcloud(self, threshold=0.5, sampling_rate=0.01):
        """Преобразование медицинского скана в облако точек"""
        if self.medical_volume is None:
            self.logger.error("Medical scan not loaded")
            return False

        self.logger.info("Converting medical scan to point cloud...")

        try:
            # Бинаризация изображения по порогу
            binary_volume = self.medical_volume > threshold

            # Извлечение поверхности с помощью Marching Cubes
            verts, faces, _, _ = measure.marching_cubes(
                binary_volume,
                level=0.5,
                spacing=(1, 1, 1))

            # Случайная выборка точек с поверхности
            num_points = int(len(verts) * sampling_rate)
            if num_points > 0:
                indices = np.random.choice(len(verts), size=num_points, replace=False)
                self.point_cloud = verts[indices]
                self.logger.info(f"Generated {len(self.point_cloud)} points from medical scan")
                return True
            else:
                self.logger.error("No points generated from medical scan")
                return False

        except Exception as e:
            self.logger.error(f"Failed to convert medical scan: {e}")
            return False


def visualize_data(self, data_type='lidar', sample_size=10000):
    """Визуализация данных"""
    fig = plt.figure(figsize=(15, 10))

    if data_type == 'lidar' and self.lidar_data is not None:
        # Визуализация LiDAR данных
        sample = self.lidar_data.sample(min(sample_size, len(self.lidar_data)))

        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(
            sample['x'], sample['y'], sample['z'],
            c=sample['z'], cmap='viridis', s=1
        )
        plt.colorbar(sc, label='Elevation')
        ax.set_title('LiDAR Point Cloud')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    elif data_type == 'medical' and self.point_cloud is not None:
        # Визуализация медицинского облака точек
        if len(self.point_cloud) > sample_size:
            indices = np.random.choice(len(self.point_cloud), size=sample_size, replace=False)
            points = self.point_cloud[indices]
        else:
            points = self.point_cloud

        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            c=points[:, 2], cmap='plasma', s=1
        )
        ax.set_title('Medical Scan Point Cloud')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    elif data_type == 'volume' and self.medical_volume is not None:
        # Визуализация медицинского объема
        ax = fig.add_subplot(111)
        slice_idx = self.medical_volume.shape[2] // 2
        ax.imshow(self.medical_volume[:, :, slice_idx], cmap='gray')
        ax.set_title('Medical Scan Slice')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    else:
        self.logger.warning("No data available for visualization")
        return

    plt.tight_layout()
    plt.show()


def analyze_density(self, resolution=1.0):
    """Анализ плотности точек"""
    if self.lidar_data is None and self.point_cloud is None:
        self.logger.error("No data available for density analysis")
        return None

    # Объединение данных LiDAR и медицинских сканов
    if self.lidar_data is not None and self.point_cloud is not None:
        lidar_points = self.lidar_data[['x', 'y', 'z']].values
        all_points = np.vstack([lidar_points, self.point_cloud])
    elif self.lidar_data is not None:
        all_points = self.lidar_data[['x', 'y', 'z']].values
    else:
        all_points = self.point_cloud

    # Определение границ
    x_min, y_min, z_min = all_points.min(axis=0)
    x_max, y_max, z_max = all_points.max(axis=0)

    # Создание 3D гистограммы
    x_bins = int((x_max - x_min) / resolution)
    y_bins = int((y_max - y_min) / resolution)
    z_bins = int((z_max - z_min) / resolution)

    # Рассчет гистограммы
    hist, edges = np.histogramdd(
        all_points,
        bins=(x_bins, y_bins, z_bins),
        range=((x_min, x_max), (y_min, y_max), (z_min, z_max)))

    self.logger.info(f"Created density histogram with shape {hist.shape}")
    return hist


def calculate_surface_area(self, threshold=0.5):
    """Расчет площади поверхности для медицинских сканов"""
    if self.medical_volume is None:
        self.logger.error("Medical scan not loaded")
        return None

    try:
        # Бинаризация изображения по порогу
        binary_volume = self.medical_volume > threshold

        # Расчет площади поверхности
        verts, faces, _, _ = measure.marching_cubes(binary_volume, level=0.5)
        surface_area = measure.mesh_surface_area(verts, faces)

        self.logger.info(f"Calculated surface area: {surface_area:.2f} units²")
        return surface_area
    except Exception as e:
        self.logger.error(f"Failed to calculate surface area: {e}")
        return None


def export_point_cloud(self, output_path, format='las'):
    """Экспорт объединенного облака точек"""
    if self.lidar_data is None and self.point_cloud is None:
        self.logger.error("No data to export")
        return False

    try:
        # Создание объединенного облака точек
        points = []

        if self.lidar_data is not None:
            lidar_points = self.lidar_data[['x', 'y', 'z']].values
            points.append(lidar_points)

        if self.point_cloud is not None:
            points.append(self.point_cloud)

        if not points:
            self.logger.error("No points to export")
            return False

        all_points = np.vstack(points)

        if format == 'las':
            # Экспорт в LAS
            header = laspy.LasHeader(point_format=3, version="1.4")
            las = laspy.LasData(header)

            las.x = all_points[:, 0]
            las.y = all_points[:, 1]
            las.z = all_points[:, 2]

            las.write(output_path)
            self.logger.info(f"Exported {len(all_points)} points to {output_path}")
            return True
        elif format == 'ply':
            # Экспорт в PLY
            with open(output_path, 'w') as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(all_points)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("end_header\n")

                for point in all_points:
                    f.write(f"{point[0]} {point[1]} {point[2]}\n")

            self.logger.info(f"Exported {len(all_points)} points to {output_path}")
            return True
        else:
            self.logger.error("Unsupported export format")
            return False
    except Exception as e:
        self.logger.error(f"Export failed: {e}")
        return False


def spatial_analysis(self):
    """Пространственный анализ с использованием SpatialDB"""
    if self.lidar_data is None and self.point_cloud is None:
        self.logger.error("No data for spatial analysis")
        return None

    # Подготовка данных для SpatialDB
    points = []

    if self.lidar_data is not None:
        lidar_points = self.lidar_data[['x', 'y', 'z']].values
        points.append(lidar_points)

    if self.point_cloud is not None:
        points.append(self.point_cloud)

    if not points:
        self.logger.error("No points for spatial analysis")
        return None

    all_points = np.vstack(points)

    # Загрузка данных в SpatialDB
    self.logger.info("Loading data into SpatialDB...")
    try:
        # Для демонстрации - создание вертикальных лучей
        origins = all_points.copy()
        origins[:, 2] += 10  # Поднимаем источник на 10 единиц

        directions = np.tile([0, 0, -1], (len(origins), 1))
        max_dists = np.full(len(origins), 20.0)  # Макс. расстояние 20 единиц

        # Выполнение пакетного raycast
        hits = self.db.batch_raycast(origins, directions, max_dists)

        # Анализ результатов
        hit_distances = [h.distance for h in hits if h.distance > 0]
        hit_ratio = len(hit_distances) / len(origins) if len(origins) > 0 else 0

        self.logger.info(f"Raycast results: {len(hit_distances)} hits ({hit_ratio:.1%} hit ratio)")

        # Расчет среднего расстояния
        if hit_distances:
            avg_distance = np.mean(hit_distances)
            self.logger.info(f"Average hit distance: {avg_distance:.2f} units")
            return avg_distance
        else:
            self.logger.warning("No hits detected")
            return 0
    except Exception as e:
        self.logger.error(f"Spatial analysis failed: {e}")
        return None


def main():
    """Основная функция для демонстрации"""
    processor = LidarMedicalProcessor()

    # Создание тестовых данных, если реальные файлы не найдены
    test_data_created = False

    # Пример обработки LiDAR данных
    lidar_file = "path/to/your/lidar_data.las"
    if os.path.exists(lidar_file):
        processor.load_lidar(lidar_file)
        processor.visualize_data('lidar')
    else:
        # Создание синтетических LiDAR данных
        processor.logger.warning("LiDAR file not found, creating synthetic data...")
        processor.lidar_data = pd.DataFrame({
            'x': np.random.uniform(0, 1000, 10000),
            'y': np.random.uniform(0, 1000, 10000),
            'z': np.random.uniform(0, 100, 10000),
            'intensity': np.random.uniform(0, 255, 10000),
            'classification': np.random.randint(0, 10, 10000),
            'return_number': np.random.randint(1, 4, 10000),
            'number_of_returns': np.random.randint(1, 4, 10000)
        })
        test_data_created = True

    # Пример обработки медицинских сканов
    medical_file = "path_to_your_medical_scan.nii.gz"
    if os.path.exists(medical_file):
        processor.load_medical_scan(medical_file)
        processor.visualize_data('volume')
        processor.convert_medical_to_pointcloud()
        processor.visualize_data('medical')
    else:
        # Создание синтетического медицинского скана
        processor.logger.warning("Medical file not found, creating synthetic data...")
        processor.medical_volume = np.zeros((100, 100, 100))

        # Создаем сферу внутри объема
        x, y, z = np.indices((100, 100, 100))
        sphere = (x - 50) ** 2 + (y - 50) ** 2 + (z - 50) ** 2 < 30 ** 2
        processor.medical_volume[sphere] = 1.0

        # Добавляем шум
        processor.medical_volume += np.random.normal(0, 0.1, (100, 100, 100))
        processor.medical_volume = np.clip(processor.medical_volume, 0, 1)

        processor.convert_medical_to_pointcloud()
        test_data_created = True

    # Визуализация данных
    if test_data_created:
        processor.visualize_data('lidar')
        processor.visualize_data('medical')

    # Анализ данных
    processor.analyze_density()
    spatial_result = processor.spatial_analysis()

    # Экспорт объединенных данных
    processor.export_point_cloud("combined_point_cloud.las")

    # Расчет площади поверхности для медицинских сканов
    surface_area = processor.calculate_surface_area()
    if surface_area:
        processor.logger.info(f"Surface area: {surface_area:.2f} units²")

    processor.logger.info("Processing completed successfully")


if __name__ == "__main__":
    main()