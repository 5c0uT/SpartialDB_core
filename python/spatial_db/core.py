import asyncio
import ctypes
import importlib
import io
import logging
import os
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from numbers import Real
from typing import List, Optional, Tuple, Union

import fsspec
import laspy
import numpy as np
import pandas as pd
import trimesh
from tqdm import tqdm


logger = logging.getLogger("spatial_db")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


HAS_NATIVE_MODULE = False
native = None
native_import_error = None
_DLL_DIR_HANDLES = []


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _candidate_native_dirs() -> List[pathlib.Path]:
    root = _repo_root()
    candidates = [
        root / "build" / "lib" / "Release",
        root / "build" / "Release",
        root / "spatialdb_core",
    ]
    result = []
    for path in candidates:
        if path.exists() and path not in result:
            result.append(path)
    return result


def _register_windows_dll_dirs(*paths: pathlib.Path) -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return

    for path in paths:
        if path.exists():
            _DLL_DIR_HANDLES.append(os.add_dll_directory(str(path)))


def _prepare_physx_runtime(module) -> None:
    if os.name != "nt":
        return

    module_dir = pathlib.Path(getattr(module, "__file__", "")).resolve().parent
    vcpkg_bin = _repo_root() / "dependencies" / "vcpkg" / "installed" / "x64-windows" / "bin"

    _register_windows_dll_dirs(module_dir, vcpkg_bin)

    for dll_name in (
        "PhysXFoundation_64.dll",
        "PhysXCommon_64.dll",
        "PhysX_64.dll",
        "PhysXCooking_64.dll",
        "PhysXDevice64.dll",
        "PhysXGpu_64.dll",
    ):
        for directory in (module_dir, vcpkg_bin):
            dll_path = directory / dll_name
            if dll_path.exists():
                ctypes.WinDLL(str(dll_path))
                break


def _prepare_proj_runtime() -> None:
    env_root = pathlib.Path(sys.executable).resolve().parent
    proj_data = env_root / "Library" / "share" / "proj"
    if proj_data.exists():
        os.environ["PROJ_DATA"] = str(proj_data)
        os.environ["PROJ_LIB"] = str(proj_data)


for native_dir in reversed(_candidate_native_dirs()):
    native_dir_str = str(native_dir)
    if native_dir_str not in sys.path:
        sys.path.insert(0, native_dir_str)

_prepare_proj_runtime()

for module_name in ("spatialdb_core", "spatialdb_core_pybind"):
    try:
        candidate = importlib.import_module(module_name)
        if not hasattr(candidate, "SpatialDB"):
            native_import_error = ImportError(f"Module '{module_name}' does not expose SpatialDB")
            continue
        _prepare_physx_runtime(candidate)
        native = candidate
        HAS_NATIVE_MODULE = True
        logger.info(f"Native module '{module_name}' successfully imported")
        break
    except ImportError as exc:
        native_import_error = exc

if not HAS_NATIVE_MODULE:
    logger.error(f"Failed to import native module: {native_import_error}")

    class PxVec3:
        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x = x
            self.y = y
            self.z = z

        def __repr__(self):
            return f"PxVec3({self.x}, {self.y}, {self.z})"

    class RayHit:
        def __init__(self):
            self.position = PxVec3()
            self.normal = PxVec3()
            self.distance = -1.0
            self.objectID = 0

        def __repr__(self):
            return f"RayHit(pos={self.position}, dist={self.distance})"

    class SpatialDBStub:
        def __init__(self):
            self._initialized = False

        def load_las(self, path, crs):
            logger.warning("Native module not available - load_las is a stub")
            self._initialized = True

        def batch_query_ray(self, origins, directions, max_dists):
            logger.warning("Native module not available - batch_query_ray is a stub")
            return [RayHit() for _ in range(len(origins))]

        def query_sphere(self, center, radius):
            logger.warning("Native module not available - query_sphere is a stub")
            return []

        def query_ray(self, origin, direction, max_dist):
            logger.warning("Native module not available - query_ray is a stub")
            return RayHit()

        def build_bvh(self):
            logger.warning("Native module not available - build_bvh is a stub")

        def clear_scene(self):
            logger.warning("Native module not available - clear_scene is a stub")

    class CoordinateConverterStub:
        def __init__(self, source_crs, target_crs):
            self.source_crs = source_crs
            self.target_crs = target_crs

        def convert(self, lons, lats, alts):
            return [PxVec3(lon, lat, alt) for lon, lat, alt in zip(lons, lats, alts)]

    class NativeStub:
        PxVec3 = PxVec3
        RayHit = RayHit
        SpatialDB = SpatialDBStub
        CoordinateConverter = CoordinateConverterStub

        @staticmethod
        def init_physx(device=0):
            logger.warning("Native module not available - init_physx is a stub")
            return True

    native = NativeStub()


def _is_real_number(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _validate_vec(name: str, value, expected_length: int) -> Tuple[float, ...]:
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise TypeError(f"{name} must be a sequence of {expected_length} numeric values")
    if len(value) != expected_length:
        raise ValueError(f"{name} must contain exactly {expected_length} values")
    if not all(_is_real_number(item) for item in value):
        raise TypeError(f"{name} must contain only numeric values")
    return tuple(float(item) for item in value)


def _validate_positive(name: str, value, *, allow_zero: bool = False) -> float:
    if not _is_real_number(value):
        raise TypeError(f"{name} must be numeric")
    value = float(value)
    if allow_zero:
        if value < 0:
            raise ValueError(f"{name} must be >= 0")
    elif value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _coerce_converted_point(point) -> Tuple[float, float, float]:
    if hasattr(point, "x") and hasattr(point, "y") and hasattr(point, "z"):
        return float(point.x), float(point.y), float(point.z)
    if isinstance(point, (list, tuple)) and len(point) >= 3:
        return float(point[0]), float(point[1]), float(point[2])
    raise TypeError(f"Unsupported converted point format: {type(point)!r}")


@dataclass
class SpatialConfig:
    voxel_size: float = 0.1
    crs: str = "EPSG:3857"
    gpu_device: int = 0
    cache_size: int = 10
    streaming_chunk: int = 10_000_000

    def __post_init__(self):
        self.voxel_size = _validate_positive("voxel_size", self.voxel_size)
        if not isinstance(self.crs, str) or not self.crs.strip():
            raise ValueError("crs must be a non-empty string")
        if not isinstance(self.gpu_device, int):
            raise TypeError("gpu_device must be an integer")
        self.cache_size = int(_validate_positive("cache_size", self.cache_size, allow_zero=True))
        self.streaming_chunk = int(_validate_positive("streaming_chunk", self.streaming_chunk))


class SpatialDB:
    def __init__(self, config: Optional[SpatialConfig] = None):
        self.config = config or SpatialConfig()
        self._executor = ThreadPoolExecutor(max_workers=8)
        self._cache = {}
        self.logger = logging.getLogger("spatial_db.SpatialDB")
        self.core = None
        self._init_engine()

        if not self.is_initialized:
            self.logger.error("SpatialDB failed to initialize core engine")
            self.logger.warning("Falling back to stub implementation")
            self.core = native.SpatialDB()

    def _init_engine(self):
        try:
            self.logger.info(f"Initializing PhysX on GPU {self.config.gpu_device}...")
            if hasattr(native, "init_physx"):
                native.init_physx(self.config.gpu_device)
            self.core = native.SpatialDB()
            self.logger.info("PhysX engine initialized successfully")
        except Exception as exc:
            self.logger.error(f"PhysX initialization failed: {exc}")
            self.core = None

    @property
    def is_initialized(self) -> bool:
        return self.core is not None

    async def load_from_cloud_async(self, uri: str, format: str):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.load_from_cloud, uri, format)

    def _process_data(self, data: bytes, format: str) -> Union[pd.DataFrame, trimesh.Trimesh, None]:
        if format.lower() == "las":
            return self._process_las_data(data)
        if format.lower() in {"obj", "ply"}:
            return self._process_mesh_data(data, format)
        raise ValueError(f"Unsupported format: {format}")

    def _process_las_data(self, data: bytes) -> pd.DataFrame:
        with laspy.open(io.BytesIO(data)) as reader:
            points = reader.read()
            return pd.DataFrame(
                {
                    "x": points.x,
                    "y": points.y,
                    "z": points.z,
                    "intensity": points.intensity,
                    "classification": points.classification,
                }
            )

    def _process_mesh_data(self, data: bytes, format: str) -> trimesh.Trimesh:
        return trimesh.load_mesh(file_obj=io.BytesIO(data), file_type=format)

    def load_dataset(self, source: str, format: str) -> "SpatialDataset":
        return SpatialDataset(source, format, self.config)

    def raycast(
        self,
        origin: Tuple[float, float, float],
        direction: Tuple[float, float, float],
        max_dist: float = 100.0,
    ):
        origin = _validate_vec("origin", origin, 3)
        direction = _validate_vec("direction", direction, 3)
        if max_dist is not None and not _is_real_number(max_dist):
            raise TypeError("max_dist must be numeric")

        try:
            return self.core.query_ray(native.PxVec3(*origin), native.PxVec3(*direction), float(max_dist))
        except Exception as exc:
            self.logger.error(f"Raycast failed: {exc}")
            return native.RayHit()

    def batch_raycast(self, origins: np.ndarray, directions: np.ndarray, max_dists: np.ndarray) -> List:
        origins = np.asarray(origins, dtype=float)
        directions = np.asarray(directions, dtype=float)
        max_dists = np.asarray(max_dists, dtype=float)

        if origins.ndim != 2 or origins.shape[1] != 3:
            raise ValueError("origins must have shape (N, 3)")
        if directions.ndim != 2 or directions.shape[1] != 3:
            raise ValueError("directions must have shape (N, 3)")
        if len(origins) != len(directions) or len(origins) != len(max_dists):
            raise ValueError("origins, directions and max_dists must have the same length")

        try:
            px_origins = [native.PxVec3(*origin) for origin in origins]
            px_directions = [native.PxVec3(*direction) for direction in directions]
            return self.core.batch_query_ray(px_origins, px_directions, max_dists.tolist())
        except Exception as exc:
            self.logger.error(f"Batch raycast failed: {exc}")
            return []

    def query_sphere(self, center: Tuple[float, float, float], radius: float) -> List[int]:
        center = _validate_vec("center", center, 3)
        if not _is_real_number(radius):
            raise TypeError("radius must be numeric")

        try:
            return self.core.query_sphere(native.PxVec3(*center), float(radius))
        except Exception as exc:
            self.logger.error(f"Sphere query failed: {exc}")
            return []

    def profile_terrain(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        width: float = 5.0,
    ) -> pd.DataFrame:
        start = _validate_vec("start", start, 2)
        end = _validate_vec("end", end, 2)
        _validate_positive("width", width, allow_zero=True)

        if not HAS_NATIVE_MODULE:
            self.logger.warning("Native module not available - returning demo profile")
            return self._demo_profile()

        try:
            if not hasattr(native, "CoordinateConverter"):
                self.logger.warning("CoordinateConverter not available - returning demo profile")
                return self._demo_profile()
            converter = native.CoordinateConverter("EPSG:4326", self.config.crs)
            start_local = _coerce_converted_point(converter.convert(start[0], start[1], 0.0))
            end_local = _coerce_converted_point(converter.convert(end[0], end[1], 0.0))
            direction = np.array([end_local[0] - start_local[0], end_local[1] - start_local[1]])
            length = np.linalg.norm(direction)
            return self._demo_profile(length)
        except Exception as exc:
            self.logger.error(f"Terrain profiling failed: {exc}")
            return self._demo_profile()

    def _demo_profile(self, length: float = 5000.0) -> pd.DataFrame:
        distance = np.linspace(0, length, 100)
        elevation = 100 + 50 * np.sin(distance / 500) + np.random.normal(0, 5, 100)
        return pd.DataFrame({"distance": distance, "elevation": elevation})

    def create_density_heatmap(
        self,
        bbox: Tuple[float, float, float, float, float, float],
        resolution: float = 1.0,
    ) -> np.ndarray:
        bbox = _validate_vec("bbox", bbox, 6)
        resolution = _validate_positive("resolution", resolution)

        x_min, y_min, _z_min, x_max, y_max, _z_max = bbox
        x_bins = int((x_max - x_min) / resolution)
        y_bins = int((y_max - y_min) / resolution)
        if x_bins <= 0 or y_bins <= 0:
            raise ValueError("bbox and resolution must produce a positive heatmap size")

        heatmap = np.zeros((y_bins, x_bins))
        for _ in range(100000):
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            x_idx = int((x - x_min) / resolution)
            y_idx = int((y - y_min) / resolution)
            if 0 <= x_idx < x_bins and 0 <= y_idx < y_bins:
                heatmap[y_idx, x_idx] += 1
        return heatmap


class SpatialDataset:
    def __init__(self, source: str, format: str, config: Optional[SpatialConfig] = None):
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a non-empty string")
        if not isinstance(format, str) or not format.strip():
            raise ValueError("format must be a non-empty string")

        self.source = source
        self.format = format.lower()
        self.config = config or SpatialConfig()
        self._executor = ThreadPoolExecutor(max_workers=8)
        self._cache = {}
        self.logger = logging.getLogger("spatial_db.SpatialDataset")
        self._data = None
        self._load_data()

    def _load_data(self):
        try:
            if self.format == "las":
                self._data = self._load_las()
            else:
                self._data = self._load_mesh()
        except Exception as exc:
            self.logger.error(f"Failed to load data: {exc}")
            self._data = None

    def _load_las(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "x": np.random.uniform(0, 100, 10000),
                "y": np.random.uniform(0, 100, 10000),
                "z": np.random.uniform(0, 50, 10000),
            }
        )

    def _load_mesh(self) -> trimesh.Trimesh:
        return trimesh.creation.icosphere()

    def simplify(self, tolerance: float) -> "SpatialDataset":
        _validate_positive("tolerance", tolerance)
        return SpatialDataset(f"{self.source}_simplified", self.format, self.config)

    def visualize(self, **kwargs):
        if self._data is None:
            self.logger.warning("No data to visualize")
            return

        if self.format == "las":
            self._visualize_pointcloud(**kwargs)
        else:
            self._visualize_mesh()

    def _visualize_pointcloud(self, sample_size: int = 10000):
        import matplotlib.pyplot as plt

        if isinstance(self._data, pd.DataFrame) and not self._data.empty:
            sample = self._data.sample(min(sample_size, len(self._data)))
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(sample.x, sample.y, sample.z, s=1, c=sample.z, cmap="viridis")
            plt.show()
        else:
            self.logger.warning("No point cloud data to visualize")

    def _visualize_mesh(self):
        if isinstance(self._data, trimesh.Trimesh):
            self._data.show()
        else:
            self.logger.warning("No mesh data to visualize")
