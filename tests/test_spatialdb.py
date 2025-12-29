# pytest: pytest-asyncio
# -*- coding: utf-8 -*-
"""
Комплексные тесты для SpatialDB_Core

Покрытие:
- Unit тесты для Python API
- Интеграционные тесты с Native модулем
- Тесты производительности
- Тесты обработки ошибок
"""

import pytest
import sys
import os
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import time

# Добавляем пути для импорта
sys.path.insert(0, str(Path(__file__).parent.parent / "spatialdb_core"))
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

# Импортируем модули
import spatial_db
from spatial_db import SpatialDB, SpatialConfig, SpatialDataset

# ============================================================================
# КОНФИГУРАЦИЯ ЛОГИРОВАНИЯ
# ============================================================================

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def config():
    """Конфигурация по умолчанию"""
    return SpatialConfig(
        voxel_size=0.1,
        crs="EPSG:3857",
        gpu_device=0,
        cache_size=10
    )


@pytest.fixture
def spatial_db(config):
    """Экземпляр SpatialDB"""
    db = SpatialDB(config)
    yield db
    if db.is_initialized:
        db.core.clear_scene()


@pytest.fixture
def sample_points():
    """Генерация тестовых точек"""
    np.random.seed(42)
    return np.random.uniform(-100, 100, size=(1000, 3))


@pytest.fixture
def sample_rays():
    """Генерация тестовых лучей"""
    np.random.seed(42)
    origins = np.random.uniform(-50, 50, size=(100, 3))
    directions = np.random.uniform(-1, 1, size=(100, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    max_dists = np.full(100, 1000.0)
    return origins, directions, max_dists


# ============================================================================
# UNIT TESTS: КОНФИГУРАЦИЯ
# ============================================================================

class TestSpatialConfig:
    """Тесты класса SpatialConfig"""

    def test_default_config(self):
        """Проверка конфигурации по умолчанию"""
        config = SpatialConfig()
        assert config.voxel_size == 0.1
        assert config.crs == "EPSG:3857"
        assert config.gpu_device == 0
        assert config.cache_size == 10

    def test_custom_config(self):
        """Проверка пользовательской конфигурации"""
        config = SpatialConfig(
            voxel_size=0.5,
            crs="EPSG:4326",
            gpu_device=1,
            cache_size=20
        )
        assert config.voxel_size == 0.5
        assert config.crs == "EPSG:4326"
        assert config.gpu_device == 1
        assert config.cache_size == 20

    def test_invalid_voxel_size(self):
        """Проверка невалидного размера вокселя"""
        with pytest.raises((ValueError, TypeError)):
            SpatialConfig(voxel_size=-0.1)

    def test_invalid_cache_size(self):
        """Проверка невалидного размера кеша"""
        with pytest.raises((ValueError, TypeError)):
            SpatialConfig(cache_size=-1)


# ============================================================================
# UNIT TESTS: ИНИЦИАЛИЗАЦИЯ
# ============================================================================

class TestSpatialDBInitialization:
    """Тесты инициализации SpatialDB"""

    def test_initialization_with_config(self, config):
        """Проверка инициализации с конфигурацией"""
        db = SpatialDB(config)
        assert db is not None
        assert db.config == config
        assert db.is_initialized

    def test_initialization_default_config(self):
        """Проверка инициализации с конфигурацией по умолчанию"""
        db = SpatialDB()
        assert db is not None
        assert db.is_initialized

    def test_executor_created(self, spatial_db):
        """Проверка создания executor"""
        assert spatial_db._executor is not None
        assert spatial_db._executor._max_workers == 8

    def test_cache_initialized(self, spatial_db):
        """Проверка инициализации кеша"""
        assert isinstance(spatial_db._cache, dict)
        assert len(spatial_db._cache) == 0

    def test_logger_configured(self, spatial_db):
        """Проверка конфигурации логгера"""
        assert spatial_db.logger is not None


# ============================================================================
# UNIT TESTS: RAYCAST ОПЕРАЦИИ
# ============================================================================

class TestRaycastOperations:
    """Тесты raycast операций"""

    def test_single_raycast(self, spatial_db):
        """Проверка одиночного raycast"""
        origin = (0.0, 0.0, 0.0)
        direction = (0.0, 0.0, 1.0)
        
        hit = spatial_db.raycast(origin, direction)
        assert hit is not None

    def test_raycast_with_max_distance(self, spatial_db):
        """Проверка raycast с ограничением расстояния"""
        origin = (0.0, 0.0, 0.0)
        direction = (1.0, 0.0, 0.0)
        max_dist = 50.0
        
        hit = spatial_db.raycast(origin, direction, max_dist)
        assert hit is not None

    def test_batch_raycast(self, spatial_db, sample_rays):
        """Проверка пакетного raycast"""
        origins, directions, max_dists = sample_rays
        
        hits = spatial_db.batch_raycast(origins, directions, max_dists)
        assert isinstance(hits, list)
        assert len(hits) == len(origins)

    def test_raycast_with_invalid_direction(self, spatial_db):
        """Проверка raycast с невалидным направлением"""
        origin = (0.0, 0.0, 0.0)
        direction = (0.0, 0.0, 0.0)  # Нулевой вектор
        
        # Должно обработаться без ошибки
        hit = spatial_db.raycast(origin, direction)
        assert hit is not None

    def test_batch_raycast_empty(self, spatial_db):
        """Проверка пакетного raycast с пустыми данными"""
        origins = np.array([]).reshape(0, 3)
        directions = np.array([]).reshape(0, 3)
        max_dists = np.array([])
        
        hits = spatial_db.batch_raycast(origins, directions, max_dists)
        assert len(hits) == 0


# ============================================================================
# UNIT TESTS: СФЕРИЧЕСКИЕ ЗАПРОСЫ
# ============================================================================

class TestSphereQueries:
    """Тесты сферических запросов"""

    def test_sphere_query_basic(self, spatial_db):
        """Проверка базового запроса сферы"""
        center = (0.0, 0.0, 0.0)
        radius = 10.0
        
        result = spatial_db.query_sphere(center, radius)
        assert isinstance(result, list)

    def test_sphere_query_large_radius(self, spatial_db):
        """Проверка запроса с большим радиусом"""
        center = (0.0, 0.0, 0.0)
        radius = 1000.0
        
        result = spatial_db.query_sphere(center, radius)
        assert isinstance(result, list)

    def test_sphere_query_zero_radius(self, spatial_db):
        """Проверка запроса с нулевым радиусом"""
        center = (0.0, 0.0, 0.0)
        radius = 0.0
        
        result = spatial_db.query_sphere(center, radius)
        assert isinstance(result, list)

    def test_sphere_query_negative_radius(self, spatial_db):
        """Проверка запроса с отрицательным радиусом"""
        center = (0.0, 0.0, 0.0)
        radius = -10.0
        
        # Должно обработаться корректно
        result = spatial_db.query_sphere(center, radius)
        assert isinstance(result, list)


# ============================================================================
# UNIT TESTS: АНАЛИЗ МЕСТНОСТИ
# ============================================================================

class TestTerrainAnalysis:
    """Тесты анализа местности"""

    def test_profile_terrain_basic(self, spatial_db):
        """Проверка создания профиля местности"""
        start = (0.0, 0.0)
        end = (100.0, 100.0)
        
        profile = spatial_db.profile_terrain(start, end)
        assert isinstance(profile, pd.DataFrame)
        assert len(profile) > 0
        assert 'distance' in profile.columns
        assert 'elevation' in profile.columns

    def test_profile_terrain_with_width(self, spatial_db):
        """Проверка профиля местности с шириной"""
        start = (0.0, 0.0)
        end = (100.0, 100.0)
        width = 10.0
        
        profile = spatial_db.profile_terrain(start, end, width)
        assert isinstance(profile, pd.DataFrame)

    def test_density_heatmap_basic(self, spatial_db):
        """Проверка создания теплокарты плотности"""
        bbox = (-100.0, -100.0, -10.0, 100.0, 100.0, 10.0)
        
        heatmap = spatial_db.create_density_heatmap(bbox)
        assert isinstance(heatmap, np.ndarray)
        assert heatmap.shape[0] > 0
        assert heatmap.shape[1] > 0

    def test_density_heatmap_custom_resolution(self, spatial_db):
        """Проверка теплокарты с пользовательским разрешением"""
        bbox = (-100.0, -100.0, -10.0, 100.0, 100.0, 10.0)
        resolution = 5.0
        
        heatmap = spatial_db.create_density_heatmap(bbox, resolution)
        assert isinstance(heatmap, np.ndarray)
        assert heatmap.dtype == np.float64


# ============================================================================
# INTEGRATION TESTS: ПОЛНЫЕ СЦЕНАРИИ
# ============================================================================

class TestIntegrationScenarios:
    """Интеграционные тесты полных сценариев"""

    def test_workflow_raycast_and_sphere(self, spatial_db, sample_rays):
        """Проверка полного workflow raycast и sphere запросов"""
        origins, directions, max_dists = sample_rays
        
        # Первый raycast
        hits = spatial_db.batch_raycast(origins, directions, max_dists)
        assert len(hits) > 0
        
        # Sphere запрос
        if len(hits) > 0:
            center = (0.0, 0.0, 0.0)
            result = spatial_db.query_sphere(center, 100.0)
            assert isinstance(result, list)

    def test_workflow_terrain_analysis(self, spatial_db):
        """Проверка полного workflow анализа местности"""
        # Профиль местности
        profile = spatial_db.profile_terrain((0, 0), (100, 100))
        assert len(profile) > 0
        
        # Теплокарта
        heatmap = spatial_db.create_density_heatmap(
            (-100, -100, -10, 100, 100, 10)
        )
        assert heatmap.shape[0] > 0

    def test_concurrent_queries(self, spatial_db):
        """Проверка конкурентных запросов"""
        from concurrent.futures import ThreadPoolExecutor
        
        def raycast_task():
            origin = (0.0, 0.0, 0.0)
            direction = (0.0, 0.0, 1.0)
            return spatial_db.raycast(origin, direction)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(raycast_task) for _ in range(10)]
            results = [f.result() for f in futures]
        
        assert len(results) == 10


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Тесты производительности"""

    @pytest.mark.performance
    def test_raycast_throughput(self, spatial_db):
        """Проверка пропускной способности raycast"""
        num_rays = 1000
        origins = np.random.uniform(-50, 50, size=(num_rays, 3))
        directions = np.random.uniform(-1, 1, size=(num_rays, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        max_dists = np.full(num_rays, 1000.0)
        
        start_time = time.time()
        hits = spatial_db.batch_raycast(origins, directions, max_dists)
        elapsed = time.time() - start_time
        
        throughput = num_rays / elapsed
        logger.info(f"Raycast throughput: {throughput:.0f} rays/sec")
        
        assert len(hits) == num_rays
        assert throughput > 100  # Минимальная пропускная способность

    @pytest.mark.performance
    def test_sphere_query_performance(self, spatial_db):
        """Проверка производительности sphere запросов"""
        num_queries = 100
        centers = np.random.uniform(-50, 50, size=(num_queries, 3))
        
        start_time = time.time()
        for center in centers:
            spatial_db.query_sphere(tuple(center), 10.0)
        elapsed = time.time() - start_time
        
        throughput = num_queries / elapsed
        logger.info(f"Sphere query throughput: {throughput:.0f} queries/sec")
        
        assert elapsed > 0

    @pytest.mark.performance
    def test_memory_efficiency(self, spatial_db):
        """Проверка эффективности использования памяти"""
        import gc
        
        gc.collect()
        
        # Выполняем множество операций
        for _ in range(100):
            origins = np.random.uniform(-50, 50, size=(10, 3))
            directions = np.random.uniform(-1, 1, size=(10, 3))
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            max_dists = np.full(10, 1000.0)
            
            spatial_db.batch_raycast(origins, directions, max_dists)


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Тесты обработки ошибок"""

    def test_raycast_invalid_origin_type(self, spatial_db):
        """Проверка raycast с невалидным типом origin"""
        with pytest.raises((TypeError, ValueError)):
            spatial_db.raycast([0.0, 0.0], (0, 0, 1))

    def test_sphere_query_invalid_center_type(self, spatial_db):
        """Проверка sphere query с невалидным типом center"""
        with pytest.raises((TypeError, ValueError)):
            spatial_db.query_sphere([0.0, 0.0], 10.0)

    def test_profile_terrain_invalid_coordinates(self, spatial_db):
        """Проверка profile_terrain с невалидными координатами"""
        with pytest.raises((TypeError, ValueError)):
            spatial_db.profile_terrain(None, (100, 100))

    def test_density_heatmap_invalid_bbox(self, spatial_db):
        """Проверка density_heatmap с невалидным bbox"""
        with pytest.raises((TypeError, ValueError)):
            spatial_db.create_density_heatmap((1, 2, 3, 4, 5))


# ============================================================================
# NATIVE MODULE TESTS
# ============================================================================

class TestNativeModuleIntegration:
    """Тесты интеграции с нативным модулем"""

    def test_native_module_import(self):
        """Проверка импорта нативного модуля"""
        assert spatial_db.HAS_NATIVE_MODULE or spatial_db.native is not None

    def test_native_module_attributes(self):
        """Проверка атрибутов нативного модуля"""
        if spatial_db.HAS_NATIVE_MODULE:
            assert hasattr(spatial_db.native, 'SpatialDB')
            assert hasattr(spatial_db.native, 'PxVec3')
            assert hasattr(spatial_db.native, 'RayHit')

    def test_pxvec3_creation(self):
        """Проверка создания PxVec3"""
        vec = spatial_db.native.PxVec3(1.0, 2.0, 3.0)
        assert vec.x == 1.0
        assert vec.y == 2.0
        assert vec.z == 3.0

    def test_rayhit_creation(self):
        """Проверка создания RayHit"""
        hit = spatial_db.native.RayHit()
        assert hit.position is not None
        assert hit.normal is not None
        assert hit.distance == -1.0


# ============================================================================
# SPATIAL DATASET TESTS
# ============================================================================

class TestSpatialDataset:
    """Тесты для SpatialDataset"""

    def test_dataset_creation(self):
        """Проверка создания dataset"""
        dataset = SpatialDataset("test_source", "las")
        assert dataset is not None

    def test_dataset_with_config(self, config):
        """Проверка dataset с конфигурацией"""
        dataset = SpatialDataset("test_source", "las", config)
        assert dataset.config == config


# ============================================================================
# PARAMETRIZED TESTS
# ============================================================================

class TestParametrized:
    """Параметризованные тесты"""

    @pytest.mark.parametrize("origin,direction,expected", [
        ((0, 0, 0), (0, 0, 1), True),
        ((10, 10, 10), (1, 0, 0), True),
        ((-50, -50, -50), (-1, -1, -1), True),
    ])
    def test_raycast_variants(self, spatial_db, origin, direction, expected):
        """Проверка raycast с разными параметрами"""
        hit = spatial_db.raycast(origin, direction)
        assert (hit is not None) == expected

    @pytest.mark.parametrize("radius", [1.0, 10.0, 100.0, 1000.0])
    def test_sphere_query_radii(self, spatial_db, radius):
        """Проверка sphere query с разными радиусами"""
        result = spatial_db.query_sphere((0, 0, 0), radius)
        assert isinstance(result, list)

    @pytest.mark.parametrize("voxel_size", [0.01, 0.1, 0.5, 1.0])
    def test_config_voxel_sizes(self, voxel_size):
        """Проверка конфигурации с разными размерами вокселя"""
        config = SpatialConfig(voxel_size=voxel_size)
        assert config.voxel_size == voxel_size


# ============================================================================
# SMOKE TESTS
# ============================================================================

class TestSmokeTests:
    """Smoke тесты для быстрой проверки"""

    def test_basic_initialization(self):
        """Базовая проверка инициализации"""
        db = SpatialDB()
        assert db.is_initialized

    def test_basic_raycast(self):
        """Базовая проверка raycast"""
        db = SpatialDB()
        hit = db.raycast((0, 0, 0), (0, 0, 1))
        assert hit is not None

    def test_basic_sphere_query(self):
        """Базовая проверка sphere query"""
        db = SpatialDB()
        result = db.query_sphere((0, 0, 0), 10.0)
        assert isinstance(result, list)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
