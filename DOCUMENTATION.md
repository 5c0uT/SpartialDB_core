# SpatialDB - GPU-ускоренная пространственная база данных

---

## Обзор проекта
**SpatialDB** - это высокопроизводительная система для работы с 3D-пространственными данными, использующая NVIDIA PhysX для GPU-ускорения. Проект предназначен для обработки больших объемов геопространственных данных (LiDAR, фотограмметрия, медицинские сканы) с акцентом на производительность и точность расчетов.

### Ключевые возможности:
- 🚀 GPU-ускорение пространственных запросов
- 🌐 Поддержка облачных хранилищ данных
- 🔄 Автоматическое преобразование систем координат
- 📊 Интерактивная 3D-визуализация
- ⚡ Пакетная обработка запросов

---

## Архитектура системы
### Технологический стек:
```
    A[Пользователь] --> B[Python API]
    B --> C[Native Core]
    C --> D[PhysX Engine]
    C --> E[PROJ Library]
    D --> F[CUDA]
    E --> G[Geospatial Data]
    C --> H[Cloud Storage]
```
## Принципы работы:
1. Загрузка данных из различных источников:
* Локальные файлы (LAS, OBJ, PLY)
* Облачные хранилища (S3, GCS)
* Медицинские сканы (DICOM, NIfTI)

2. Преобразование координат в единую систему

3. Оптимизация данных:
* Вокселизация
* Построение BVH

4. Выполнение запросов:
* Raycasting
* Пространственные запросы
* Анализ плотности

5. Визуализация и экспорт результатов

_____

## Установка и настройка
# Требования к системе:
* ОС: Windows 10/11 или Ubuntu 20.04+
* GPU: NVIDIA с поддержкой CUDA (RTX 3060+)
* RAM: 16ГБ+
* ПО:
* * CUDA Toolkit 11.0+
* * PhysX SDK 5.1+
* * Python 3.8+

Установка:
```
# Клонирование репозитория
https://github.com/5c0uT/SpartialDB_core.git
cd spatialdb

# Установка зависимостей
pip install -r requirements.txt

# Сборка нативного модуля (Windows)
.\clean_and_build.ps1

# Сборка нативного модуля (Linux)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```
# Конфигурация:

```
from spatial_db import SpatialConfig

config = SpatialConfig(
    voxel_size=0.1,      # Размер вокселя для дискретизации
    crs="EPSG:3857",     # Целевая система координат
    gpu_device=0,        # Индекс GPU устройства
    cache_size=10,       # Размер кэша (в ГБ)
    streaming_chunk=10_000_000  # Размер чанка для потоковой обработки
)
```

## Основные компоненты
# SpatialDB
Основной класс для работы с базой данных

# Инициализация:
```
from spatial_db import SpatialDB, SpatialConfig

config = SpatialConfig(voxel_size=0.1, crs="EPSG:3857")
db = SpatialDB(config)
```

Методы:

| Метод | Параметры | Возвращаемое значение | Описание |
|-------|-----------|-----------------------|----------|
| `load_las` | `path: str`, `crs: str` | `None` | Загрузка данных LiDAR |
| `load_mesh` | `path: str` | `None` | Загрузка 3D модели |
| `batch_raycast` | `origins: np.ndarray`, `directions: np.ndarray`, `max_dists: np.ndarray` | `List[RayHit]` | Пакетный raycast |
| `profile_terrain` | `start: Tuple[float, float]`, `end: Tuple[float, float]`, `width: float=5.0` | `pd.DataFrame` | Построение профиля местности |
| `query_sphere` | `center: Tuple[float, float, float]`, `radius: float` | `List[int]` | Поиск объектов в сфере |

# SpatialDataset
Класс для работы с пространственными наборами данных

Пример использования:
```
dataset = db.load_dataset("https://storage.example.com/data.las", "las")
simplified = dataset.simplify(tolerance=0.5)
simplified.visualize()
```

# CoordinateConverter
Преобразование систем координат

Пример:
```
converter = CoordinateConverter("EPSG:4326", "EPSG:3857")
x, y, z = 37.617, 55.755, 150
converted = converter.convert(x, y, z)
```

## Работа с API
# Форматы данных
Точки:
```
# Одиночная точка
point = (x, y, z)

# Массив точек
points = np.array([
    [x1, y1, z1],
    [x2, y2, z2],
    ...
])
```
Направления:
```
# Вертикальный луч
direction = (0, 0, -1)

# Массив направлений
directions = np.tile([0, 0, -1], (1000, 1))
```
# RayHit объект
Структура результатов raycast:
```
class RayHit:
    position: Tuple[float, float, float]  # Позиция попадания
    normal: Tuple[float, float, float]     # Нормаль поверхности
    distance: float                       # Дистанция до попадания
    objectID: int                         # ID объекта
```
# Обработка ошибок
```
try:
    hits = db.batch_raycast(origins, directions, max_dists)
except SpatialDBError as e:
    logger.error(f"Raycast failed: {e}")
    # Fallback to CPU implementation
```
# Пакетный raycast
```
import numpy as np
from spatial_db import SpatialDB

db = SpatialDB()
origins = np.random.uniform(0, 100, (10000, 3))
directions = np.tile([0, 0, -1], (10000, 1))
max_dists = np.full(10000, 200.0)

hits = db.batch_raycast(origins, directions, max_dists)
hit_distances = [h.distance for h in hits if h.distance > 0]
```

____

Примеры использования - gis_terrain.py, lidar_medical.py
