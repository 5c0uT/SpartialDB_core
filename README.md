# 🧠 SpatialDB - GPU-ускоренная пространственная база данных

> Высокопроизводительная система для работы с 3D-пространственными данными, использующая **NVIDIA PhysX** для GPU-ускорения.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)
[![Tests: 52/52 Passing (100%)](https://img.shields.io/badge/Tests-52%2F52%20Passing%20(100%25)-green)]()
[![Status: Alpha](https://img.shields.io/badge/Status-Alpha-orange)]()

---

## 📋 Оглавление

- [🎯 Возможности](#-возможности)
- [✅ Что работает](#-что-работает)
- [❌ Известные проблемы](#-известные-проблемы)
- [🚀 Установка](#-установка)
- [💻 Использование](#-использование)
- [🧪 Тестирование](#-тестирование)
- [📊 Примеры](#-примеры)
- [🛣️ Roadmap](#️-roadmap)
- [📝 Лицензия](#-лицензия)

---

## 🎯 Возможности

### ⚡ GPU-ускорение
- Использует **NVIDIA PhysX 5.x** для GPU-вычислений
- Поддержка CUDA на GeForce/RTX/Quadro видеокартах
- Асинхронная обработка больших наборов данных

### 🗺️ Пространственные запросы
- **Raycast queries** - пересечение лучей с объектами
- **Sphere queries** - сферические поиски соседей
- **Terrain profiling** - анализ профилей рельефа
- **Density heatmaps** - тепловые карты плотности

### 📊 Поддерживаемые форматы
- **LiDAR**: LAS/LAZ точечные облака
- **Фотограмметрия**: 3D модели и облака точек
- **Медицина**: DICOM сканы и 3D томография

### 🔧 Интеграция
- Python 3.11+ с pybind11 биндингами
- C++ ядро для максимальной производительности
- Кроссплатформенность (Windows, Linux в progress)

---

## ✅ Что работает

### 🟢 Полностью функциональные компоненты

| Компонент | Статус | Тесты | Примечание |
|-----------|--------|-------|-----------|
| **Инициализация SpatialDB** | ✅ | 5/5 | Конфигурация, потоки |
| **Raycast операции** | ✅ | 5/5 | Одиночные/групповые запросы |
| **Сферические запросы** | ✅ | 4/4 | Разные радиусы, граничные случаи |
| **Анализ рельефа** | ✅ | 4/4 | Профили, тепловые карты |
| **Интеграционные тесты** | ✅ | 3/3 | Сценарии работы |
| **Производительность** | ✅ | 3/3 | Пропускная способность, память |
| **Параметризованные тесты** | ✅ | 11/11 | Варианты запросов и конфигов |

**ИТОГО: 52 теста PASSED** ✅

---

## ❌ Известные проблемы

### 🟡 Текущие ограничения

- **Windows-first**: основной сценарий сборки и CI сейчас ориентирован на Windows + MSVC
- **PhysX runtime required**: нативный модуль требует локально доступные PhysX DLL
- **PROJ/CURL feature split**: PROJ интегрирован, CURL часть пока не используется
- **Demo-level query results**: примеры и тесты используют стабильный API, но сложные сценовые запросы пока демонстрационные

### 📊 Статистика тестирования

```
============ Test Summary =============
✅ PASSED:   52 / 52 (100%)
❌ FAILED:   0 / 52 (0%)

Среднее время выполнения: 2.36s
```

---

## 🚀 Установка

### Требования

- Python 3.11+
- Windows 10/11 (Visual Studio Build Tools или MSVC)
- NVIDIA GPU с CUDA поддержкой
- CMake 3.20+
- Conda environment `spatial_env`
- NVIDIA PhysX 5.x SDK

### Быстрая установка

```bash
# 1. Клонировать репозиторий
git clone https://github.com/5c0uT/SpartialDB_core.git
cd SpartialDB_core

# 2. Полный запуск: зависимости, сборка, smoke import, тесты
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

### Ручной запуск

```bash
# Установить зависимости
pip install -r requirements.txt

# Собрать и проверить без повторной установки пакетов
powershell -ExecutionPolicy Bypass -File .\run.ps1 -SkipPip -SkipVcpkg
```

### Из исходников

```bash
# Клонировать и установить
git clone https://github.com/5c0uT/SpartialDB_core.git
cd SpartialDB_core
pip install -e .
```

---

## 💻 Использование

### Базовый пример

```python
from spatial_db import SpatialDB, SpatialConfig

# Создать конфигурацию
config = SpatialConfig(
    voxel_size=0.1,
    cache_size=1000,
    gpu_device=0
)

# Инициализировать БД
db = SpatialDB(config)

# Raycast запрос
origin = [0.0, 0.0, 0.0]
direction = [1.0, 0.0, 0.0]
hit = db.raycast(origin, direction, max_dist=100.0)

# Сферический запрос
center = [10.0, 10.0, 10.0]
results = db.query_sphere(center, radius=5.0)
```

### Terrain profiling

```python
from spatial_db import SpatialDB, SpatialConfig

db = SpatialDB(SpatialConfig(crs="EPSG:3857"))
profile = db.profile_terrain(
    start=(37.6173, 55.7558),
    end=(37.6273, 55.7658)
)
```

### Heatmap

```python
heatmap = db.create_density_heatmap(
    bbox=[-100, -100, -100, 100, 100, 100],
    resolution=5.0
)
```

### API quick reference

```python
from spatial_db import SpatialDB, SpatialConfig, SpatialDataset

db = SpatialDB(SpatialConfig())
hit = db.raycast((0, 0, 0), (0, 0, 1))
hits = db.batch_raycast(origins, directions, max_dists)
sphere = db.query_sphere((0, 0, 0), 10.0)
profile = db.profile_terrain((37.6173, 55.7558), (37.6273, 55.7658))
heatmap = db.create_density_heatmap((-10, -10, -1, 10, 10, 1), resolution=1.0)
dataset = SpatialDataset("synthetic", "las")
```

---

## 🧪 Тестирование

### Запустить все тесты

```bash
conda run -n spatial_env python -m pytest tests -q
```

### Запустить только smoke тесты

```bash
conda run -n spatial_env python -m pytest tests -v -m smoke
```

### С отчетом о покрытии

```bash
conda run -n spatial_env python -m pytest tests -v --cov=spatial_db --cov-report=html
```

### Параметризованные тесты

```bash
# Тестировать разные радиусы
conda run -n spatial_env python -m pytest tests -v -k "sphere_query_radii"

# Тестировать разные размеры вокселей
conda run -n spatial_env python -m pytest tests -v -k "config_voxel_sizes"
```

---

## 📊 Примеры

### 1. Анализ облака точек LiDAR

```python
# python/spatial_db/examples/api_smoke.py
# Быстрая проверка API и сохранение CSV
```

### 2. Обработка медицинских данных

```python
# lidar_medical.py
# Синтетический LiDAR + медицинский pipeline с экспортом файлов
```

### 3. Бенчмарки производительности

```python
# python/spatial_db/examples/benchmark.py
# Smoke benchmark для batch_raycast
```

### 4. Визуализация

```python
# python/spatial_db/examples/proj_check.py
# Проверка CoordinateConverter и terrain profile через PROJ
```

---

## 🛣️ Roadmap

### v0.1.0 (Текущая версия) ✅
- [x] Основная архитектура SpatialDB
- [x] Интеграция PhysX для GPU
- [x] Python биндинги pybind11
- [x] Базовые пространственные запросы
- [x] Тестовая база (52/52 тестов)

### v0.2.0 (Q1 2026) 🚧
- [x] Исправить все test failures
- [ ] Полная обработка ошибок
- [ ] Linux поддержка
- [x] Документация API
- [x] CI/CD pipeline (GitHub Actions)

### v0.3.0 (Q2 2026) 📋
- [ ] Оптимизация GPU памяти
- [ ] Поддержка Multi-GPU
- [ ] Advanced queries (k-NN, range queries)
- [ ] PyPI пакет

### v1.0.0 (Q3 2026) 🎯
- [ ] Production-ready
- [ ] Полное покрытие тестами (95%+)
- [x] Оптимизированные примеры
- [ ] Лучшие практики документации

---

## 📁 Структура проекта

```
SpartialDB_core/
├── include/              # C++ headers
│   ├── SpatialDB.hpp
│   ├── PhysXCore.hpp
│   └── ...
├── src/                  # C++ implementation
│   ├── SpatialDB.cpp
│   ├── pybind_module.cpp
│   └── ...
├── python/
│   └── spatial_db/       # Python package
│       ├── core.py       # Main API
│       └── examples/     # Example scripts
├── tests/                # Test suite
│   ├── test_spatialdb.py
│   └── conftest.py
├── CMakeLists.txt        # Build configuration
├── run.ps1               # One-shot build/test entrypoint
├── pytest.ini            # Test configuration
└── README.md             # This file
```

---

## 🔧 Разработка

### Построение из исходников

```bash
# Windows (recommended)
powershell -ExecutionPolicy Bypass -File .\run.ps1 -SkipPip -SkipVcpkg
```

### Запуск тестов после сборки

```bash
conda run -n spatial_env python -m pytest tests -v --tb=short
```

### Контрибьютинг

1. Fork репозиторий
2. Создать feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit изменений (`git commit -m 'Add some AmazingFeature'`)
4. Push в branch (`git push origin feature/AmazingFeature`)
5. Открыть Pull Request

---

## 📞 Поддержка

- **Issues**: [GitHub Issues](https://github.com/5c0uT/SpartialDB_core/issues)
- **Discussions**: [GitHub Discussions](https://github.com/5c0uT/SpartialDB_core/discussions)
- **Documentation**: [GitHub Wiki](https://github.com/5c0uT/SpartialDB_core/wiki)

---

## 📝 Лицензия

MIT License - см. [LICENSE.md](LICENSE.md)

---

## 🙏 Благодарности

- NVIDIA за PhysX и CUDA
- pybind11 за Python/C++ интеграцию
- pytest за тестовую фреймворк

---

## 👨‍💻 Автор

**@5c0uT** - Разработка и архитектура

---

**Последний обновлен**: 2026-03-30  
**Статус проекта**: Alpha (активная разработка)  
**Python версия**: 3.11+  
**License**: MIT

---

> 💡 **Совет**: Начните с [примеров](python/spatial_db/examples) для быстрого ознакомления!
