import sys
import os
import logging
import numpy as np
import matplotlib.pyplot as plt

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


def setup_logging():
    """Настройка системы логирования"""
    logger = logging.getLogger("gis_terrain")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def analyze_visibility(db: SpatialDB, tower_positions, logger, resolution=1.0):
    """Анализ зон видимости для вышек связи"""
    results = {}
    for i, (x, y, height) in enumerate(tower_positions):
        logger.info(f"Calculating visibility for tower {i + 1}/{len(tower_positions)}")

        # Генерация лучей во всех направлениях
        angles = np.linspace(0, 2 * np.pi, 36)  # 36 лучей для демонстрации
        origins = np.tile([x, y, height], (len(angles), 1))
        directions = np.column_stack([
            np.cos(angles),
            np.sin(angles),
            np.zeros_like(angles) - 0.1  # Немного вниз
        ])
        max_dists = np.full(len(angles), 5000.0)

        try:
            hits = db.batch_raycast(origins, directions, max_dists)
        except Exception as e:
            logger.error(f"Raycast failed: {e}")
            continue

        visible_points = []
        for j, hit in enumerate(hits):
            if hasattr(hit, 'distance') and hit.distance > 0:
                angle = angles[j]
                distance = hit.distance

                # Проверяем структуру hit.position
                if hasattr(hit.position, 'z'):
                    z_coord = hit.position.z
                elif isinstance(hit.position, (list, tuple)) and len(hit.position) >= 3:
                    z_coord = hit.position[2]
                else:
                    z_coord = 0

                visible_points.append([
                    x + distance * np.cos(angle),
                    y + distance * np.sin(angle),
                    z_coord
                ])

        results[f"tower_{i}"] = {
            "position": (x, y, height),
            "visible_points": np.array(visible_points) if visible_points else np.empty((0, 3))
        }

    return results


def visualize_visibility(results, output_file="visibility_map.png"):
    """Визуализация результатов анализа видимости"""
    if not results:
        print("No visibility results to visualize")
        return

    plt.figure(figsize=(12, 10))

    # Создаем карту местности
    x = np.linspace(0, 1000, 100)
    y = np.linspace(0, 1000, 100)
    xx, yy = np.meshgrid(x, y)
    zz = 50 * np.sin(xx / 100) * np.cos(yy / 100) + np.random.normal(0, 5, (100, 100))

    # Топографическая карта
    plt.contourf(xx, yy, zz, 20, cmap='terrain', alpha=0.6)
    plt.colorbar(label='Elevation')

    # Рисуем карту видимости
    for tower_id, data in results.items():
        pos = data["position"]
        points = data["visible_points"]

        # Позиция вышки
        plt.scatter(pos[0], pos[1], s=100, c='red', marker='^', label=f"Tower at ({pos[0]}, {pos[1]})")

        # Зона видимости (если есть точки)
        if points.size > 0:
            plt.scatter(points[:, 0], points[:, 1], s=2, alpha=0.7)

    # Линии связи между вышками
    positions = [data["position"] for data in results.values()]
    xs, ys, _ = zip(*positions)
    plt.plot(xs, ys, 'b--', linewidth=1, alpha=0.5)

    plt.title("Communication Tower Visibility Analysis")
    plt.xlabel("X Coordinate (m)")
    plt.ylabel("Y Coordinate (m)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(output_file, dpi=150)
    print(f"Visibility map saved to {output_file}")


def run_terrain_analysis():
    """Основная функция анализа местности"""
    logger = setup_logging()
    logger.info("Starting terrain analysis")

    try:
        config = SpatialConfig(
            voxel_size=1.0,
            crs="EPSG:32637",
            gpu_device=0
        )
        db = SpatialDB(config)
        logger.info("SpatialDB initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize SpatialDB: {e}")
        return

    # Пример вышек связи
    tower_positions = [
        (500, 500, 50),  # Центральная вышка
        (200, 800, 30),  # Северо-западная
        (800, 200, 40),  # Юго-восточная
        (300, 300, 25)  # Юго-западная
    ]

    logger.info("Analyzing visibility...")
    visibility_results = analyze_visibility(db, tower_positions, logger)

    logger.info("Visualizing results...")
    visualize_visibility(visibility_results)

    logger.info("Creating terrain profile...")
    try:
        profile = db.profile_terrain(
            start=(500, 500),
            end=(800, 800)
        )

        if profile is not None:
            # Сохранение профиля
            profile.to_csv("terrain_profile.csv", index=False)
            logger.info("Terrain profile saved to terrain_profile.csv")

            # Визуализация профиля
            plt.figure(figsize=(12, 6))
            plt.plot(profile['distance'], profile['elevation'], 'b-', linewidth=2)
            plt.fill_between(profile['distance'], profile['elevation'].min(),
                             profile['elevation'], color='blue', alpha=0.1)

            # Добавляем маркеры вышек
            for i, (x, y, z) in enumerate(tower_positions):
                # Вычисляем расстояние до вышки
                dist = np.sqrt((x - 500) ** 2 + (y - 500) ** 2)
                plt.plot(dist, z, 'ro', markersize=8)
                plt.text(dist, z + 5, f"Tower {i + 1}", fontsize=10)

            plt.title("Terrain Elevation Profile with Communication Towers")
            plt.xlabel("Distance from Start Point (m)")
            plt.ylabel("Elevation (m)")
            plt.grid(alpha=0.3)
            plt.savefig("terrain_profile.png", dpi=150)
            logger.info("Terrain profile visualization saved")
        else:
            logger.warning("Terrain profiling returned no data")
    except Exception as e:
        logger.error(f"Terrain profiling failed: {e}")

    logger.info("Terrain analysis completed")


if __name__ == "__main__":
    run_terrain_analysis()