import sys
import os
import time
import numpy as np

# Добавляем путь к корню пакета
package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if package_root not in sys.path:
    sys.path.insert(0, package_root)

from spatial_db.core import SpatialDB, SpatialConfig


def run_benchmark():
    print("=" * 50)
    print("SpatialDB Benchmark")
    print("=" * 50)

    config = SpatialConfig(
        voxel_size=0.1,
        crs="EPSG:3857",
        gpu_device=0
    )

    try:
        db = SpatialDB(config)
        print("SpatialDB initialized successfully")
    except Exception as e:
        print(f"Failed to initialize SpatialDB: {e}")
        return

    # Генерация тестовых данных
    ray_count = 1000
    print(f"\nGenerating {ray_count} test rays...")

    origins = np.random.uniform(-100, 100, (ray_count, 3))
    directions = np.tile([0, 0, -1], (ray_count, 1))
    max_dists = np.full(ray_count, 200.0)

    print("Running raycast benchmark...")
    start_time = time.time()

    try:
        results = db.batch_raycast(origins, directions, max_dists)
        success = True
    except Exception as e:
        print(f"Raycast failed: {e}")
        success = False
        results = []

    duration = time.time() - start_time

    if success:
        # Анализ результатов
        hit_distances = [r.distance for r in results if r.distance > 0]
        hit_ratio = len(hit_distances) / ray_count if ray_count > 0 else 0

        print("\nBenchmark Results:")
        print(f"  Rays processed: {ray_count}")
        print(f"  Total time: {duration:.4f} sec")

        if duration > 0:
            print(f"  Throughput: {ray_count / duration:.2f} rays/sec")

        print(f"  Hit ratio: {hit_ratio:.2%}")

        if hit_distances:
            print(f"  Average hit distance: {np.mean(hit_distances):.2f} m")
            print(f"  Min distance: {np.min(hit_distances):.2f} m")
            print(f"  Max distance: {np.max(hit_distances):.2f} m")

    return results


if __name__ == "__main__":
    run_benchmark()