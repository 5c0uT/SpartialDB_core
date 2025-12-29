from ..core import SpatialDB, SpatialConfig, SpatialDataset
import matplotlib.pyplot as plt


def visualize_terrain():
    config = SpatialConfig(crs="EPSG:32637")
    db = SpatialDB(config)
    dataset = db.load_dataset("https://example.com/terrain.las", "LAS")

    # 3D визуализация
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    sample = dataset.data.sample(10000)
    sc = ax.scatter(
        sample['x'], sample['y'], sample['z'],
        c=sample['z'], cmap='terrain', s=1
    )

    plt.colorbar(sc, label='Elevation (m)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Terrain Visualization')
    plt.show()


def visualize_heatmap():
    config = SpatialConfig()
    db = SpatialDB(config)

    bbox = (0, 0, 0, 1000, 1000, 100)
    heatmap = db.create_density_heatmap(bbox, resolution=10)

    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap, cmap='hot', interpolation='bilinear')
    plt.colorbar(label='Point Density')
    plt.title('Point Cloud Density Heatmap')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.show()


if __name__ == "__main__":
    visualize_terrain()
    visualize_heatmap()