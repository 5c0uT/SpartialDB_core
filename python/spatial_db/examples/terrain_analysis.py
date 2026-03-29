import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if package_root not in sys.path:
    sys.path.insert(0, package_root)

from spatial_db.core import SpatialConfig, SpatialDB


def analyze_terrain():
    config = SpatialConfig(voxel_size=1.0, crs="EPSG:32637", gpu_device=0)
    db = SpatialDB(config)
    _ = db  # Keeps example focused on public initialization path

    print("Calculating slopes...")
    grid_size = 100
    x = np.linspace(0, 1000, grid_size)
    y = np.linspace(0, 1000, grid_size)
    xx, yy = np.meshgrid(x, y)
    zz = 50 * np.sin(xx / 100) * np.cos(yy / 100) + np.random.normal(0, 5, (grid_size, grid_size))

    dx, dy = np.gradient(zz)
    slope = np.degrees(np.arctan(np.sqrt(dx ** 2 + dy ** 2)))

    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(zz, cmap="terrain", origin="lower")
    plt.colorbar(label="Elevation (m)")
    plt.title("Elevation Map")

    plt.subplot(132)
    plt.imshow(slope, cmap="hot", origin="lower", vmin=0, vmax=45)
    plt.colorbar(label="Slope (degrees)")
    plt.title("Slope Analysis")

    plt.subplot(133)
    plt.hist(slope.flatten(), bins=30, color="skyblue", edgecolor="black")
    plt.xlabel("Slope (degrees)")
    plt.ylabel("Frequency")
    plt.title("Slope Distribution")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("terrain_analysis.png", dpi=150)
    print("Analysis saved to terrain_analysis.png")

    return {
        "elevation": zz,
        "slope": slope,
        "mean_elevation": np.mean(zz),
        "max_elevation": np.max(zz),
        "mean_slope": np.mean(slope),
        "max_slope": np.max(slope),
    }


def elevation_profile_analysis():
    config = SpatialConfig(crs="EPSG:3857")
    db = SpatialDB(config)

    print("Creating elevation profile...")
    profile_df = db.profile_terrain(start=(37.6173, 55.7558), end=(37.6273, 55.7658))
    profile_df["slope"] = np.degrees(
        np.arctan(np.gradient(profile_df["elevation"].to_numpy(), profile_df["distance"].to_numpy()))
    )

    plt.figure(figsize=(12, 8))

    plt.subplot(211)
    plt.plot(profile_df["distance"], profile_df["elevation"], "b-", linewidth=1.5)
    plt.fill_between(
        profile_df["distance"],
        profile_df["elevation"].min() - 10,
        profile_df["elevation"],
        color="b",
        alpha=0.1,
    )
    plt.xlabel("Distance (m)")
    plt.ylabel("Elevation (m)")
    plt.title("Terrain Elevation Profile")
    plt.grid(alpha=0.3)

    plt.subplot(212)
    plt.plot(profile_df["distance"], profile_df["slope"], "r-", linewidth=1.5)
    plt.xlabel("Distance (m)")
    plt.ylabel("Slope (degrees)")
    plt.title("Slope Profile")
    plt.grid(alpha=0.3)
    plt.ylim(-30, 30)

    plt.tight_layout()
    plt.savefig("elevation_profile.png", dpi=150)
    profile_df.to_csv("elevation_profile.csv", index=False)
    print("Profile saved to elevation_profile.png and elevation_profile.csv")

    return profile_df


if __name__ == "__main__":
    analyze_terrain()
    elevation_profile_analysis()
