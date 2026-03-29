import os
import sys

import pandas as pd


package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if package_root not in sys.path:
    sys.path.insert(0, package_root)

from spatial_db.core import SpatialConfig, SpatialDB


def main():
    db = SpatialDB(SpatialConfig())

    hit = db.raycast((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    sphere = db.query_sphere((0.0, 0.0, 0.0), 10.0)
    profile = db.profile_terrain((0.0, 0.0), (100.0, 100.0))
    heatmap = db.create_density_heatmap((-10.0, -10.0, -1.0, 10.0, 10.0, 1.0), resolution=1.0)

    print("SpatialDB initialized:", db.is_initialized)
    print("Raycast distance:", hit.distance)
    print("Sphere result count:", len(sphere))
    print("Profile rows:", len(profile))
    print("Heatmap shape:", heatmap.shape)

    profile.to_csv("api_smoke_profile.csv", index=False)
    pd.DataFrame({"heatmap_sum": [float(heatmap.sum())]}).to_csv("api_smoke_stats.csv", index=False)
    print("Saved api_smoke_profile.csv and api_smoke_stats.csv")


if __name__ == "__main__":
    main()
