import os
import sys


package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if package_root not in sys.path:
    sys.path.insert(0, package_root)

from spatial_db.core import SpatialConfig, SpatialDB, native


def main():
    db = SpatialDB(SpatialConfig(crs="EPSG:3857"))
    converter = native.CoordinateConverter("EPSG:4326", "EPSG:3857")

    lon, lat, alt = 37.6173, 55.7558, 0.0
    x, y, z = converter.convert(lon, lat, alt)

    print("Source CRS: EPSG:4326")
    print("Target CRS: EPSG:3857")
    print(f"Input : lon={lon}, lat={lat}, alt={alt}")
    print(f"Output: x={x:.3f}, y={y:.3f}, z={z:.3f}")

    profile = db.profile_terrain((37.6173, 55.7558), (37.6273, 55.7658))
    print(f"profile rows: {len(profile)}")
    profile.to_csv("proj_check_profile.csv", index=False)
    print("Saved proj_check_profile.csv")


if __name__ == "__main__":
    main()
