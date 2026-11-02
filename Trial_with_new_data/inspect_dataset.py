from pathlib import Path
import cv2
import tifffile
import numpy as np

ROOT = Path("flame3_severity dataset/FLAME 3 CV Dataset (Sycan Marsh)")

for category in ["Fire", "No Fire"]:
    print(f"\n{'='*45}")
    print(category)
    print("="*45)

    rgb_dir = ROOT / category / "RGB" / "Corrected FOV"
    thermal_dir = ROOT / category / "Thermal" / "Celsius TIFF"

    if not rgb_dir.exists() or not thermal_dir.exists():
        print("Expected folders not found!")
        print("RGB:", rgb_dir.exists(), "Thermal:", thermal_dir.exists())
        continue

    rgb_files = {p.stem: p for p in rgb_dir.glob("*.JPG")}
    thermal_files = {p.stem: p for p in thermal_dir.glob("*.TIFF")}

    common = sorted(set(rgb_files) & set(thermal_files))

    print("Corrected RGB images:", len(rgb_files))
    print("Thermal TIFF images:", len(thermal_files))
    print("Matching filenames:", len(common))
    print("RGB without thermal:", len(set(rgb_files) - set(thermal_files)))
    print("Thermal without RGB:", len(set(thermal_files) - set(rgb_files)))

    for stem in common[:3]:
        rgb = cv2.imread(str(rgb_files[stem]))
        thermal = tifffile.imread(str(thermal_files[stem]))

        print(f"\nSample: {stem}")
        print("RGB shape:", rgb.shape)
        print("Thermal shape:", thermal.shape)
        print("Thermal dtype:", thermal.dtype)
        print("Temperature min:", np.nanmin(thermal))
        print("Temperature max:", np.nanmax(thermal))
        print("Temperature mean:", np.nanmean(thermal))
        print("Temperature p95:", np.nanpercentile(thermal, 95))