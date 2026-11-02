from pathlib import Path
import numpy as np
import pandas as pd
import tifffile

ROOT = Path(
    "flame3_severity dataset/"
    "FLAME 3 CV Dataset (Sycan Marsh)"
)

results = []

for category in ["Fire", "No Fire"]:

    folder = ROOT / category / "Thermal" / "Celsius TIFF"

    for file in sorted(folder.glob("*.TIFF")):

        img = tifffile.imread(file)

        results.append({
            "class": category,
            "filename": file.stem,
            "min": np.nanmin(img),
            "max": np.nanmax(img),
            "mean": np.nanmean(img),
            "p95": np.nanpercentile(img, 95),
            "hot_100": np.mean(img >= 100),
            "hot_150": np.mean(img >= 150),
            "hot_200": np.mean(img >= 200),
            "finite_pixels": np.isfinite(img).mean()
        })

df = pd.DataFrame(results)

print("\nDataset summary:")
print(df.groupby("class").agg({
    "min": "mean",
    "max": "mean",
    "mean": "mean",
    "p95": "mean",
    "hot_100": "mean",
    "hot_150": "mean",
    "hot_200": "mean",
    "finite_pixels": "mean"
}).round(4))

df.to_csv("thermal_summary.csv", index=False)

print("\nSaved thermal_summary.csv")