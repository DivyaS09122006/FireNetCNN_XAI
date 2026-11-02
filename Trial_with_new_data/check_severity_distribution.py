import pandas as pd

df = pd.read_csv("thermal_summary.csv")
fire = df[df["class"] == "Fire"]

features = ["max", "mean", "p95", "hot_100", "hot_150", "hot_200"]

print("\nFIRE IMAGE FEATURE DISTRIBUTIONS")
print(fire[features].describe(
    percentiles=[0.10, 0.25, 0.50, 0.75, 0.90]
).round(4))

print("\nImages with no pixels above 100 C:")
print((fire["hot_100"] == 0).sum())

print("\nImages with pixels above 100 C:")
print((fire["hot_100"] > 0).sum())