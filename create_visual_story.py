"""
Create the visualizations and summary dataset for the Week 2 storytelling report.
Usage: python create_visual_story.py
Requirements: pandas, numpy, matplotlib, scikit-learn
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

data = load_diabetes(as_frame=True, scaled=True)
df = data.data.copy()
df["disease_progression_1y"] = data.target
target = df["disease_progression_1y"]
corr = df.corr(numeric_only=True)["disease_progression_1y"].drop("disease_progression_1y")

plt.figure(figsize=(8, 5))
plt.hist(target, bins=24, edgecolor="white")
plt.axvline(target.mean(), linestyle="--", label=f"Mean = {target.mean():.1f}")
plt.axvline(target.median(), linestyle=":", label=f"Median = {target.median():.1f}")
plt.title("1. Distribution of one-year progression")
plt.xlabel("Progression measure")
plt.ylabel("Participants")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "01_target_distribution.png", dpi=180)
plt.close()

plt.figure(figsize=(8, 5))
corr.sort_values().plot(kind="barh")
plt.axvline(0, color="black", linewidth=0.8)
plt.title("2. Feature correlations with progression")
plt.xlabel("Pearson correlation")
plt.tight_layout()
plt.savefig(OUT / "02_correlations.png", dpi=180)
plt.close()

bmi_corr = df["bmi"].corr(target)
plt.figure(figsize=(8, 5))
plt.scatter(df["bmi"], target, alpha=0.55)
m, b = __import__("numpy").polyfit(df["bmi"], target, 1)
xs = __import__("numpy").linspace(df["bmi"].min(), df["bmi"].max(), 100)
plt.plot(xs, m * xs + b)
plt.title(f"3. BMI and progression (r = {bmi_corr:.2f})")
plt.xlabel("Standardized BMI feature")
plt.ylabel("Progression measure")
plt.tight_layout()
plt.savefig(OUT / "03_bmi_scatter.png", dpi=180)
plt.close()

bins = pd.qcut(df["bmi"], q=4, duplicates="drop")
groups = df.groupby(bins, observed=False)[target.name]
means, stds = groups.mean(), groups.std()
plt.figure(figsize=(8, 5))
plt.bar([f"Q{i+1}" for i in range(len(means))], means.values, yerr=stds.values, capsize=4)
plt.title("4. Progression across BMI quartiles")
plt.xlabel("BMI quartile")
plt.ylabel("Mean progression ± 1 SD")
plt.tight_layout()
plt.savefig(OUT / "04_bmi_quartiles.png", dpi=180)
plt.close()

selected = ["bmi", "bp", "s5", "s4", "s1", "age", target.name]
corr_matrix = df[selected].corr()
plt.figure(figsize=(8, 6))
plt.imshow(corr_matrix, vmin=-1, vmax=1, aspect="auto")
plt.colorbar(label="Pearson correlation")
plt.xticks(range(len(selected)), selected, rotation=35, ha="right")
plt.yticks(range(len(selected)), selected)
for i in range(len(selected)):
    for j in range(len(selected)):
        plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
plt.title("5. Correlation heatmap")
plt.tight_layout()
plt.savefig(OUT / "05_correlation_heatmap.png", dpi=180)
plt.close()

age_bins = pd.qcut(df["age"], q=4, duplicates="drop")
age_groups = df.groupby(age_bins, observed=False)[target.name]
age_means, age_stds = age_groups.mean(), age_groups.std()
plt.figure(figsize=(8, 5))
plt.errorbar(range(len(age_means)), age_means.values, yerr=age_stds.values, fmt="o-", capsize=4)
plt.xticks(range(len(age_means)), [f"Q{i+1}" for i in range(len(age_means))])
plt.title(f"6. Progression across age quartiles (r = {df['age'].corr(target):.2f})")
plt.xlabel("Age quartile")
plt.ylabel("Mean progression ± 1 SD")
plt.tight_layout()
plt.savefig(OUT / "06_age_quartiles.png", dpi=180)
plt.close()

df.to_csv("diabetes_dataset_with_target.csv", index=False)
print(f"Created 6 figures from {len(df)} observations.")
print("Top positive correlation:", corr.idxmax(), round(corr.max(), 3))
print("BMI correlation:", round(bmi_corr, 3))
