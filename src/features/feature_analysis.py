import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[2]
df = pd.read_csv(BASE_DIR / "data/processed/model_dataset.csv")

# Scatter plot
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df,
    x="rating_diff",
    y="kd_diff",
    hue="target",
    alpha=0.7
)
plt.title("Feature Separation: rating_diff vs kd_diff")
plt.show()

plt.figure(figsize=(8,4))
sns.histplot(data=df, x="rating_diff", hue="target", bins=20, kde=True)
plt.title("Distribution of rating_diff by outcome")
plt.show()

plt.figure(figsize=(8,4))
sns.histplot(data=df, x="kd_diff", hue="target", bins=20, kde=True)
plt.title("Distribution of kd_diff by outcome")
plt.show()