import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams['font.family'] = 'serif'

# Load CSV
df = pd.read_csv("embeddings_all_metrics.csv")

# Merge model + similarity for labeling (optional)
df["method"] = df["model"] + " (" + df["similarity"] + ")"

# Metrics to plot
metrics = ['NDCG_mean', 'F1_mean', 'MRR_mean', 'Recall@5_mean', 'Time_ms_mean']

# Plot grouped bars for L2 vs Cosine per model
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 12))
axes = axes.flatten()

for i, metric in enumerate(metrics):
    ax = axes[i+1]
    models = df["model"].unique()
    x = np.arange(len(models))
    width = 0.35

    # Group by similarity
    l2_vals = df[df["similarity"] == "L2"][metric].values
    cos_vals = df[df["similarity"] == "Cosine"][metric].values

    ax.bar(x - width/2, l2_vals, width, label="L2", color="steelblue")
    ax.bar(x + width/2, cos_vals, width, label="Cosine", color="orange")

    ax.set_title(metric.replace("_", " "), fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45)
    ax.set_ylabel(metric)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

key_ax = axes[0]
key_ax.axis("off")
key_ax.text(0, 1.0, "Metric Key", fontsize=15, fontweight="bold")
key_ax.text(0, 0.85, "F1-score:     Balance Precision and Recall", fontsize=12)
key_ax.text(0, 0.70, "NDCG:        Ranked Relevance Quality", fontsize=12)
key_ax.text(0, 0.55, "Recall@5:   Top-5 Result Coverage", fontsize=12)
key_ax.text(0, 0.40, "MRR:          First Relevant Answer Ranking", fontsize=12)
key_ax.text(0, 0.25, "Time:          Processing Speed and Efficiency", fontsize=12)

plt.tight_layout()
plt.savefig("bar_graph.png", dpi=300, bbox_inches='tight')
plt.show()