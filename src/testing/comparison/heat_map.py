import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("embeddings_all_metrics.csv")

# Filter for cosine similarity only and remove E5-base
df = df[(df["similarity"] == "Cosine")]

# Create a combined label for rows
df["label"] = df["model"] + " (" + df["similarity"] + ")"

# Extract std dev columns and set the model label as index
std_df = df.set_index("label")[["NDCG_std", "F1_std", "MRR_std", "Recall@5_std"]] # excluded time

# Plot: Models on Y-axis, metrics on X-axis
plt.figure(figsize=(10, 6))
sns.heatmap(std_df, annot=True, fmt=".3f", cmap="Oranges", cbar_kws={'label': 'Standard Deviation'})
plt.title("Stability of Cosine Similarity Retrieval (Standard Deviation across Queries)")
plt.ylabel("Embedding Model")
plt.xlabel("Metric")
plt.tight_layout()
plt.savefig("heat_map.png", dpi=300, bbox_inches='tight')
plt.show()
