import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import ListedColormap

mpl.rcParams['font.family'] = 'serif'

df = pd.read_csv("embeddings_all_metrics.csv")

# Filter for cosine similarity only and remove E5-base
df = df[(df["similarity"] == "Cosine")]

# Create a combined label for rows
df["label"] = df["model"] + " (" + df["similarity"] + ")"

# Extract std dev columns and set the model label as index
std_df = df.set_index("label")[["NDCG_std", "F1_std", "MRR_std", "Recall@5_std"]] # excluded time

normalized_df = (std_df - std_df.min()) / (std_df.max() - std_df.min())

original = cm.get_cmap("YlOrBr")
flattened = ListedColormap(original(np.linspace(0.3, 0.4, 512))) # smoother color transition

# Plot: Models on Y-axis, metrics on X-axis
plt.figure(figsize=(10, 6))
sns.heatmap(normalized_df, annot=std_df, fmt=".3f", cmap=flattened, cbar_kws={'label': 'Normalized Std Dev'})
plt.title("Stability of Retrieval Performance (Standard Deviation across Queries)")
plt.ylabel("Embedding Model")
plt.xlabel("Metric")
plt.tight_layout()
plt.savefig("heat_map.png", dpi=300, bbox_inches='tight')
plt.show()