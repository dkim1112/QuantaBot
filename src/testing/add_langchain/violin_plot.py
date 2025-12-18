import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import MinMaxScaler

mpl.rcParams['font.family'] = 'serif'

# Load CSVs
baseline_df = pd.read_csv("baseline_mpnet_results.csv")
langchain_df = pd.read_csv("langchain_mpnet_results.csv")

# Combine both datasets
combined_df = pd.concat([baseline_df, langchain_df], ignore_index=True)

# Melt for seaborn
melted_df = pd.melt(
    combined_df,
    id_vars=["method"],
    value_vars=["NDCG", "F1", "MRR", "Recall@5"],
    var_name="Metric",
    value_name="Score"
)

# Rename methods title for clarity
melted_df["method"] = melted_df["method"].replace({
    "Baseline+MPNet": "Baseline MPNet",
    "LangChain+MPNet": "LangChain+MPNet"
})

# Violin plot with IQR and touching halves
plt.figure(figsize=(12, 6))
sns.violinplot(
    data=melted_df,
    x="Metric",
    y="Score",
    hue="method",
    split=True, # Split violins — touching shape
    inner="quart", # Show median and IQR
    linewidth=0.4,
)

plt.title("Metric Distribution Comparison: Baseline MPNet vs LangChain+MPNet", fontsize=14)
plt.ylabel("Score")
plt.xlabel("Metric")
plt.legend(title="Retrieval Approach", loc="upper right")
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.figtext(
    0.5, 0.01,
    "Note: Updated results using MPNet model (best performing from embedding comparison study)",
    wrap=True,
    horizontalalignment='center',
    fontsize=10,
    style='italic'
)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("violin_plot.png", dpi=300)
plt.show()
