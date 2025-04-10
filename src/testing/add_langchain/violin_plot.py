import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import MinMaxScaler

mpl.rcParams['font.family'] = 'serif'

# Load CSVs
baseline_df = pd.read_csv("baseline_nli_bert_results.csv")
langchain_df = pd.read_csv("langchain_vector_retrieval_results.csv")

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
    "Baseline+NLI-BERT": "Baseline BERT",
    "LangChain+Vector": "LangChain"
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

plt.title("Metric Distribution Comparison: Baseline vs LangChain", fontsize=14)
plt.ylabel("Score")
plt.xlabel("Metric")
plt.legend(title="Retrieval Approach", loc="upper right")
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.figtext(
    0.5, 0.01,
    "Note: Baseline BERT averages 3.54 ms and 3.73MB, while LangChain averages 77.41 ms and 0.1372MB memory usage.",
    wrap=True,
    horizontalalignment='center',
    fontsize=10,
    style='italic'
)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("violin_plot.png", dpi=300)
plt.show()
