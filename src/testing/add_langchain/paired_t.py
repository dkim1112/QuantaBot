import pandas as pd
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV files
baseline_df = pd.read_csv("baseline_nli_bert_results.csv")
langchain_df = pd.read_csv("langchain_vector_retrieval_results.csv")

# Merge on query_id
merged_df = pd.merge(
    baseline_df,
    langchain_df,
    on="query_id",
    suffixes=("_baseline", "_langchain")
)

# Run paired t-tests
f1_ttest = ttest_rel(merged_df["F1_langchain"], merged_df["F1_baseline"])
ndcg_ttest = ttest_rel(merged_df["NDCG_langchain"], merged_df["NDCG_baseline"])
mrr_ttest = ttest_rel(merged_df["MRR_langchain"], merged_df["MRR_baseline"])

# Show result of p-value
print("Paired T-Test Results:\n")
print(f"F1:    p = {f1_ttest.pvalue:.5f} | baseline = {merged_df['F1_baseline'].mean():.4f} | langchain = {merged_df['F1_langchain'].mean():.4f}")
print(f"NDCG:  p = {ndcg_ttest.pvalue:.5f} | baseline = {merged_df['NDCG_baseline'].mean():.4f} | langchain = {merged_df['NDCG_langchain'].mean():.4f}")
print(f"MRR:   p = {mrr_ttest.pvalue:.5f} | baseline = {merged_df['MRR_baseline'].mean():.4f} | langchain = {merged_df['MRR_langchain'].mean():.4f}")

# INSIGHTS
# We've done: Baseline (NLI-Roberta) vs Addition of LangChain (VectorDB)
# These p-values are below 0.05, meaning the improvements in F1, NDCG, and MRR by using LangChain's retriever over manual cosine search are statistically significant.
# Even without an LLM, LangChain's integration, chunking, and possibly better retriever logic make a difference.

# Showing Bar Plot
sns.set_theme(style="whitegrid")
metrics = ["F1", "NDCG", "MRR"]
baseline_means = [
    merged_df["F1_baseline"].mean(),
    merged_df["NDCG_baseline"].mean(),
    merged_df["MRR_baseline"].mean()
]
langchain_means = [
    merged_df["F1_langchain"].mean(),
    merged_df["NDCG_langchain"].mean(),
    merged_df["MRR_langchain"].mean()
]

x = range(len(metrics))
bar_width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x, baseline_means, bar_width, label='Baseline', alpha=0.8)
ax.bar([p + bar_width for p in x], langchain_means, bar_width, label='LangChain (Vector)', alpha=0.8)

ax.set_xlabel('Metric')
ax.set_ylabel('Mean Score')
ax.set_title('Baseline vs. LangChain (Vector Retriever) Performance')
ax.set_xticks([p + bar_width / 2 for p in x])
ax.set_xticklabels(metrics)
ax.legend()

plt.tight_layout()
plt.show()