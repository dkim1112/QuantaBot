import pandas as pd
from scipy.stats import ttest_rel, shapiro, wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV files
baseline_df = pd.read_csv("baseline_mpnet_results.csv")
langchain_df = pd.read_csv("langchain_mpnet_results.csv")

# Merge on query_id
merged_df = pd.merge(
    baseline_df,
    langchain_df,
    on="query_id",
    suffixes=("_baseline", "_langchain")
)

# Compute differences
f1_diff = merged_df["F1_langchain"] - merged_df["F1_baseline"]
ndcg_diff = merged_df["NDCG_langchain"] - merged_df["NDCG_baseline"]
mrr_diff = merged_df["MRR_langchain"] - merged_df["MRR_baseline"]
r5_diff = merged_df["Recall@5_langchain"] - merged_df["Recall@5_baseline"]

# Shapiro-Wilk test for normality on differences - condition check for paired t-test
print("Shapiro-Wilk Test for Normality of Paired Differences:")
f1_shapiro = shapiro(f1_diff).pvalue
ndcg_shapiro = shapiro(ndcg_diff).pvalue
mrr_shapiro = shapiro(mrr_diff).pvalue
r5_shapiro = shapiro(r5_diff).pvalue

print(f"F1:       {f1_shapiro:.5f} {'(NORMAL)' if f1_shapiro > 0.05 else '(NON-NORMAL, use Wilcoxon)'}")
print(f"NDCG:     {ndcg_shapiro:.5f} {'(NORMAL)' if ndcg_shapiro > 0.05 else '(NON-NORMAL, use Wilcoxon)'}")
print(f"MRR:      {mrr_shapiro:.5f} {'(NORMAL)' if mrr_shapiro > 0.05 else '(NON-NORMAL, use Wilcoxon)'}")
print(f"Recall@5: {r5_shapiro:.5f} {'(NORMAL)' if r5_shapiro > 0.05 else '(NON-NORMAL, use Wilcoxon)'}\n")

# Statistical Tests
print("Statistical Test Results:")
print("=" * 50)

# F1
if f1_shapiro > 0.05:
    f1_test = ttest_rel(merged_df["F1_langchain"], merged_df["F1_baseline"])
    f1_p = f1_test.pvalue
    test_type = "Paired t-test"
else:
    f1_test = wilcoxon(f1_diff)
    f1_p = f1_test.pvalue
    test_type = "Wilcoxon"

significance = "SIGNIFICANT" if f1_p < 0.05 else "NOT SIGNIFICANT"
print(f"F1 ({test_type}): p = {f1_p:.5f} | {significance}")
print(f"  Baseline: {merged_df['F1_baseline'].mean():.4f} → LangChain: {merged_df['F1_langchain'].mean():.4f}")

# NDCG
if ndcg_shapiro > 0.05:
    ndcg_test = ttest_rel(merged_df["NDCG_langchain"], merged_df["NDCG_baseline"])
    ndcg_p = ndcg_test.pvalue
    test_type = "Paired t-test"
else:
    ndcg_test = wilcoxon(ndcg_diff)
    ndcg_p = ndcg_test.pvalue
    test_type = "Wilcoxon"

significance = "SIGNIFICANT" if ndcg_p < 0.05 else "NOT SIGNIFICANT"
print(f"NDCG ({test_type}): p = {ndcg_p:.5f} | {significance}")
print(f"  Baseline: {merged_df['NDCG_baseline'].mean():.4f} → LangChain: {merged_df['NDCG_langchain'].mean():.4f}")

# MRR
if mrr_shapiro > 0.05:
    mrr_test = ttest_rel(merged_df["MRR_langchain"], merged_df["MRR_baseline"])
    mrr_p = mrr_test.pvalue
    test_type = "Paired t-test"
else:
    mrr_test = wilcoxon(mrr_diff)
    mrr_p = mrr_test.pvalue
    test_type = "Wilcoxon"

significance = "SIGNIFICANT" if mrr_p < 0.05 else "NOT SIGNIFICANT"
print(f"MRR ({test_type}): p = {mrr_p:.5f} | {significance}")
print(f"  Baseline: {merged_df['MRR_baseline'].mean():.4f} → LangChain: {merged_df['MRR_langchain'].mean():.4f}")

# Recall@5
if r5_shapiro > 0.05:
    r5_test = ttest_rel(merged_df["Recall@5_langchain"], merged_df["Recall@5_baseline"])
    r5_p = r5_test.pvalue
    test_type = "Paired t-test"
else:
    r5_test = wilcoxon(r5_diff)
    r5_p = r5_test.pvalue
    test_type = "Wilcoxon"

significance = "SIGNIFICANT" if r5_p < 0.05 else "NOT SIGNIFICANT"
print(f"Recall@5 ({test_type}): p = {r5_p:.5f} | {significance}")
print(f"  Baseline: {merged_df['Recall@5_baseline'].mean():.4f} → LangChain: {merged_df['Recall@5_langchain'].mean():.4f}")

print("\n" + "=" * 50)

# CONCLUSIONS
print("\nCONCLUSIONS:")
print("Baseline (MPNet) vs LangChain+MPNet:")

significant_improvements = []
if f1_p < 0.05: significant_improvements.append("F1")
if ndcg_p < 0.05: significant_improvements.append("NDCG")
if mrr_p < 0.05: significant_improvements.append("MRR")
if r5_p < 0.05: significant_improvements.append("Recall@5")

if significant_improvements:
    print(f"LangChain shows statistically significant improvements in: {', '.join(significant_improvements)}")
else:
    print("No statistically significant improvements found")

print(f"📊 Effect sizes: F1 improvement = {((merged_df['F1_langchain'].mean() / merged_df['F1_baseline'].mean()) - 1) * 100:+.1f}%")

# Showing Bar Plot
sns.set_theme(style="whitegrid")
metrics = ["F1", "NDCG", "MRR", "Recall@5"]
baseline_means = [
    merged_df["F1_baseline"].mean(),
    merged_df["NDCG_baseline"].mean(),
    merged_df["MRR_baseline"].mean(),
    merged_df["Recall@5_baseline"].mean()
]
langchain_means = [
    merged_df["F1_langchain"].mean(),
    merged_df["NDCG_langchain"].mean(),
    merged_df["MRR_langchain"].mean(),
    merged_df["Recall@5_langchain"].mean()
]

x = range(len(metrics))
bar_width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x, baseline_means, bar_width, label='Baseline MPNet', alpha=0.8)
ax.bar([p + bar_width for p in x], langchain_means, bar_width, label='LangChain+MPNet', alpha=0.8)

ax.set_xlabel('Metric')
ax.set_ylabel('Mean Score')
ax.set_title('Baseline MPNet vs. LangChain+MPNet Performance')
ax.set_xticks([p + bar_width / 2 for p in x])
ax.set_xticklabels(metrics)
ax.legend()

plt.tight_layout()
plt.show()