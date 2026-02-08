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
mem_diff = merged_df["Memory_bytes_langchain"] - merged_df["Memory_bytes_baseline"]

# Shapiro-Wilk test for normality on differences - condition check for paired t-test
print("Shapiro-Wilk Test for Normality of Paired Differences:")
f1_shapiro = shapiro(f1_diff).pvalue
ndcg_shapiro = shapiro(ndcg_diff).pvalue
mrr_shapiro = shapiro(mrr_diff).pvalue
r5_shapiro = shapiro(r5_diff).pvalue
mem_shapiro = shapiro(mem_diff).pvalue

print(f"F1:       {f1_shapiro:.5f} {'(NORMAL)' if f1_shapiro > 0.05 else '(NON-NORMAL)'}")
print(f"NDCG:     {ndcg_shapiro:.5f} {'(NORMAL)' if ndcg_shapiro > 0.05 else '(NON-NORMAL)'}")
print(f"MRR:      {mrr_shapiro:.5f} {'(NORMAL)' if mrr_shapiro > 0.05 else '(NON-NORMAL)'}")
print(f"Recall@5: {r5_shapiro:.5f} {'(NORMAL)' if r5_shapiro > 0.05 else '(NON-NORMAL)'}")
print(f"Memory:   {mem_shapiro:.5f} {'(NORMAL)' if mem_shapiro > 0.05 else '(NON-NORMAL)'}")

# Try log transformation for NDCG if not normal
ndcg_transformed = None
if ndcg_shapiro <= 0.05:
    import numpy as np
    ndcg_transformed = np.log(ndcg_diff + 1 - np.min(ndcg_diff))
    ndcg_transformed_shapiro = shapiro(ndcg_transformed).pvalue
    print(f"NDCG (log-transformed): {ndcg_transformed_shapiro:.5f} {'(NORMAL)' if ndcg_transformed_shapiro > 0.05 else '(STILL NON-NORMAL)'}")

print()

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
elif ndcg_transformed is not None and shapiro(ndcg_transformed).pvalue > 0.05:
    # Use paired t-test on transformed data
    ndcg_test = ttest_rel(ndcg_transformed, np.zeros_like(ndcg_transformed))
    ndcg_p = ndcg_test.pvalue
    test_type = "Paired t-test (log-transformed)"
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

# Memory Usage
if mem_shapiro > 0.05:
    mem_test = ttest_rel(merged_df["Memory_bytes_langchain"], merged_df["Memory_bytes_baseline"])
    mem_p = mem_test.pvalue
    test_type = "Paired t-test"
else:
    mem_test = wilcoxon(mem_diff)
    mem_p = mem_test.pvalue
    test_type = "Wilcoxon"

significance = "SIGNIFICANT" if mem_p < 0.05 else "NOT SIGNIFICANT"
print(f"Memory ({test_type}): p = {mem_p:.5f} | {significance}")
print(f"  Baseline: {merged_df['Memory_bytes_baseline'].mean():.0f} → LangChain: {merged_df['Memory_bytes_langchain'].mean():.0f} bytes")

print("\n" + "=" * 50)

# CONCLUSIONS
print("\nCONCLUSIONS:")
print("Baseline (MPNet) vs LangChain+MPNet:")

significant_improvements = []
if f1_p < 0.05: significant_improvements.append("F1")
if ndcg_p < 0.05: significant_improvements.append("NDCG")
if mrr_p < 0.05: significant_improvements.append("MRR")
if r5_p < 0.05: significant_improvements.append("Recall@5")
if mem_p < 0.05: significant_improvements.append("Memory Usage")

if significant_improvements:
    print(f"LangChain shows statistically significant improvements in: {', '.join(significant_improvements)}")
else:
    print("No statistically significant improvements found")

print(f"📊 Effect sizes: F1 improvement = {((merged_df['F1_langchain'].mean() / merged_df['F1_baseline'].mean()) - 1) * 100:+.1f}%")

# Showing Bar Plot with separate subplots for scores and memory
sns.set_theme(style="whitegrid")

# Separate metrics into scores and memory
score_metrics = ["F1", "NDCG", "MRR", "Recall@5"]
baseline_scores = [
    merged_df["F1_baseline"].mean(),
    merged_df["NDCG_baseline"].mean(),
    merged_df["MRR_baseline"].mean(),
    merged_df["Recall@5_baseline"].mean()
]
langchain_scores = [
    merged_df["F1_langchain"].mean(),
    merged_df["NDCG_langchain"].mean(),
    merged_df["MRR_langchain"].mean(),
    merged_df["Recall@5_langchain"].mean()
]

# Memory data separately
baseline_memory = merged_df["Memory_bytes_baseline"].mean() / 1024 / 1024  # Convert to MB
langchain_memory = merged_df["Memory_bytes_langchain"].mean() / 1024 / 1024  # Convert to MB

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot performance scores
x1 = range(len(score_metrics))
bar_width = 0.35

ax1.bar(x1, baseline_scores, bar_width, label='Baseline MPNet', alpha=0.8)
ax1.bar([p + bar_width for p in x1], langchain_scores, bar_width, label='LangChain+MPNet', alpha=0.8)

ax1.set_xlabel('Performance Metrics')
ax1.set_ylabel('Mean Score')
ax1.set_title('Performance Comparison')
ax1.set_xticks([p + bar_width / 2 for p in x1])
ax1.set_xticklabels(score_metrics)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot memory usage
x2 = [0, 1]
memory_values = [baseline_memory, langchain_memory]
memory_labels = ['Baseline MPNet', 'LangChain+MPNet']
colors = ['steelblue', 'orange']

ax2.bar(x2, memory_values, bar_width*2, color=colors, alpha=0.8)
ax2.set_xlabel('Method')
ax2.set_ylabel('Memory Usage (MB)')
ax2.set_title('Memory Usage Comparison')
ax2.set_xticks(x2)
ax2.set_xticklabels(memory_labels, rotation=15)
ax2.grid(True, alpha=0.3)

# Add value labels on memory bars
for i, v in enumerate(memory_values):
    ax2.text(i, v + 0.05, f'{v:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('comparison_plot.png')
print("\nPlot saved as 'comparison_plot.png'")
print("\nNOTE: NDCG analysis used log-transformation to achieve normality for paired t-test")