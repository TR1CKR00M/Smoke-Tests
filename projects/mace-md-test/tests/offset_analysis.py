# %% [markdown]
# # 脚本 1：模型稳态数据的系统性偏差与差异显著性分析

# %%
import glob
import os

import numpy as np
import pandas as pd
from IPython.display import display
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# 1. 加载数据并截取稳态区间（假设前 20% 为弛豫阶段）
csv_files = sorted(glob.glob("results/raw/*.csv"))[:4]
EQUILIBRIUM_START_RATIO = 0.2  # 截去前 20%

data_y1 = {}  # 存储第 2 列稳态数据
for file_path in csv_files:
    model_name = os.path.splitext(os.path.basename(file_path))[0]
    df = pd.read_csv(file_path)

    n_samples = len(df)
    start_idx = int(n_samples * EQUILIBRIUM_START_RATIO)

    # 提取稳态段第 2 列数据 (Y1)
    data_y1[model_name] = df.iloc[start_idx:, 1].values

# 2. 基础统计量计算
stats_summary = []
for model, values in data_y1.items():
    stats_summary.append(
        {
            "Model": model,
            "Mean": np.mean(values),
            "Std": np.std(values, ddof=1),
            "Median": np.median(values),
            "IQR": stats.iqr(values),
        }
    )

df_stats = pd.DataFrame(stats_summary)
print("=== 1. 稳态基础统计量 ===")
display(df_stats)

# 3. 总体显著性检验 (ANOVA / Kruskal-Wallis)
all_values = list(data_y1.values())
f_val, p_val = stats.f_oneway(*all_values)
kw_val, kw_p_val = stats.kruskal(*all_values)

print("\n=== 2. 总体差异检验 ===")
print(f"One-way ANOVA: F = {f_val:.4f}, p-value = {p_val:.4e}")
print(f"Kruskal-Wallis: H = {kw_val:.4f}, p-value = {kw_p_val:.4e}")

# 4. 事后检验 (Tukey's HSD 两两对比) & Cohen's d 效应量计算
combined_data = []
labels = []
for model, values in data_y1.items():
    combined_data.extend(values)
    labels.extend([model] * len(values))

tukey = pairwise_tukeyhsd(endog=combined_data, groups=labels, alpha=0.05)
print("\n=== 3. 两两模型事后对比 (Tukey HSD) ===")
print(tukey)


# 计算 Cohen's d 函数
def calc_cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / s_pooled


models = list(data_y1.keys())
print("\n=== 4. 两两模型间系统性 Offset (Cohen's d 效应量) ===")
for i in range(len(models)):
    for j in range(i + 1, len(models)):
        m1, m2 = models[i], models[j]
        d_val = calc_cohens_d(data_y1[m1], data_y1[m2])
        mean_diff=np.mean(data_y1[m1]) - np.mean(data_y1[m2])
        print(
            f"{m1} vs {m2} -> Offset (Mean Diff): {mean_diff:.4f}, Cohen's d: {d_val:.4f}"
        )

