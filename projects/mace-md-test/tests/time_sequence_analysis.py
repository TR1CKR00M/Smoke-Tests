# %% [markdown]
# # 脚本 2：模型间差异的时间序列演化与收敛性分析

# %%
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from statsmodels.tsa.stattools import adfuller

output_dir = "results/figures"
output_path = os.path.join(output_dir, "time_sequence_analysis.png")

# 1. 读取数据并对齐时间轴
csv_files = sorted(glob.glob("results/raw/*.csv"))[:4]
df_dict = {}

for file_path in csv_files:
    model_name = os.path.splitext(os.path.basename(file_path))[0][3:-9]
    df_dict[model_name] = pd.read_csv(file_path)

models = list(df_dict.keys())
time_axis = df_dict[models[0]].iloc[:, 0].values  # 假设时间轴一致


# 2. 指数衰减拟合函数
def exp_decay(t, A, tau, C):
    return A * np.exp(-t / tau) + C


# 3. 计算两两模型间的绝对差值序列，并拟合收敛趋势
fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

# 选用滑窗平滑展示残差趋势
WINDOW = 200

print("=== 时间序列平稳性 (ADF 检验) 与 收敛拟合结果 ===")

# 以第 0 个模型为基准（基准模型），或两两组合对比
ref_model = models[0]

for i in range(1, len(models)):
    target_model = models[i]
    pair_label = f"{target_model} vs {ref_model}"

    # 实时绝对差值
    y_diff = np.abs(
        df_dict[target_model].iloc[:, 1].values
        - df_dict[ref_model].iloc[:, 1].values
    )

    # (1) ADF 检验 (检验后半段稳态区是否平稳)
    adf_result = adfuller(y_diff[int(len(y_diff) * 0.3) :])
    print(
        f"[{pair_label}] ADF p-value: {adf_result[1]:.4e} -> {'已平稳 (Converged)' if adf_result[1] < 0.05 else '未平稳 (Drifting)'}"
    )

    # (2) 滑动平均平滑曲线
    y_diff_smooth = (
        pd.Series(y_diff)
        .rolling(window=WINDOW, min_periods=1, center=True)
        .mean()
    )

    # (3) 指数衰减拟合提取收敛极限 C
    try:
        popt, _ = curve_fit(
            exp_decay,
            time_axis,
            y_diff_smooth,
            p0=[y_diff_smooth.iloc[0], time_axis[-1] / 5, y_diff_smooth.iloc[-1]],
            bounds=(0, [np.inf, np.inf, np.inf]),
        )
        fit_A, fit_tau, fit_C = popt
        print(
            f"   拟合极限残差 C: {fit_C:.4f}, 时间常数 tau: {fit_tau:.2f} fs"
        )
    except Exception:
        fit_C, fit_tau = np.nan, np.nan
        print("   拟合未收敛")

    # 绘制实时残差与平滑线 (上图)
    axes[0].plot(
        time_axis,
        y_diff_smooth,
        linewidth=1.5,
        label=f"{pair_label} (C={fit_C:.3f})",
    )

    # 绘制累计均值差演化 (下图：观察全局均值收敛状态)
    cum_diff = np.cumsum(y_diff) / (np.arange(len(y_diff)) + 1)
    axes[1].plot(time_axis, cum_diff, linewidth=1.5, label=pair_label)

# ----------------- 样式修饰 -----------------
axes[0].set_ylabel(r"Absolute Difference $|\Delta Y(t)|$")
axes[0].set_title(
    "(a) Dynamic Pairwise Difference & Decay Fit", loc="left", fontweight="bold"
)
axes[0].grid(True, linestyle="--", alpha=0.4)
axes[0].legend(loc="upper right", fontsize=8.5)

axes[1].set_xlabel(df_dict[models[0]].columns[0])  # Time 标签
axes[1].set_ylabel(r"Cumulative Mean Difference")
axes[1].set_title(
    "(b) Trajectory-Cumulative Difference", loc="left", fontweight="bold"
)
axes[1].grid(True, linestyle="--", alpha=0.4)
axes[1].legend(loc="upper right", fontsize=8.5)

plt.tight_layout()
plt.savefig(output_path, dpi=600, bbox_inches="tight")
plt.show()