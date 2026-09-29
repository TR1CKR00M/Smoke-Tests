# %% [markdown]
# # 科研绘图：平滑对比图（200 fs 滑动均值平滑 + 噪声背景）

# %%
import glob
import os
import matplotlib.pyplot as plt
import pandas as pd

output_dir = "results/figures"

# 设置科研绘图基本样式
plt.style.use("seaborn-v0_8-paper" if "seaborn-v0_8-paper" in plt.style.available else "default")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 300,
        "mathtext.fontset": "stix",
    }
)

# %% [markdown]
# ## 1. 参数配置与数据读取

# %%
# 获取文件列表（假设正好对应 4 个模型的数据文件）
csv_files = sorted(glob.glob("results/raw/*.csv"))[:4]

# 预设科研高对比度调色板（支持 4 个模型清晰区分）
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # 蓝、橘、绿、红

# 滑动窗口设置（窗口大小为 200 fs）
WINDOW_SIZE = 200  # 若 X 轴步长不是 1 fs，需设置为：200 // (X轴采样间隔)

# %% [markdown]
# ## 2. 批量读取并绘制带背景噪声的平滑折线图

# %%
fig, (ax1, ax2) = plt.subplots(
    nrows=2, ncols=1, figsize=(5.5, 7), sharex=True
)

x_label, y1_label, y2_label = "Time (fs)", "Y1", "Y2"

for i, file_path in enumerate(csv_files):
    # 解析模型名称（文件名作为 Model 标识）
    model_name = os.path.splitext(os.path.basename(file_path))[0][3:-9]
    color = colors[i % len(colors)]

    # 读取数据
    df = pd.read_csv(file_path)

    col_x = df.iloc[:, 0]
    col_y1 = df.iloc[:, 1]
    col_y2 = df.iloc[:, 2]

    # 获取坐标轴名称
    if i == 0:
        x_label = df.columns[0]
        y1_label = df.columns[1]
        y2_label = df.columns[2]

    # ----------------- 计算 200 fs 窗口滑动均值 -----------------
    # min_periods=1 保证数据开头不产生缺失值 NaN
    y1_smooth = col_y1.rolling(window=WINDOW_SIZE, min_periods=1, center=True).mean()
    y2_smooth = col_y2.rolling(window=WINDOW_SIZE, min_periods=1, center=True).mean()

    # ----------------- 绘制上图 (第 2 列 vs 第 1 列) -----------------
    # 1. 原始数据（含噪背景：粗线/细线均可，透明度 0.15，不加 label 避免图例重复）
    ax1.plot(col_x, col_y1, color=color, alpha=0.15, linewidth=1.0)
    # 2. 200fs 平滑曲线（较细实线，透明度 1.0，用于展示趋势与稳态）
    ax1.plot(
        col_x,
        y1_smooth,
        color=color,
        linewidth=1.2,
        linestyle="-",
        label=model_name,
    )

    # ----------------- 绘制下图 (第 3 列 vs 第 1 列) -----------------
    # 1. 原始数据背景
    ax2.plot(col_x, col_y2, color=color, alpha=0.15, linewidth=1.0)
    # 2. 200fs 平滑曲线
    ax2.plot(
        col_x,
        y2_smooth,
        color=color,
        linewidth=1.2,
        linestyle="-",
        label=model_name,
    )

# ----------------- 图表全局细节修饰 -----------------
# 上图设置
ax1.set_ylabel(y1_label)
ax1.set_title("(a) " + y1_label, loc="left", fontweight="bold")
ax1.grid(True, linestyle="--", alpha=0.4)
ax1.tick_params(direction="in", top=True, right=True)
ax1.legend(frameon=True, edgecolor="gray", fontsize=8.5, loc="best")

# 下图设置
ax2.set_xlabel(x_label)
ax2.set_ylabel(y2_label)
ax2.set_title("(b) " + y2_label, loc="left", fontweight="bold")
ax2.grid(True, linestyle="--", alpha=0.4)
ax2.tick_params(direction="in", top=True, right=True)
ax2.legend(frameon=True, edgecolor="gray", fontsize=8.5, loc="best")

plt.tight_layout()

# 保存高质量图表
plt.savefig(os.path.join(output_dir, "multi_file_comparison_alpha_adjusted.png"), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(output_dir, "multi_file_comparison_alpha_adjusted.svg"), format="svg", bbox_inches="tight")

plt.show()