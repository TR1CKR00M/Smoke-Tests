# %% [markdown]
# # 科研绘图：平滑对比图（200 fs 滑动均值平滑 + 噪声背景）

#usage: python tests/plot_comparison_beta.py group_1.txt

# %%
import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

# ----------------- 修改点 1：解析命令行输入的 txt 文件名 -----------------
# 设置基础目录路径
METADATA_DIR = "data/metadata"

parser = argparse.ArgumentParser(description="科研平滑对比图绘制程序")
parser.add_argument(
    "txt_name",
    type=str,
    help="data/metadata 目录下的 txt 文件名（如 file_list.txt 或 file_list）",
)

# 兼容在 Jupyter Notebook/Interactive 模式与终端命令行模式下的运行
if "ipykernel" in sys.modules:
    # 如果在 Notebook/Interactive 环境下运行，设置默认文件名方便测试
    args = parser.parse_args(["file_list.txt"])
else:
    # 在命令行终端下正常解析参数
    args = parser.parse_args()

# 确保文件名包含 .txt 后缀
txt_filename = args.txt_name if args.txt_name.endswith(".txt") else f"{args.txt_name}.txt"
txt_file_path = os.path.join(METADATA_DIR, txt_filename)

# 校验文件是否存在
if not os.path.isfile(txt_file_path):
    raise FileNotFoundError(f"未找到指定的配置文件: {txt_file_path}")

# 提取 txt 文件主名（不含扩展名），作为生成图片的独特标识
txt_basename = os.path.splitext(txt_filename)[0]

# ----------------- 修改点 2：输出图片路径拼接 -----------------
output_dir = "results/figures"
os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
output_path = os.path.join(output_dir, f"multi_file_comparison_{txt_basename}.png")

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
# 从指定的 txt 文件中读取 csv 文件列表（忽略空行和前后空格）
with open(txt_file_path, "r", encoding="utf-8") as f:
    csv_files = [line.strip() for line in f if line.strip()]

# 预设科研高对比度调色板（支持多个模型清晰区分）
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

# 滑动窗口设置（窗口大小为 200 fs）
WINDOW_SIZE = 200  # 若 X 轴步长不是 1 fs，需设置为：200 // (X轴采样间隔)

# %% [markdown]
# ## 2. 批量读取并绘制带背景噪声的平滑折线图

# %%
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(5.5, 7), sharex=True)

x_label, y1_label, y2_label = "Time (fs)", "Y1", "Y2"

for i, file_path in enumerate(csv_files):
    # 解析模型名称（文件名作为 Model 标识）
    model_name = os.path.splitext(os.path.basename(file_path))[0]
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
    y1_smooth = col_y1.rolling(window=WINDOW_SIZE, min_periods=1, center=True).mean()
    y2_smooth = col_y2.rolling(window=WINDOW_SIZE, min_periods=1, center=True).mean()

    # ----------------- 绘制上图 (第 2 列 vs 第 1 列) -----------------
    # 1. 原始数据（含噪背景）
    ax1.plot(col_x, col_y1, color=color, alpha=0.15, linewidth=1.0)
    # 2. 200fs 平滑曲线
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
plt.savefig(output_path, dpi=600, bbox_inches="tight")

plt.show()