import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

output_dir = "results/figures"
output_path = os.path.join(output_dir, "multi_file_comparison.png")

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
csv_files = ["results/raw/md_medium-mpa-0_600K_log.csv","results/raw/md_medium-0b3_600K_log.csv"]
#csv_files = sorted(glob.glob("results/raw/*.csv"))

for f in csv_files:
    print(" -", os.path.basename(f))

colors = plt.cm.tab10.colors  # 学术常用 Tab10 调色板
markers = ["o", "s", "^", "D", "v", "p", "*", "X"]

fig, (ax1, ax2) = plt.subplots(
    nrows=2, ncols=1, figsize=(5, 6.5), sharex=True
)

# 用于保存列名做坐标轴标签（默认取第一个文件的列名）
x_label, y1_label, y2_label = "X", "Y1", "Y2"

for i, file_path in enumerate(csv_files):
    # 提取文件名（不含扩展名）作为图例标签
    filename = os.path.splitext(os.path.basename(file_path))[0]

    # 读取 CSV
    df = pd.read_csv(file_path)

    # 提取第 1、2、3 列
    col_x = df.iloc[:, 0]
    col_y1 = df.iloc[:, 2]
    col_y2 = df.iloc[:, 3]

    # 更新坐标轴标签
    if i == 0:
        x_label = df.columns[0]
        y1_label = df.columns[2]
        y2_label = df.columns[3]

    # 选择当前文件的颜色与数据点形状
    color = colors[i % len(colors)]
    marker = markers[i % len(markers)]

    # 控制数据点标记密度（避免密集点重叠）
    markevery = max(1, len(col_x) // 10)

    ax1.plot(
        col_x,
        col_y1,
        label=filename,
        color=color,
        linestyle="-",
        linewidth=1.5,
        marker=marker,
        markersize=4.5,
        markevery=markevery,
    )

    ax2.plot(
        col_x,
        col_y2,
        label=filename,
        color=color,
        linestyle="-",
        linewidth=1.5,
        marker=marker,
        markersize=4.5,
        markevery=markevery,
    )

# ----------------- 上图样式设置 -----------------
ax1.set_ylabel(y1_label)
ax1.set_title("(a) " + y1_label, loc="left", fontweight="bold")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.tick_params(direction="in", top=True, right=True)
# 上图添加图例
ax1.legend(frameon=True, edgecolor="gray", fontsize=8.5, loc="best")

# ----------------- 下图样式设置 -----------------
ax2.set_xlabel(x_label)
ax2.set_ylabel(y2_label)
ax2.set_title("(b) " + y2_label, loc="left", fontweight="bold")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.tick_params(direction="in", top=True, right=True)
# 下图添加图例（如果两图数据集一致且标签相同，也可只保留上图或统一放置）
ax2.legend(frameon=True, edgecolor="gray", fontsize=8.5, loc="best")

# 调整布局
plt.tight_layout()

# 保存高清图像
plt.savefig(output_path, dpi=600, bbox_inches="tight")

plt.show()