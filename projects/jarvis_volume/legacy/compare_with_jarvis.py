import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jarvis.core.atoms import Atoms as JarvisAtoms
from jarvis.db.figshare import data as jarvis_data
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==========================================
# 1. 加载预测数据并提取标准的 JID (例如 JVASP-1002)
# ==========================================
pred_csv_path = "equilibrium_volume_results.csv"  # 上一步生成的 CSV 文件
df_pred = pd.read_csv(pred_csv_path)

# 强制转换数据类型
df_pred["Literature_output"] = pd.to_numeric(df_pred["Literature_output"], errors="coerce")

# 提取正统的 JARVIS ID（匹配 JVASP-数字）
# 防止文件名如 "JVASP-1002_mac" 影响与 JARVIS 数据库的匹配
df_pred["jid"] = df_pred["ID"].apply(
    lambda x: re.search(r"JVASP-\d+", str(x)).group(0) if re.search(r"JVASP-\d+", str(x)) else None
)

# 剔除无效行
df_pred = df_pred.dropna(subset=["jid", "Literature_output"])
print(f"成功读取并提取 {len(df_pred)} 条模拟预测数据。")

# ==========================================
# 2. 获取 JARVIS-DFT 官方基准数据 (Ground Truth)
# ==========================================
print("正在加载 JARVIS-DFT (dft_3d) 官方数据...")
dft_3d = jarvis_data("dft_3d")

# 提取 JARVIS 中对应的基准体积 (optB88-vdW 泛函计算的体积)
dft_dict = []
for entry in dft_3d:
    # 将 atoms 字典转为 Atoms 对象，自动从晶格矩阵计算体积
    atoms_obj = JarvisAtoms.from_dict(entry["atoms"])
    dft_dict.append(
        {
            "jid": entry["jid"],
            "dft_volume": atoms_obj.volume,  # 准确获取晶胞平衡体积 (Å³)
        }
    )

df_dft = pd.DataFrame(dft_dict)

# ==========================================
# 3. 合并数据并计算误差指标
# ==========================================
merged = pd.merge(df_pred, df_dft, on="jid", how="inner")

if merged.empty:
    print("错误：未找到任何能匹配上的 JID，请检查 CSV 中的 ID 格式！")
else:
    # 提取数组
    v_pred = merged["Literature_output"].values
    v_true = merged["dft_volume"].values

    # 计算 MAE, RMSE 和 Relative Error (%)
    mae = mean_absolute_error(v_true, v_pred)
    rmse = np.sqrt(mean_squared_error(v_true, v_pred))
    mape = np.mean(np.abs((v_pred - v_true) / v_true)) * 100

    print("\n" + "=" * 40)
    print("      MACE vs JARVIS-DFT 评估结果      ")
    print("=" * 40)
    print(f"匹配材料数量 (N)   : {len(merged)}")
    print(f"体积平均绝对误差 MAE : {mae:.4f} Å³")
    print(f"均方根误差 RMSE      : {rmse:.4f} Å³")
    print(f"平均相对百分比误差   : {mape:.2f} %")
    print("=" * 40)

    # 保存对比详表
    merged[["jid", "ID", "Test_output", "dft_volume"]].to_csv("comparison_detail.csv", index=False)
    print("对齐后的详细对比数据已保存至 'comparison_detail.csv'")

    # ==========================================
    # 4. 绘制对比散点图 (Parity Plot)
    # ==========================================
    plt.figure(figsize=(6, 6))
    plt.scatter(v_true, v_pred, alpha=0.7, color="#1f77b4", edgecolors="k", label="Materials")

    # 绘制 y = x 理想对角线
    max_val = max(v_true.max(), v_pred.max()) * 1.05
    min_val = min(v_true.min(), v_pred.min()) * 0.95
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal (y = x)")

    plt.xlabel("JARVIS-DFT Volume (Å³)", fontsize=12)
    plt.ylabel("MACE Predicted Volume (Å³)", fontsize=12)
    plt.title(f"Equilibrium Volume Comparison\nMAE = {mae:.3f} Å³", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("volume_parity_plot.png", dpi=300)
    print("对比图表已保存至 'volume_parity_plot.png'")
