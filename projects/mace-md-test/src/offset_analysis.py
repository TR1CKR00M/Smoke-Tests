'''
Usage:
  python tests/offset_analysis.py -f <FILE1> <FILE2> [OPTIONS]

Examples:
  # 1. 默认分析（压力变量，自动保存至 ./plots，并弹出交互界面）：
  python tests/offset_analysis.py -f results/raw/md_medium-mpa-0_600K_log.csv results/raw/md_medium_600K_log.csv

  # 2. 分析温度变量、指定输出目录，并开启无界面（静默）保存：
  python tests/offset_analysis.py \
      --files results/raw/md_medium-mpa-0_600K_log.csv results/raw/md_medium_600K_log.csv \
      --var temperature \
      --save-dir results/figures \
      --no-show

  # 3. 调整平衡态截断比例 (Burn-in) 为 30%：
  python tests/offset_analysis.py \
      -f results/raw/md_medium-mpa-0_600K_log.csv results/raw/md_medium_600K_log.csv \
      --burn-in 0.30 \
      --var temperature \
      --save-dir results/figures \
      --no-show

Options:
  -f, --files FILE1 FILE2    指定需要对比的两个 MD CSV 数据文件路径 (必填)
  --var {pressure,temperature}
                             指定分析的目标物理量 (默认: pressure)
  --burn-in FLOAT            裁剪数据前段平衡态 Burn-in 的比例 (默认: 0.20，即前 20%)
  --save-dir PATH            指定输出诊断图的保存路径 (默认: ./plots)
  --no-show                  静默运行模式，仅保存图片，不弹出 Matplotlib 绘图窗口
'''

import os
import argparse
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图风格
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 1. 核心统计工具函数
# ==========================================

def calc_autocorr_time(x: np.ndarray, max_lag: int = None) -> tuple[float, np.ndarray]:
    """使用 FFT 计算时间序列的积分自相关时间 tau_int (Sokal 窗口法)"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if max_lag is None:
        max_lag = n // 4

    x_centered = x - np.mean(x)

    f = np.fft.fft(x_centered, n=2 * n)
    acf = np.fft.ifft(f * np.conj(f)).real[:n]
    if acf[0] == 0:
        return 0.5, acf
    acf /= acf[0]

    tau = 0.5
    for k in range(1, max_lag):
        tau += acf[k]
        if k >= 5 * tau:
            break

    return max(0.5, tau), acf


def compute_block_averaging(x: np.ndarray, min_blocks: int = 10) -> tuple[np.ndarray, np.ndarray, float]:
    """
    计算块平均标准误（Block-Averaged Standard Errors）

    参数:
        x: 一维数值序列 (平稳段)
        min_blocks: 允许的最小块数量 (低于此数量停止增加块大小, 保证统计样本量)

    返回:
        block_sizes: 块大小 M 的数组
        block_se: 对应块大小下的标准误 SE(M)
        plateau_se: 估计的平台期标准误 (Plateau SE)
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    max_block_size = max(1, n // min_blocks)

    # 采用对数间隔或线性间隔采样块大小
    block_sizes = np.unique(np.int_([2 ** i for i in range(0, int(np.log2(max_block_size)) + 1)]))
    block_se = []

    for m in block_sizes:
        b = n // m  # 完整的块数量
        if b < min_blocks:
            break

        # 截断数据以整除块大小
        x_truncated = x[:b * m]
        # 重塑数组并计算每块的均值
        blocks = x_truncated.reshape(b, m)
        block_means = np.mean(blocks, axis=1)

        # 块均值的标准误 SE(M) = std(block_means) / sqrt(B)
        se_m = np.std(block_means, ddof=1) / np.sqrt(b)
        block_se.append(se_m)

    block_sizes = block_sizes[:len(block_se)]
    block_se = np.array(block_se)

    # 估计平台值 Plateau SE: 取后 30% 块大小下的 SE 均值
    num_tail = max(1, int(len(block_se) * 0.3))
    plateau_se = float(np.mean(block_se[-num_tail:]))

    return block_sizes, block_se, plateau_se


def welch_ttest_neff(mean1: float, var1: float, neff1: float,
                     mean2: float, var2: float, neff2: float) -> dict:
    """考虑有效样本量 N_eff 修正后的 Welch's t 检验"""
    se1_sq = var1 / neff1
    se2_sq = var2 / neff2
    se_diff = np.sqrt(se1_sq + se2_sq)

    t_stat = (mean1 - mean2) / se_diff
    df = (se1_sq + se2_sq) ** 2 / ((se1_sq ** 2 / (neff1 - 1)) + (se2_sq ** 2 / (neff2 - 1)))
    p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=df))

    return {
        "t_stat": t_stat,
        "p_value": p_val,
        "df": df,
        "is_significant": p_val < 0.05
    }


# ==========================================
# 2. 数据加载与分析类
# ==========================================

def load_md_csv(filepath: str, col_mapping: dict) -> pd.DataFrame:
    """读取 CSV 并根据映射重命名列"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"未找到指定的 CSV 文件: {filepath}")

    df = pd.read_csv(filepath)

    missing_cols = [src for src in col_mapping.keys() if src not in df.columns]
    if missing_cols:
        raise KeyError(f"文件 '{filepath}' 中缺少以下映射指定的列: {missing_cols}")

    return df[list(col_mapping.keys())].rename(columns=col_mapping)


class MDAnalyzer:
    def __init__(self):
        self.raw_data = {}
        self.trimmed_data = {}
        self.stats_summary = {}
        self.block_avg_results = {}

    def add_model_data(self, model_name: str, df: pd.DataFrame):
        for req_col in ["time", "temperature", "pressure"]:
            if req_col not in df.columns:
                raise ValueError(f"模型 '{model_name}' 的数据缺少列: {req_col}")
        self.raw_data[model_name] = df.sort_values(by="time").reset_index(drop=True)

    def remove_burn_in(self, burn_in_ratio: float = 0.20):
        for name, df in self.raw_data.items():
            t_min, t_max = df["time"].min(), df["time"].max()
            cutoff = t_min + burn_in_ratio * (t_max - t_min)
            trimmed = df[df["time"] >= cutoff].reset_index(drop=True)
            self.trimmed_data[name] = trimmed
            print(f"[{name}] Cutoff t >= {cutoff:.2f} | 保留帧数: {len(trimmed)} / {len(df)}")

    def compute_statistics(self, target_variable: str = "pressure") -> pd.DataFrame:
        """扩展计算：对比 朴素 SEM、tau_int 修正 SEM 与 块平均 Plateau SEM"""
        results = []
        for name, df in self.trimmed_data.items():
            data = df[target_variable].dropna().values
            n_raw = len(data)
            mean_val = np.mean(data)
            var_val = np.var(data, ddof=1)
            std_val = np.sqrt(var_val)

            # 1. 朴素 SEM (未考虑自相关)
            naive_sem = std_val / np.sqrt(n_raw)

            # 2. tau_int 修正 SEM
            tau_int, _ = calc_autocorr_time(data)
            n_eff = n_raw / (2.0 * tau_int)
            tau_sem = std_val / np.sqrt(n_eff)

            # 3. 块平均法 (Block Averaging)
            b_sizes, b_se, plateau_se = compute_block_averaging(data)
            self.block_avg_results[name] = {
                "block_sizes": b_sizes,
                "block_se": b_se,
                "plateau_se": plateau_se
            }

            self.stats_summary[name] = {
                "mean": mean_val, "var": var_val, "std": std_val,
                "tau_int": tau_int, "n_raw": n_raw, "n_eff": n_eff,
                "naive_sem": naive_sem, "tau_sem": tau_sem, "plateau_sem": plateau_se
            }

            results.append({
                "Model": name,
                "Mean": mean_val,
                "Std Dev": std_val,
                "Tau_int": tau_int,
                "Effective N": n_eff,
                "Naive SEM": naive_sem,
                "Tau_int SEM": tau_sem,
                "Block Avg SEM": plateau_se,
                "Rel Diff (%)": abs(tau_sem - plateau_se) / plateau_se * 100
            })

        return pd.DataFrame(results).set_index("Model")

    def compare_two_models(self, model1_name: str, model2_name: str, target_variable: str = "pressure") -> pd.DataFrame:
        m1_stats = self.stats_summary[model1_name]
        m2_stats = self.stats_summary[model2_name]

        d1 = self.trimmed_data[model1_name][target_variable].values
        d2 = self.trimmed_data[model2_name][target_variable].values

        min_len = min(len(d1), len(d2))
        diff = d2[:min_len] - d1[:min_len]

        mbe = np.mean(diff)
        rmse = np.sqrt(np.mean(diff ** 2))
        mad = np.mean(np.abs(diff))

        ttest_res = welch_ttest_neff(
            mean1=m1_stats["mean"], var1=m1_stats["var"], neff1=m1_stats["n_eff"],
            mean2=m2_stats["mean"], var2=m2_stats["var"], neff2=m2_stats["n_eff"]
        )

        res = [{
            "Model (Target)": model2_name,
            "vs Reference": model1_name,
            "MBE (Offset)": mbe,
            "RMSE": rmse,
            "MAD": mad,
            "t-statistic": ttest_res["t_stat"],
            "p-value": ttest_res["p_value"],
            "Significant Offset?": "Yes" if ttest_res["is_significant"] else "No"
        }]
        return pd.DataFrame(res).set_index("Model (Target)")

    def plot_diagnostics(self, target_variable: str = "pressure", save_path: str = None, show: bool = True):
        """绘制 2x2 四合一诊断图，支持指定路径保存且防止图像空白"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. 完整时序图
        for name, df in self.raw_data.items():
            axes[0, 0].plot(df["time"], df[target_variable], alpha=0.35, label=f"{name} (raw)")
            trimmed = self.trimmed_data[name]
            axes[0, 0].plot(trimmed["time"], trimmed[target_variable], alpha=0.8, label=f"{name} (trimmed)")

        axes[0, 0].set_title(f"Time Series ({target_variable.capitalize()})")
        axes[0, 0].set_xlabel("Time")
        axes[0, 0].set_ylabel(target_variable.capitalize())
        axes[0, 0].legend(loc="upper right", fontsize=8)

        # 2. 自相关衰减 (ACF)
        for name, trimmed in self.trimmed_data.items():
            data = trimmed[target_variable].values
            _, acf = calc_autocorr_time(data, max_lag=min(500, len(data) // 4))
            axes[0, 1].plot(acf[:200], label=name)

        axes[0, 1].axhline(y=0, color='black', linestyle=':', alpha=0.5)
        axes[0, 1].set_title(f"ACF ({target_variable.capitalize()})")
        axes[0, 1].set_xlabel("Lag (frames)")
        axes[0, 1].set_ylabel("ACF")
        axes[0, 1].legend(loc="upper right", fontsize=8)

        # 3. 块平均标准误曲线 SE(M) 与 平台期 (Block Averaging Plateau)
        for name, blk_res in self.block_avg_results.items():
            b_sizes = blk_res["block_sizes"]
            b_se = blk_res["block_se"]
            plateau = blk_res["plateau_se"]
            tau_sem = self.stats_summary[name]["tau_sem"]

            p = axes[1, 0].plot(b_sizes, b_se, 'o-', label=f"{name} (Block SE)")
            color = p[0].get_color()
            axes[1, 0].axhline(y=plateau, color=color, linestyle='--', alpha=0.7,
                               label=f"{name} Plateau ({plateau:.3f})")
            axes[1, 0].axhline(y=tau_sem, color=color, linestyle=':', alpha=0.5,
                               label=f"{name} Tau_SEM ({tau_sem:.3f})")

        axes[1, 0].set_xscale("log")
        axes[1, 0].set_title("Block Averaging: SE(M) vs Block Size M")
        axes[1, 0].set_xlabel("Block Size M (log scale)")
        axes[1, 0].set_ylabel("Standard Error SE(M)")
        axes[1, 0].legend(loc="lower right", fontsize=7)

        # 4. 概率密度曲线 (KDE)
        for name, trimmed in self.trimmed_data.items():
            sns.kdeplot(trimmed[target_variable], ax=axes[1, 1], label=name, fill=True, alpha=0.2)

        axes[1, 1].set_title(f"Density Plot - Offset Visualizer")
        axes[1, 1].set_xlabel(target_variable.capitalize())
        axes[1, 1].set_ylabel("Density")
        axes[1, 1].legend(loc="upper right", fontsize=8)

        plt.tight_layout()

        # 关键修正：在 plt.show() 清空画布之前先保存
        if save_path:
            save_dir = os.path.dirname(os.path.abspath(save_path))
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"\n[图像已保存] 诊断图成功存至: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig


# ==========================================
# 3. 命令行参数解析入口
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对两个 MD CSV 文件进行 Offset 及统计学差异分析")

    parser.add_argument(
        "-f", "--files",
        nargs=2,
        required=True,
        metavar=("FILE1", "FILE2"),
        help="传入两个 CSV 文件的路径，如: --files path/to/model1.csv path/to/model2.csv"
    )

    parser.add_argument("--burn-in", type=float, default=0.20, help="剪裁 Burn-in 的比例 (默认: 0.20 代表前 20%)")
    parser.add_argument("--var", type=str, default="pressure", choices=["pressure", "temperature"],
                        help="分析的变量 (默认: pressure)")
    parser.add_argument("--save-dir", type=str, default="./plots", help="指定图表保存的文件夹路径 (默认: ./plots)")
    parser.add_argument("--no-show", action="store_true", help="加入此参数后只保存图片，不在屏幕上展示弹窗")

    args = parser.parse_args()

    COLUMN_MAPPING = {
        "time_fs": "time",
        "temperature_K": "temperature",
        "pressure_GPa": "pressure"
    }

    file1, file2 = args.files
    model1_name = os.path.basename(file1).replace(".csv", "")[3:-9]
    model2_name = os.path.basename(file2).replace(".csv", "")[3:-9]

    print(f"正在加载文件 1 [Ref]: {file1}")
    df1 = load_md_csv(file1, col_mapping=COLUMN_MAPPING)

    print(f"正在加载文件 2 [Comp]: {file2}")
    df2 = load_md_csv(file2, col_mapping=COLUMN_MAPPING)

    analyzer = MDAnalyzer()
    analyzer.add_model_data(model1_name, df1)
    analyzer.add_model_data(model2_name, df2)

    analyzer.remove_burn_in(burn_in_ratio=args.burn_in)

    # 计算统计特征并进行 Block Averaging 对比
    df_stats = analyzer.compute_statistics(target_variable=args.var)
    print(f"\n[表 1: 平稳段统计量与 SEM 交叉验证 (Naive vs Tau_int vs Block Averaging)]")
    print(df_stats.to_string())

    # 两模型对比
    df_compare = analyzer.compare_two_models(
        model1_name=model1_name,
        model2_name=model2_name,
        target_variable=args.var
    )
    print(f"\n[表 2: {model2_name} 相对 {model1_name} 的 Offset 检验结果]")
    print(df_compare.to_string())

    # 动态拼接包含模型名称与变量名的独立文件名
    filename = f"diagnostics_{model1_name}_vs_{model2_name}_{args.var}.png"
    save_path = os.path.join(args.save_dir, filename)

    # 绘图（包含保存逻辑）
    analyzer.plot_diagnostics(
        target_variable=args.var,
        save_path=save_path,
        show=not args.no_show
    )
