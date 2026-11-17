"""Relax defect structures with MACE and compare them with DFT structures."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

# 1. 定义实验参数
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data/1_FM_benchmark/V_Sb1/DFT_PBE"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "model/mace-mh-1.model"
CHARGE_VALUES = {"mi3": -3, "mi2": -2, "mi1": -1, "neu": 0, "pl1": 1}
DEFAULT_DISTORTIONS = tuple(
    f"D_{d}" for d in [-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60]
)


def calculate_rmsd(atoms_pred, atoms_ref) -> float:
    """计算两个结构之间的原子坐标根均方偏差 (RMSD)。"""
    import numpy as np

    pos_pred = atoms_pred.get_positions()
    pos_ref = atoms_ref.get_positions()
    if pos_pred.shape != pos_ref.shape:
        raise ValueError(f"原子数或坐标维度不匹配: {pos_pred.shape} != {pos_ref.shape}")

    # 简单的 RMSD 计算（如需考虑晶胞平移/转动对齐，可引入 ase.geometry.neighborlist 或 signac 等）
    diff = pos_pred - pos_ref
    rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
    return rmsd


def parse_model_specs(values: Iterable[str]) -> list[tuple[str, str]]:
    """解析 NAME=MODEL 格式的模型参数。"""
    specs = []
    for value in values:
        name, separator, model = value.partition("=")
        if not separator or not name or not model:
            raise argparse.ArgumentTypeError(f"模型格式应为 NAME=MODEL，收到: {value!r}")
        specs.append((name, model))
    return specs


def find_structure(directory: Path, prefix: str) -> Path | None:
    matches = sorted(path for path in directory.iterdir() if path.name.startswith(prefix))
    return matches[0] if matches else None


def evaluate_model(
    model_name,
    model_spec,
    data_root,
    device,
    dtype,
    head,
    charge_states,
    distortions,
    max_structures,
    fmax,
    steps,
):
    try:
        import numpy  # noqa: F401 - validate the runtime dependency
        import torch
        from ase.io import read
        from ase.optimize import BFGS
        from mace.calculators import mace_mp
    except ImportError as exc:
        raise RuntimeError(
            "运行此脚本需要安装 torch、mace-torch、ase、numpy 和 matplotlib。"
        ) from exc

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 --device cuda，但当前 PyTorch 不支持 CUDA。")

    print(f"\n--- 正在评估模型: {model_name} ({model_spec}) [{device}] ---")
    calc = mace_mp(model=model_spec, device=device, default_dtype=dtype, head=head)
    rmsd_by_charge = {}

    processed = 0
    for charge in charge_states:
        charge_dir = data_root / charge
        rmsds = []
        if not charge_dir.is_dir():
            print(f"跳过 {charge}: 目录不存在 {charge_dir}")
            continue
        for distortion in distortions:
            if max_structures is not None and processed >= max_structures:
                break
            distortion_dir = charge_dir / distortion
            if not distortion_dir.is_dir():
                continue
            pos_path = find_structure(distortion_dir, "POSCAR")
            ref_path = find_structure(distortion_dir, "CONTCAR")
            if pos_path is None or ref_path is None:
                print(f"跳过 {charge}/{distortion}: 缺少 POSCAR 或 CONTCAR")
                continue
            try:
                atoms = read(pos_path)
                dft_ref = read(ref_path)
                atoms.calc = calc
                BFGS(atoms, trajectory=None, logfile=None).run(fmax=fmax, steps=steps)
                rmsds.append(calculate_rmsd(atoms, dft_ref))
                processed += 1
            except Exception as exc:
                print(f"优化失败 {charge}/{distortion}: {exc}")
        if rmsds:
            rmsd_by_charge[charge] = min(rmsds)
    return rmsd_by_charge


def plot_results(results, output: Path, show: bool):
    import matplotlib.pyplot as plt
    import numpy as np

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.axhspan(0, 0.1, color="green", alpha=0.15, label="Successful identification (RMSD ≤ 0.1 Å)")
    plt.axhline(0.1, color="green", linestyle="--", alpha=0.7)
    markers = {"MACE-MH-1": "D", "MACE-MPA-0": "o"}
    colors = {"MACE-MH-1": "#d4ac0d", "MACE-MPA-0": "#b7950b"}
    for model_name, charge_data in results.items():
        x_values = list(CHARGE_VALUES.values())
        y_values = [charge_data.get(charge, np.nan) for charge in CHARGE_VALUES]
        plt.scatter(x_values, y_values, marker=markers.get(model_name, "o"),
                    color=colors.get(model_name, "blue"), label=model_name, s=50, zorder=3)
    plt.xlabel("Defect charge state ($q$)", fontsize=12)
    plt.ylabel("RMSD relative to DFT ($\\mathrm{\\AA}$)", fontsize=12)
    plt.xticks(list(CHARGE_VALUES.values()))
    plt.ylim(0, 0.6)
    plt.legend(frameon=True, loc="upper right", fontsize=10)
    plt.title("Replication of Fig. 1(a): $V_{\\text{Sb}(1)}$ in $\\mathrm{Sb_2Se_3}$", fontsize=11)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()
    print(f"结果图已保存到: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--model",
        action="append",
        metavar="NAME=MODEL_OR_PATH",
        help="可选的 MACE 模型；默认只使用 model/mace-mh-1.model。",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--head",
        default="omat_pbe",
        help="MH-1 多头模型使用的 head；标准无机 PBE 数据默认使用 omat_pbe。",
    )
    parser.add_argument(
        "--charges",
        nargs="+",
        choices=tuple(CHARGE_VALUES),
        default=list(CHARGE_VALUES),
        help="只运行指定的电荷态。",
    )
    parser.add_argument(
        "--distortions",
        nargs="+",
        default=list(DEFAULT_DISTORTIONS),
        help="只运行指定的畸变目录，例如 D_0 D_10。",
    )
    parser.add_argument(
        "--max-structures",
        type=int,
        default=None,
        help="限制本次运行的结构总数，适合 smoke test。",
    )
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "fig1a_replicated.png")
    parser.add_argument("--show", action="store_true", help="完成后显示图形窗口")
    args = parser.parse_args()
    if not args.data_root.is_dir():
        parser.error(f"数据目录不存在: {args.data_root}")
    if args.model is None and not DEFAULT_MODEL_PATH.is_file():
        parser.error(f"默认 MH-1 模型不存在: {DEFAULT_MODEL_PATH}")
    model_specs = parse_model_specs(
        args.model or [f"MACE-MH-1={DEFAULT_MODEL_PATH}"]
    )
    try:
        results = {
            name: evaluate_model(
                name,
                spec,
                args.data_root,
                args.device,
                args.dtype,
                args.head,
                args.charges,
                args.distortions,
                args.max_structures,
                args.fmax,
                args.steps,
            )
            for name, spec in model_specs
        }
    except RuntimeError as exc:
        parser.error(str(exc))
    plot_results(results, args.output, args.show)


if __name__ == "__main__":
    main()
