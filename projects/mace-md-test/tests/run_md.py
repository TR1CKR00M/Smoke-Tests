#!/usr/bin/env python3
"""
run_md.py

整合单次 MACE MD 模拟与基于 Run 目录的续跑机制。
续跑时自动继承并无缝合并父目录的轨迹文件 (.traj) 和热力学日志 (.csv)，
生成包含完整历史记录的输出文件，并保存包含完整元数据、依赖版本及 SHA256 校验的 manifest.yaml。

Usage:
python run_md.py --temp 600 --steps 20000 --model medium-mpa-0
python run_md.py --temp 600 --steps 20000 --model medium-mpa-0 --restart
python run_md.py --steps 50000 --parent-run-dir results/runs/run_20261002_195728_a1b2c3d4

"""

import argparse
import csv
import hashlib
import importlib.metadata
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import yaml
from ase import units
from ase.io import Trajectory, read
from ase.io.trajectory import Trajectory as TrajReader
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 默认基础输出路径
BASE_RESULTS_DIR = Path("results/runs")


def compute_sha256(filepath: str | Path) -> str:
    """计算文件的 SHA256 哈希值。"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def get_software_versions() -> Dict[str, str]:
    """获取关键依赖库及环境版本。"""
    versions = {
        "python": sys.version.split()[0],
    }
    packages = ["ase", "mace", "torch", "numpy", "scipy", "pyyaml"]
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not installed"

    try:
        import torch
        versions["cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            versions["cuda_version"] = torch.version.cuda or "unknown"
    except Exception:
        pass

    return versions


def attach_calculator(atoms, model: str, device: str = "", dtype: str = "float32"):
    """挂载 MACE 计算器。"""
    from mace.calculators import mace_mp

    calc = mace_mp(model=model, device=device, default_dtype=dtype)
    atoms.calc = calc
    return atoms


def create_run_directory(base_dir: Path) -> Tuple[Path, str, str]:
    """创建全局唯一的 Run 目录。"""
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_uuid = str(uuid.uuid4())
    run_id = f"run_{timestamp_str}_{run_uuid[:8]}"

    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, run_id, timestamp_str


def find_latest_run(base_dir: Path, model: str, temp: float) -> Optional[Path]:
    """在基础目录下查找符合特定模型和温度的最佳/最新 Run 目录。"""
    if not base_dir.exists():
        return None

    candidate_runs = []
    for manifest_path in base_dir.glob("*/manifest.yaml"):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                meta = yaml.safe_load(f)

            params = meta.get("parameters", {})
            if params.get("model") == model and abs(params.get("temperature", 0) - temp) < 1e-2:
                candidate_runs.append((meta.get("timestamp", ""), manifest_path.parent))
        except Exception:
            continue

    if not candidate_runs:
        return None

    candidate_runs.sort(key=lambda x: x[0], reverse=True)
    return candidate_runs[0][1]


def load_previous_csv_logs(csv_path: Path) -> List[Tuple[float, float, float, float]]:
    """读取已有的 CSV 日志，以便追加续跑数据。"""
    rows = []
    if not csv_path.exists():
        return rows
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # 跳过表头
        for line in reader:
            if line:
                rows.append((float(line[0]), float(line[1]), float(line[2]), float(line[3])))
    return rows


def run_nvt_md(args):
    """MD 运行主流程：包含初始化/读取断点、历史文件继承合并、MD 演化及元数据写出。"""
    base_dir = Path(args.output_dir)
    parent_run_dir = None
    parent_manifest = None

    # 1. 确定是新运行还是续跑
    if args.parent_run_dir:
        parent_run_dir = Path(args.parent_run_dir)
    elif args.restart:
        parent_run_dir = find_latest_run(base_dir, args.model, args.temp)

    # 创建本次运行的独立 Run 目录
    run_dir, run_id, timestamp_str = create_run_directory(base_dir)
    traj_path = run_dir / "trajectory.traj"
    log_path = run_dir / "thermo_log.csv"
    manifest_path = run_dir / "manifest.yaml"

    log_rows = []
    parent_traj_reader = None

    # 2. 读取结构与合并历史记录
    if parent_run_dir and parent_run_dir.exists():
        parent_traj_path = parent_run_dir / "trajectory.traj"
        parent_log_path = parent_run_dir / "thermo_log.csv"
        parent_manifest_path = parent_run_dir / "manifest.yaml"

        print(f"[*] 准备进行续跑，目标父运行目录: {parent_run_dir}")
        if parent_manifest_path.exists():
            with open(parent_manifest_path, "r", encoding="utf-8") as f:
                parent_manifest = yaml.safe_load(f)

        # 读取上一运行的轨迹
        parent_traj_reader = TrajReader(parent_traj_path)
        atoms = parent_traj_reader[-1]  # 提取最后一帧（保留当前真实的 positions 与 momenta/velocities）

        # 继承历史 CSV 日志
        log_rows = load_previous_csv_logs(parent_log_path)

        past_steps = parent_manifest.get("parameters", {}).get("total_accumulated_steps", (
                    len(parent_traj_reader) - 1) * args.log_every) if parent_manifest else (
                                                                                                       len(parent_traj_reader) - 1) * args.log_every
        input_file_sha256 = compute_sha256(parent_traj_path)
        input_source = str(parent_traj_path)
        print(f"[*] 成功加载最后一帧 (已有累计步数: {past_steps})")
    else:
        if args.restart or args.parent_run_dir:
            print("[!] 未找到可续跑的 Run 目录，将回退为启动全新的模拟...")
        else:
            print("[*] 开始全新的 MD 模拟...")

        atoms = read(args.xyz)
        MaxwellBoltzmannDistribution(
            atoms,
            temperature_K=args.temp,
            rng=np.random.default_rng(args.seed),
        )
        Stationary(atoms)
        past_steps = 0
        input_file_sha256 = compute_sha256(args.xyz)
        input_source = args.xyz

    print(f"[*] 创建 Run 目录: {run_dir}")

    # 3. 初始化新轨迹文件，若为续跑则先复制写出旧轨迹的全部历史帧
    if parent_traj_reader is not None:
        print("[*] 正在合并父运行的历史轨迹帧到新轨迹文件...")
        traj_out = Trajectory(traj_path, mode="w", atoms=parent_traj_reader[0])
        for frame in parent_traj_reader[1:]:
            traj_out.write(frame)
        # 挂载为追加模式
        traj = Trajectory(traj_path, mode="a", atoms=atoms)
    else:
        traj = Trajectory(traj_path, mode="w", atoms=atoms)

    # 4. 挂载 MACE 计算器与 Langevin 积分器
    atoms = attach_calculator(atoms, model=args.model, device=args.device, dtype=args.dtype)

    dyn = Langevin(
        atoms,
        timestep=args.timestep * units.fs,
        temperature_K=args.temp,
        friction=args.friction,
    )

    dyn.attach(traj.write, interval=args.log_every)

    # 5. 记录日志（包含时间轴偏移量计算）
    time_offset_fs = past_steps * args.timestep

    def log_step():
        epot = atoms.get_potential_energy()
        ekin = atoms.get_kinetic_energy()
        temp = ekin / (1.5 * units.kB * len(atoms))
        try:
            stress = atoms.get_stress(voigt=True)
            pressure_GPa = -np.mean(stress[:3]) / units.GPa
        except Exception:
            pressure_GPa = np.nan

        # 加上历史时间偏移，确保整段 csv 时间连续
        current_time_fs = time_offset_fs + (dyn.get_time() / units.fs)
        log_rows.append((current_time_fs, epot, temp, pressure_GPa))

    dyn.attach(log_step, interval=args.log_every)

    # 6. 执行 MD 步进
    total_accumulated_steps = past_steps + args.steps
    print(f"[*] 开始运行 MD ({args.steps} 步, 本次运行后总累计步数: {total_accumulated_steps})...")
    dyn.run(args.steps)

    # 7. 保存合并后的完整 CSV 日志
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("time_fs,potential_energy_eV,temperature_K,pressure_GPa\n")
        for row in log_rows:
            f.write(f"{row[0]:.3f},{row[1]:.6f},{row[2]:.3f},{row[3]:.4f}\n")

    # 8. 计算输出文件的 SHA256 并生成 manifest.yaml
    output_files_sha256 = {
        "trajectory.traj": compute_sha256(traj_path),
        "thermo_log.csv": compute_sha256(log_path),
    }

    manifest_data = {
        "run_id": run_id,
        "timestamp": timestamp_str,
        "parameters": {
            "model": args.model,
            "temperature": args.temp,
            "steps": args.steps,
            "total_accumulated_steps": total_accumulated_steps,
            "seed": args.seed,
            "timestep": args.timestep,
            "friction": args.friction,
            "device": args.device,
            "dtype": args.dtype,
            "log_every": args.log_every,
        },
        "lineage": {
            "is_restart": parent_run_dir is not None,
            "parent_run_dir": str(parent_run_dir) if parent_run_dir else None,
            "parent_run_id": parent_manifest.get("run_id") if parent_manifest else None,
        },
        "software_versions": get_software_versions(),
        "hashes": {
            "input_source": input_source,
            "input_sha256": input_file_sha256,
            "outputs": output_files_sha256,
        },
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.dump(manifest_data, f, default_flow_style=False, sort_keys=False)

    print(f"[*] 运行完成！合并后的完整文件已存入: {run_dir}")
    print(f"[*] 元数据已写入: {manifest_path}\n")

    return traj_path, run_dir


def analyze_msd(traj_path: Path, timestep_fs: float, log_every: int):
    """对生成的轨迹计算 MSD 和估算扩散系数。"""
    traj = TrajReader(traj_path)
    positions = np.array([f.get_positions() for f in traj])
    disp = positions - positions[0]
    msd = (disp ** 2).sum(axis=2).mean(axis=1)

    dt = timestep_fs * log_every
    times_fs = np.arange(len(msd)) * dt

    fit_start = len(times_fs) // 4
    slope, _ = np.polyfit(times_fs[fit_start:], msd[fit_start:], 1)
    D_cm2_s = slope * 1e-16 / 1e-15 / 6.0
    print(f"[*] 分析合并轨迹: {traj_path} (共 {len(traj)} 帧, 总时长 {times_fs[-1] / 1000:.2f} ps)")
    print(f"[*] 估算 3D 自扩散系数 D ≈ {D_cm2_s:.3e} cm^2/s\n")
    return times_fs, msd, D_cm2_s


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MACE MD 统一运行脚本 (支持独立目录管理、合并轨迹与 CSV 日志、断点续跑)")
    p.add_argument("--xyz", default="data/raw/sb2te3_cr_81atom.xyz", help="初始结构文件路径")
    p.add_argument("--model", default="medium-mpa-0", help="MACE 模型名称")
    p.add_argument("--device", default="", help="计算设备 (如 cuda, cpu)")
    p.add_argument("--dtype", default="float32", help="数据类型 (float32, float64)")
    p.add_argument("--temp", type=float, default=600.0, help="设定温度 (K)")
    p.add_argument("--timestep", type=float, default=1.0, help="时间步长 (fs)")
    p.add_argument("--friction", type=float, default=0.02, help="Langevin 摩擦系数")
    p.add_argument("--steps", type=int, default=20000, help="本次运行增量的 MD 步数")
    p.add_argument("--seed", type=int, default=42, help="随机数种子")
    p.add_argument("--log-every", type=int, default=20, help="采样/记录间隔")
    p.add_argument("--output-dir", default="results/runs", help="所有 Run 目录的根输出目录")
    p.add_argument("--restart", action="store_true", help="自动寻找符合条件的最新 Run 目录并续跑")
    p.add_argument("--parent-run-dir", type=str, default=None, help="显式指定用于续跑的父 Run 目录")

    args = p.parse_args()

    latest_traj, run_dir = run_nvt_md(args)
    analyze_msd(latest_traj, args.timestep, args.log_every)