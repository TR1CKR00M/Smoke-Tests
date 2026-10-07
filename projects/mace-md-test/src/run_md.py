#!/usr/bin/env python3
"""
run_md.py

整合单次 MACE MD 模拟与基于 Run 目录的续跑机制。
续跑时自动继承并无缝合并父目录的轨迹文件 (.traj) 和热力学日志 (.csv)，
生成包含完整历史记录的输出文件，并保存包含完整元数据、依赖版本及 SHA256 校验的 manifest.yaml。

Usage:
python src/run_md.py --temp 600 --steps 20000 --model medium-mpa-0
python src/run_md.py --temp 600 --steps 20000 --model medium-mpa-0 --restart
python src/run_md.py --steps 50000 --parent-run-dir results/runs/<parent-run-directory>

"""

import argparse
import csv
import hashlib
import importlib.metadata
import os
import platform
import sys
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

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


def get_software_versions() -> dict[str, str]:
    """获取关键依赖库及环境版本。"""
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    packages = ["ase", "mace-torch", "torch", "numpy", "scipy", "pyyaml"]
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
            versions["cuda_device"] = torch.cuda.get_device_name(0)
        versions["rocm_version"] = torch.version.hip or "unknown"
        versions["mps_available"] = str(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
    except Exception:
        pass

    return versions


def resolve_model_provenance(model: str) -> dict[str, str | None]:
    """Resolve the requested MACE model to a concrete file and fingerprint it."""
    from mace.calculators.foundations_models import (
        download_mace_mp_checkpoint,
        mace_mp_urls,
    )

    requested = str(model)
    requested_path = Path(requested).expanduser()
    if requested_path.is_file():
        model_path = requested_path.resolve()
        source_url = None
        source_kind = "local_file"
    else:
        # Resolve aliases and URLs through the same MACE helper used by
        # mace_mp, then pass the concrete path to prevent a second resolution.
        model_path = Path(download_mace_mp_checkpoint(requested)).resolve()
        source_url = mace_mp_urls.get(requested)
        if requested.startswith(("https://", "http://")):
            source_url = requested
        source_kind = "mace_mp_alias" if requested in mace_mp_urls else (
            "url" if source_url else "mace_mp_resolver"
        )

    model_provenance = {
        "requested": requested,
        "source_kind": source_kind,
        "source_url": source_url,
        "resolved_path": str(model_path),
        "sha256": compute_sha256(model_path),
    }
    return model_provenance


def attach_calculator(
    atoms, model_provenance: dict[str, str | None], device: str = "", dtype: str = "float32"
):
    """挂载已经解析并记录来源的 MACE 计算器。"""
    from mace.calculators import mace_mp

    calc = mace_mp(
        model=model_provenance["resolved_path"], device=device, default_dtype=dtype
    )
    atoms.calc = calc
    return atoms


def create_run_directory(base_dir: Path) -> tuple[Path, str, str]:
    """创建全局唯一的 Run 目录。"""
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_uuid = str(uuid.uuid4())
    run_id = f"run_{timestamp_str}_{run_uuid[:8]}"

    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, run_id, timestamp_str


def find_latest_run(base_dir: Path, model: str, temp: float) -> Path | None:
    """在基础目录下查找符合特定模型和温度的最佳/最新 Run 目录。"""
    if not base_dir.exists():
        return None

    candidate_runs = []
    for manifest_path in base_dir.glob("*/manifest.yaml"):
        try:
            with open(manifest_path, encoding="utf-8") as f:
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


def load_previous_csv_logs(csv_path: Path) -> list[tuple[float, float, float, float]]:
    """读取已有的 CSV 日志，以便追加续跑数据。"""
    rows = []
    if not csv_path.exists():
        return rows
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        for line in reader:
            if not line or line[0].strip() == "time_fs":
                continue
            if len(line) != 4:
                raise ValueError(f"expected four columns in {csv_path}, got {len(line)}")
            rows.append((float(line[0]), float(line[1]), float(line[2]), float(line[3])))
    return rows


def run_nvt_md(args):
    """MD 运行主流程：包含初始化/读取断点、历史文件继承合并、MD 演化及元数据写出。"""
    # Keep ASE/MACE optional for repository-level provenance and plotting tests.
    # They are required only when an actual MD run is requested.
    from ase import units
    from ase.io import Trajectory, read
    from ase.io.trajectory import Trajectory as TrajReader
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

    base_dir = Path(args.output_dir)
    parent_run_dir = None
    parent_manifest = None

    # 1. 确定是新运行还是续跑
    if args.parent_run_dir:
        parent_run_dir = Path(args.parent_run_dir)
        if not parent_run_dir.is_dir():
            raise FileNotFoundError(f"parent run directory does not exist: {parent_run_dir}")
    elif args.restart:
        parent_run_dir = find_latest_run(base_dir, args.model, args.temp)
        if parent_run_dir is None:
            raise FileNotFoundError(
                f"no parent run found for model={args.model!r}, temperature={args.temp} K"
            )

    model_provenance = resolve_model_provenance(args.model)
    if parent_run_dir:
        parent_manifest_path = parent_run_dir / "manifest.yaml"
        if not parent_manifest_path.is_file():
            raise FileNotFoundError(f"parent run has no manifest: {parent_manifest_path}")
        with open(parent_manifest_path, encoding="utf-8") as f:
            parent_manifest = yaml.safe_load(f) or {}
        for name in ("trajectory.traj", "thermo_log.csv"):
            artifact = parent_run_dir / name
            if not artifact.is_file():
                raise FileNotFoundError(f"parent run is missing {name}: {artifact}")
            expected_hash = parent_manifest.get("hashes", {}).get("outputs", {}).get(name)
            if expected_hash and compute_sha256(artifact) != expected_hash:
                raise ValueError(f"parent run artifact hash mismatch: {artifact}")
        parent_sha256 = (parent_manifest.get("model_provenance") or {}).get("sha256")
        if parent_sha256 and parent_sha256 != model_provenance["sha256"]:
            raise ValueError(
                "Restart model differs from the parent run: "
                f"parent SHA256={parent_sha256}, "
                f"current SHA256={model_provenance['sha256']}"
            )
        parent_params = parent_manifest.get("parameters", {})
        for key, current in (
            ("temperature", args.temp),
            ("timestep", args.timestep),
            ("friction", args.friction),
            ("dtype", args.dtype),
            ("log_every", args.log_every),
        ):
            previous = parent_params.get(key)
            if previous is not None and previous != current:
                raise ValueError(
                    f"Restart parameter {key}={current!r} differs from parent value {previous!r}"
                )

    rng = np.random.default_rng(args.seed)
    rng_state_restored = False
    if parent_manifest and parent_manifest.get("rng_state_after_run"):
        rng.bit_generator.state = parent_manifest["rng_state_after_run"]
        rng_state_restored = True

    log_rows = []
    parent_traj_reader = None

    # 2. 读取结构与合并历史记录
    if parent_run_dir is not None:
        parent_traj_path = parent_run_dir / "trajectory.traj"
        parent_log_path = parent_run_dir / "thermo_log.csv"
        parent_manifest_path = parent_run_dir / "manifest.yaml"

        print(f"[*] 准备进行续跑，目标父运行目录: {parent_run_dir}")
        # 读取上一运行的轨迹
        parent_traj_reader = TrajReader(parent_traj_path)
        atoms = parent_traj_reader[-1]  # 提取最后一帧

        # 继承历史 CSV 日志
        log_rows = load_previous_csv_logs(parent_log_path)

        past_steps = parent_manifest.get("parameters", {}).get("total_accumulated_steps", (
                    len(parent_traj_reader) - 1) * args.log_every) if parent_manifest else (
                    len(parent_traj_reader) - 1) * args.log_every
        input_file_sha256 = compute_sha256(parent_traj_path)
        input_source = str(parent_traj_path)
        print(f"[*] 成功加载最后一帧 (已有累计步数: {past_steps})")
    else:
        print("[*] 开始全新的 MD 模拟...")

        atoms = read(args.xyz)
        MaxwellBoltzmannDistribution(
            atoms,
            temperature_K=args.temp,
            rng=rng,
        )
        Stationary(atoms)
        past_steps = 0
        input_file_sha256 = compute_sha256(args.xyz)
        input_source = args.xyz

    if (past_steps + args.steps) % args.log_every:
        raise ValueError(
            "total accumulated steps must be divisible by --log-every "
            "so the final state is included in the trajectory and log"
        )

    # Create the run only after parent data and continuation settings validate.
    run_dir, run_id, timestamp_str = create_run_directory(base_dir)
    traj_path = run_dir / "trajectory.traj"
    log_path = run_dir / "thermo_log.csv"
    manifest_path = run_dir / "manifest.yaml"

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
    atoms = attach_calculator(
        atoms, model_provenance=model_provenance, device=args.device, dtype=args.dtype
    )

    dyn = Langevin(
        atoms,
        timestep=args.timestep * units.fs,
        temperature_K=args.temp,
        friction=args.friction,
        rng=rng,
    )
    # Continue ASE's global step count so restart observers do not emit a
    # duplicate frame/log row at the parent trajectory's final time.
    dyn.nsteps = past_steps

    dyn.attach(traj.write, interval=args.log_every)

    # 5. 记录日志（包含时间轴偏移量计算）
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
        current_time_fs = dyn.get_time() / units.fs
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

    software_versions = get_software_versions()
    selected_device = args.device or (
        "cuda" if software_versions.get("cuda_available") == "True" else "cpu"
    )
    manifest_data = {
        "run_id": run_id,
        "timestamp": timestamp_str,
        "parameters": {
            "model": args.model,
            "temperature": args.temp,
            "steps": args.steps,
            "total_accumulated_steps": total_accumulated_steps,
            "seed": args.seed,
            "rng_state_restored": rng_state_restored,
            "timestep": args.timestep,
            "friction": args.friction,
            "device": args.device,
            "device_selected": selected_device,
            "dtype": args.dtype,
            "log_every": args.log_every,
        },
        "integrator": {
            "name": "ASE Langevin",
            "fix_center_of_mass": True,
            "temperature_definition": "2 * kinetic_energy / (3 * atom_count * kB)",
            "pressure_definition": (
                "negative mean of three diagonal virial stresses, converted to GPa"
            ),
        },
        "lineage": {
            "is_restart": parent_run_dir is not None,
            "parent_run_dir": str(parent_run_dir) if parent_run_dir else None,
            "parent_run_id": parent_manifest.get("run_id") if parent_manifest else None,
        },
        "software_versions": software_versions,
        "model_provenance": model_provenance,
        "rng_state_after_run": rng.bit_generator.state,
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
    from ase.io.trajectory import Trajectory as TrajReader

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
    p = argparse.ArgumentParser(description="MACE_MD 统一运行脚本")
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

    if args.temp <= 0 or args.timestep <= 0 or args.friction <= 0:
        p.error("--temp, --timestep, and --friction must be positive")
    if args.steps <= 0 or args.log_every <= 0:
        p.error("--steps and --log-every must be positive")
    if args.steps % args.log_every:
        p.error("--steps must be divisible by --log-every so the final state is recorded")

    latest_traj, run_dir = run_nvt_md(args)
    analyze_msd(latest_traj, args.timestep, args.log_every)
