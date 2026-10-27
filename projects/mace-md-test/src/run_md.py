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
import signal
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


def atomic_write_yaml(path: Path, data: dict) -> None:
    """Write checkpoint metadata atomically so a killed job leaves the prior checkpoint intact."""
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary_path, path)


def write_csv_rows(csv_path: Path, rows: list[tuple[float, float, float, float]]) -> None:
    """Atomically persist all committed thermodynamic rows."""
    temporary_path = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
    with open(temporary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("time_fs", "potential_energy_eV", "temperature_K", "pressure_GPa"))
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary_path, csv_path)


def find_latest_run(base_dir: Path, model: str, temp: float) -> Path | None:
    """Prefer the newest incomplete matching run, then completed runs.

    A pre-checkpoint legacy run has no manifest, so it can only be selected
    automatically when it is the sole manifest-less trajectory directory.
    """
    if not base_dir.exists():
        return None

    incomplete_runs = []
    completed_runs = []
    legacy_runs = []
    for run_dir in base_dir.iterdir():
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "manifest.yaml"
        trajectory_path = run_dir / "trajectory.traj"
        if not manifest_path.is_file():
            if trajectory_path.is_file():
                legacy_runs.append((run_dir.stat().st_mtime, run_dir))
            continue
        try:
            with open(manifest_path, encoding="utf-8") as f:
                meta = yaml.safe_load(f)
            if not isinstance(meta, dict):
                continue
            params = meta.get("parameters", {})
            if params.get("model") == model and abs(params.get("temperature", 0) - temp) < 1e-2:
                status = meta.get("status", "completed")
                modified = manifest_path.stat().st_mtime
                if status in {"running", "interrupted", "failed"}:
                    incomplete_runs.append((modified, run_dir))
                elif status != "resumed":
                    completed_runs.append((modified, run_dir))
        except Exception:
            continue

    if incomplete_runs:
        return max(incomplete_runs, key=lambda item: item[0])[1]

    newest_completed = max(completed_runs, key=lambda item: item[0]) if completed_runs else None
    if len(legacy_runs) == 1 and (
        newest_completed is None or legacy_runs[0][0] > newest_completed[0]
    ):
        return legacy_runs[0][1]
    if len(legacy_runs) > 1 and (
        newest_completed is None or max(legacy_runs, key=lambda item: item[0])[0] > newest_completed[0]
    ):
        raise ValueError(
            "multiple legacy runs have a trajectory but no manifest, so their model and "
            "temperature cannot be identified safely; pass --parent-run-dir explicitly"
        )

    return newest_completed[1] if newest_completed else None


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


def reconstruct_csv_logs(
    trajectory_reader,
    frame_count: int,
    model_provenance: dict[str, str | None],
    timestep: float,
    log_every: int,
    units,
    device: str = "",
    dtype: str = "float32",
) -> list[tuple[float, float, float, float]]:
    """Recalculate logs for older interrupted runs that only saved a trajectory."""
    rows = []
    for frame_index in range(frame_count):
        atoms = trajectory_reader[frame_index]
        attach_calculator(atoms, model_provenance, device=device, dtype=dtype)
        energy = atoms.get_potential_energy()
        kinetic_energy = atoms.get_kinetic_energy()
        temperature = kinetic_energy / (1.5 * units.kB * len(atoms))
        try:
            stress = atoms.get_stress(voigt=True)
            pressure = -np.mean(stress[:3]) / units.GPa
        except Exception:
            pressure = np.nan
        time_fs = frame_index * log_every * timestep
        rows.append((time_fs, energy, temperature, pressure))
    return rows


def run_nvt_md(args):
    """Run an MD segment, checkpoint progress, and resume interrupted runs safely."""
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
    auto_resume_incomplete = False

    # 1. Pick a parent. For --restart, an incomplete matching run takes priority.
    if args.parent_run_dir:
        parent_run_dir = Path(args.parent_run_dir)
        if not parent_run_dir.is_dir():
            raise FileNotFoundError(f"parent run directory does not exist: {parent_run_dir}")
    elif args.restart:
        parent_run_dir = find_latest_run(base_dir, args.model, args.temp)
        if parent_run_dir is None:
            raise FileNotFoundError(
                f"no run found for model={args.model!r}, temperature={args.temp} K"
            )

    if parent_run_dir is not None:
        parent_manifest_path = parent_run_dir / "manifest.yaml"
        if parent_manifest_path.is_file():
            with open(parent_manifest_path, encoding="utf-8") as f:
                parent_manifest = yaml.safe_load(f) or {}
        elif args.restart:
            print(
                "[!] 发现旧格式的轨迹目录，没有 manifest；无法核验原模型和参数，"
                "将按本次命令提供的设置续跑。"
            )
        auto_resume_incomplete = bool(
            args.restart
            and (
                parent_manifest is None
                or parent_manifest.get("status", "completed") != "completed"
            )
        )

    model_provenance = resolve_model_provenance(args.model)
    parent_params = (parent_manifest or {}).get("parameters", {})
    parent_traj_reader = None
    parent_frame_count = 0
    log_rows: list[tuple[float, float, float, float]] = []

    if parent_run_dir is not None:
        parent_traj_path = parent_run_dir / "trajectory.traj"
        parent_log_path = parent_run_dir / "thermo_log.csv"
        if not parent_traj_path.is_file():
            raise FileNotFoundError(f"parent run is missing trajectory.traj: {parent_traj_path}")

        expected_hashes = (parent_manifest or {}).get("hashes", {}).get("outputs", {})
        expected_hash = expected_hashes.get("trajectory.traj")
        if expected_hash and compute_sha256(parent_traj_path) != expected_hash:
            raise ValueError(f"Restart parent trajectory hash mismatch: {parent_traj_path}")
        expected_log_hash = expected_hashes.get("thermo_log.csv")
        if expected_log_hash and parent_log_path.is_file():
            if compute_sha256(parent_log_path) != expected_log_hash:
                raise ValueError(f"Restart parent log hash mismatch: {parent_log_path}")

        parent_sha256 = (parent_manifest or {}).get("model_provenance", {}).get("sha256")
        if parent_sha256 and parent_sha256 != model_provenance["sha256"]:
            raise ValueError(
                "Restart model differs from the parent run: "
                f"parent SHA256={parent_sha256}, current SHA256={model_provenance['sha256']}"
            )
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

        parent_traj_reader = TrajReader(parent_traj_path)
        available_frames = len(parent_traj_reader)
        if available_frames == 0:
            raise ValueError(f"parent trajectory contains no frames: {parent_traj_path}")

        checkpoint = (parent_manifest or {}).get("checkpoint", {})
        if auto_resume_incomplete and checkpoint.get("frames") is not None:
            parent_frame_count = min(available_frames, int(checkpoint["frames"]))
        else:
            parent_frame_count = available_frames
        if parent_frame_count < 1:
            raise ValueError(
                "the interrupted run has no committed trajectory checkpoint; "
                "resume from an earlier run instead"
            )

        past_steps = int(
            checkpoint.get(
                "completed_steps",
                parent_params.get(
                    "total_accumulated_steps", (parent_frame_count - 1) * args.log_every
                ),
            )
        )
        atoms = parent_traj_reader[parent_frame_count - 1]
        input_file_sha256 = compute_sha256(parent_traj_path)
        input_source = str(parent_traj_path)

        if parent_log_path.is_file():
            try:
                log_rows = load_previous_csv_logs(parent_log_path)
            except (ValueError, csv.Error):
                log_rows = []
        if len(log_rows) < parent_frame_count:
            print("[*] 旧运行没有完整热力学日志，正在从已保存轨迹重建 CSV...")
            log_rows = reconstruct_csv_logs(
                parent_traj_reader,
                parent_frame_count,
                model_provenance,
                args.timestep,
                args.log_every,
                units,
                args.device,
                args.dtype,
            )
        else:
            # A crash can occur after writing the next row/frame but before
            # advancing the atomic progress checkpoint. Ignore that tail.
            log_rows = log_rows[:parent_frame_count]

        if auto_resume_incomplete:
            requested_total_steps = int(
                parent_params.get("requested_total_steps", args.steps)
            )
            run_steps = max(0, requested_total_steps - past_steps)
            print(
                f"[*] 检测到未完成运行：已保存 {past_steps} 步，"
                f"目标 {requested_total_steps} 步，本次补跑 {run_steps} 步。"
            )
        else:
            run_steps = args.steps
            requested_total_steps = past_steps + run_steps
            print(f"[*] 准备续跑: {parent_run_dir} (已有累计步数: {past_steps})")

        if run_steps == 0:
            print("[*] 已达到目标步数，无需续跑。")
            if parent_manifest is not None and parent_manifest.get("status") != "completed":
                parent_manifest["status"] = "completed"
                parent_manifest.setdefault("parameters", {})[
                    "total_accumulated_steps"
                ] = past_steps
                parent_manifest["checkpoint"] = {
                    "completed_steps": past_steps,
                    "frames": parent_frame_count,
                }
                if len(log_rows) >= parent_frame_count:
                    write_csv_rows(parent_log_path, log_rows[:parent_frame_count])
                parent_manifest.setdefault("hashes", {})["outputs"] = {
                    name: compute_sha256(parent_run_dir / name)
                    for name in ("trajectory.traj", "thermo_log.csv")
                    if (parent_run_dir / name).is_file()
                }
                parent_manifest.pop("last_error", None)
                atomic_write_yaml(parent_run_dir / "manifest.yaml", parent_manifest)
            parent_traj_reader.close()
            return parent_traj_path, parent_run_dir
    else:
        print("[*] 开始全新的 MD 模拟...")
        atoms = read(args.xyz)
        rng = np.random.default_rng(args.seed)
        MaxwellBoltzmannDistribution(atoms, temperature_K=args.temp, rng=rng)
        Stationary(atoms)
        past_steps = 0
        run_steps = args.steps
        requested_total_steps = run_steps
        input_file_sha256 = compute_sha256(args.xyz)
        input_source = args.xyz

    if (past_steps + run_steps) % args.log_every:
        raise ValueError(
            "total accumulated steps must be divisible by --log-every "
            "so the final state is included in the trajectory and log"
        )

    # Legacy incomplete trajectories have no saved RNG state. New checkpoints
    # persist it after every logged frame so the stochastic integrator resumes.
    if parent_run_dir is not None:
        rng = np.random.default_rng(args.seed)
        saved_rng_state = (parent_manifest or {}).get("rng_state_after_run")
        rng_state_restored = bool(saved_rng_state)
        if saved_rng_state:
            rng.bit_generator.state = saved_rng_state
        if not rng_state_restored:
            print("[!] 父运行没有 RNG 检查点；续跑可继续，但随机序列不能精确复原。")

    run_dir, run_id, timestamp_str = create_run_directory(base_dir)
    traj_path = run_dir / "trajectory.traj"
    log_path = run_dir / "thermo_log.csv"
    manifest_path = run_dir / "manifest.yaml"
    print(f"[*] 创建 Run 目录: {run_dir}")

    if parent_traj_reader is not None:
        print("[*] 正在复制已提交的轨迹帧...")
        traj_out = Trajectory(traj_path, mode="w", atoms=parent_traj_reader[0])
        for frame_index in range(1, parent_frame_count):
            traj_out.write(parent_traj_reader[frame_index])
        traj_out.close()
        parent_traj_reader.close()
        traj = Trajectory(traj_path, mode="a", atoms=atoms)
        checkpoint_frames = parent_frame_count
    else:
        traj = Trajectory(traj_path, mode="w", atoms=atoms)
        checkpoint_frames = 1

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
    dyn.nsteps = past_steps

    software_versions = get_software_versions()
    selected_device = args.device or (
        "cuda" if software_versions.get("cuda_available") == "True" else "cpu"
    )
    manifest_data = {
        "run_id": run_id,
        "timestamp": timestamp_str,
        "status": "running",
        "parameters": {
            "model": args.model,
            "temperature": args.temp,
            "steps": run_steps,
            "requested_total_steps": requested_total_steps,
            "total_accumulated_steps": past_steps,
            "seed": args.seed,
            "rng_state_restored": bool(
                parent_run_dir is not None and rng_state_restored
            ),
            "timestep": args.timestep,
            "friction": args.friction,
            "device": args.device,
            "device_selected": selected_device,
            "dtype": args.dtype,
            "log_every": args.log_every,
        },
        "checkpoint": {"completed_steps": past_steps, "frames": checkpoint_frames},
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
            "parent_run_id": (parent_manifest or {}).get("run_id"),
            "legacy_parent_unverified": parent_run_dir is not None and parent_manifest is None,
        },
        "software_versions": software_versions,
        "model_provenance": model_provenance,
        "rng_state_after_run": rng.bit_generator.state,
        "hashes": {"input_source": input_source, "input_sha256": input_file_sha256, "outputs": {}},
    }

    write_csv_rows(log_path, log_rows)
    if parent_traj_reader is None:
        # Persist a frame-zero checkpoint before starting the integrator.
        traj.write(atoms)
        epot = atoms.get_potential_energy()
        ekin = atoms.get_kinetic_energy()
        temperature = ekin / (1.5 * units.kB * len(atoms))
        try:
            pressure = -np.mean(atoms.get_stress(voigt=True)[:3]) / units.GPa
        except Exception:
            pressure = np.nan
        log_rows.append((0.0, epot, temperature, pressure))
        checkpoint_frames = 1
        write_csv_rows(log_path, log_rows)
        manifest_data["checkpoint"] = {"completed_steps": 0, "frames": checkpoint_frames}
        manifest_data["rng_state_after_run"] = rng.bit_generator.state
    atomic_write_yaml(manifest_path, manifest_data)

    # Retire the source checkpoint only after the child has its own initial
    # committed state, so future --restart selects the child if it is interrupted.
    if auto_resume_incomplete and parent_manifest is not None:
        parent_manifest["status"] = "resumed"
        parent_manifest["resumed_by"] = run_id
        atomic_write_yaml(parent_run_dir / "manifest.yaml", parent_manifest)

    last_saved_step = past_steps

    def save_checkpoint():
        nonlocal checkpoint_frames, last_saved_step
        traj.write(atoms)
        epot = atoms.get_potential_energy()
        ekin = atoms.get_kinetic_energy()
        temperature = ekin / (1.5 * units.kB * len(atoms))
        try:
            pressure = -np.mean(atoms.get_stress(voigt=True)[:3]) / units.GPa
        except Exception:
            pressure = np.nan
        time_fs = dyn.get_time() / units.fs
        log_rows.append((time_fs, epot, temperature, pressure))
        checkpoint_frames += 1
        write_csv_rows(log_path, log_rows)
        last_saved_step = int(dyn.nsteps)
        manifest_data["parameters"]["total_accumulated_steps"] = last_saved_step
        manifest_data["checkpoint"] = {
            "completed_steps": last_saved_step,
            "frames": checkpoint_frames,
        }
        manifest_data["rng_state_after_run"] = rng.bit_generator.state
        atomic_write_yaml(manifest_path, manifest_data)

    def checkpoint_observer():
        if dyn.nsteps > last_saved_step:
            save_checkpoint()

    dyn.attach(checkpoint_observer, interval=args.log_every)
    print(
        f"[*] 开始运行 MD ({run_steps} 步，本次运行后总累计步数: "
        f"{past_steps + run_steps})..."
    )

    old_sigterm_handler = signal.getsignal(signal.SIGTERM)

    def handle_sigterm(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handle_sigterm)
    try:
        dyn.run(run_steps)
    except BaseException as exc:
        manifest_data["status"] = (
            "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        )
        manifest_data["last_error"] = repr(exc)
        atomic_write_yaml(manifest_path, manifest_data)
        raise
    finally:
        signal.signal(signal.SIGTERM, old_sigterm_handler)

    traj.close()
    total_accumulated_steps = past_steps + run_steps
    manifest_data["status"] = "completed"
    manifest_data["parameters"]["total_accumulated_steps"] = total_accumulated_steps
    manifest_data["checkpoint"] = {
        "completed_steps": total_accumulated_steps,
        "frames": checkpoint_frames,
    }
    manifest_data.pop("last_error", None)
    manifest_data["hashes"]["outputs"] = {
        "trajectory.traj": compute_sha256(traj_path),
        "thermo_log.csv": compute_sha256(log_path),
    }
    manifest_data["rng_state_after_run"] = rng.bit_generator.state
    atomic_write_yaml(manifest_path, manifest_data)

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
