#!/usr/bin/env python3
"""
run_foundation_md.py

带有自动步数识别与文件名更新功能的 MACE 分子动力学模拟脚本。
文件名会自动带上累计总步数（例如：md_medium-mpa-0_600K_400000steps.traj）。


Usage:
python run_md_auto.py --temp 600 --steps 200000
python run_md_auto.py --temp 600 --steps 200000 --restart
"""

import argparse
import glob
from pathlib import Path
import os
import re
import numpy as np
from ase import units
from ase.io import read, Trajectory
from ase.io.trajectory import Trajectory as TrajReader
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

output_dir = Path("results/raw")
output_dir.mkdir(parents=True, exist_ok=True)

def attach_calculator(
    atoms, model: str, device: str = "", dtype: str = "float32"
):
  from mace.calculators import mace_mp

  calc = mace_mp(model=model, device=device, default_dtype=dtype)
  atoms.calc = calc
  return atoms


def find_latest_traj(
    model: str, temp: float, data_dir: str = os.path.join("results", "raw")
):
  """寻找指定目录下匹配模型和温度的最新的轨迹文件，并提取其已跑的总步数。"""
  pattern = os.path.join(data_dir, f"md_{model}_{int(temp)}K_*steps.traj")
  files = glob.glob(pattern)
  if not files:
    return None, 0

  max_steps = -1
  latest_file = None
  for f in files:
    match = re.search(r"(\d+)steps\.traj$", f)
    if match:
      steps = int(match.group(1))
      if steps > max_steps:
        max_steps = steps
        latest_file = f

  return latest_file, max_steps

  max_steps = -1
  latest_file = None
  for f in files:
    match = re.search(r"(\d+)steps\.traj$", f)
    if match:
      steps = int(match.group(1))
      if steps > max_steps:
        max_steps = steps
        latest_file = f

  return latest_file, max_steps


def run_nvt(
    xyz_path: str,
    temperature_K: float = 600.0,
    timestep_fs: float = 1.0,
    friction: float = 0.02,
    n_steps: int = 200000,  # 本次要运行的步数
    log_every: int = 20,
    restart: bool = False,
    seed: int = 42,
    model: str = "medium-mpa-0",
    device: str = "",
    dtype: str = "float32",
):
  latest_traj, past_steps = find_latest_traj(model, temperature_K)

  # -------------------------------------------------------------
  # 1. 判断是全新的模拟还是续跑
  # -------------------------------------------------------------
  if restart and latest_traj is not None:
    print(
      f"检测到 --restart 标记，找到上一次轨迹文件: {latest_traj} (已跑"
      f" {past_steps} 步)"
    )
    print("正在从该文件的最后一帧读取结构与速度...")

    # 从旧轨迹提取最后一帧（自动包含位置和速度）
    reader = TrajReader(latest_traj)
    atoms = reader[-1]

    # 计算本次运行完后的新总步数
    total_steps = past_steps + n_steps
    new_traj_path = output_dir/f"md_{model}_{int(temperature_K)}K_{total_steps}steps.traj"
    new_log_path = output_dir/f"md_{model}_{int(temperature_K)}K_{total_steps}steps_log.csv"

    # 将旧轨迹文件内容复制/续接到新名称的文件中
    # (先将旧轨迹写入新文件，实现平滑继承)
    print(f"创建新的累计轨迹文件: {new_traj_path}")
    traj_out = Trajectory(new_traj_path, "w", atoms=reader[0])
    for frame in reader[1:]:
      traj_out.write(frame)

    mode = "a"  # 追加模式
  else:
    if restart:
      print("未找到符合条件的旧轨迹文件，将自动开启全新的模拟...")
    else:
      print("开始全新的 MD 模拟...")

    atoms = read(xyz_path)
    # 全新开始时随机初始化速度
    MaxwellBoltzmannDistribution(
        atoms,
        temperature_K=temperature_K,
        rng=np.random.default_rng(seed),
    )
    Stationary(atoms)

    total_steps = n_steps
    new_traj_path = output_dir/f"md_{model}_{int(temperature_K)}K_{total_steps}steps.traj"
    new_log_path = output_dir/f"md_{model}_{int(temperature_K)}K_{total_steps}steps_log.csv"
    mode = "w"

  # -------------------------------------------------------------
  # 2. 挂载 MACE 计算器与动力学积分器
  # -------------------------------------------------------------
  atoms = attach_calculator(atoms, model=model, device=device, dtype=dtype)

  dyn = Langevin(
      atoms,
      timestep=timestep_fs * units.fs,
      temperature_K=temperature_K,
      friction=friction,
  )

  # 挂载轨迹记录
  traj = Trajectory(new_traj_path, mode, atoms)
  dyn.attach(traj.write, interval=log_every)

  log_rows = []

  def log_step():
    epot = atoms.get_potential_energy()
    ekin = atoms.get_kinetic_energy()
    temp = ekin / (1.5 * units.kB * len(atoms))
    try:
      stress = atoms.get_stress(voigt=True)
      pressure_GPa = -np.mean(stress[:3]) / units.GPa
    except Exception:
      pressure_GPa = np.nan
    log_rows.append((dyn.get_time() / units.fs, epot, temp, pressure_GPa))

  dyn.attach(log_step, interval=log_every)

  print(
      f"开始运行本次的 {n_steps} 步 (累计将达到 {total_steps} 步,"
      f" 约 {total_steps * timestep_fs / 1000:.2f} ps)..."
  )
  dyn.run(n_steps)

  # 保存日志
  with open(new_log_path, "w") as f:
    f.write("time_fs,potential_energy_eV,temperature_K,pressure_GPa\n")
    for row in log_rows:
      f.write("{:.3f},{:.6f},{:.3f},{:.4f}\n".format(*row))

  print(
      f"运行完成！最新轨迹已生成: {new_traj_path}\n最新日志已保存:"
      f" {new_log_path}\n"
  )
  return new_traj_path


def analyze_msd(
    traj_path: str,
    timestep_fs: float,
    log_every: int,
    dt_between_frames_fs=None,
):
  traj = TrajReader(traj_path)
  positions = np.array([f.get_positions() for f in traj])
  disp = positions - positions[0]
  msd = (disp**2).sum(axis=2).mean(axis=1)

  dt = dt_between_frames_fs if dt_between_frames_fs else timestep_fs * log_every
  times_fs = np.arange(len(msd)) * dt

  fit_start = len(times_fs) // 4
  slope, intercept = np.polyfit(times_fs[fit_start:], msd[fit_start:], 1)
  D_cm2_s = slope * 1e-16 / 1e-15 / 6.0
  print(
      f"基于轨迹 {traj_path} (共 {len(traj)} 帧, 总时长"
      f" {times_fs[-1]/1000:.2f} ps) 计算："
  )
  print(f"估计的 3D 自扩散系数 D ~= {D_cm2_s:.3e} cm^2/s\n")
  return times_fs, msd, D_cm2_s


if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument("--xyz", default="data/raw/sb2te3_cr_81atom.xyz")
  p.add_argument("--model", default="medium-mpa-0")
  p.add_argument("--device", default="")
  p.add_argument("--dtype", default="float32")
  p.add_argument("--temp", type=float, default=600.0)
  p.add_argument("--timestep", type=float, default=1.0)
  p.add_argument("--friction", type=float, default=0.02)
  p.add_argument(
      "--steps",
      type=int,
      default=200000,
      help="本次运行增量的步数，例如 200000 代表跑 200ps",
  )
  p.add_argument("--log-every", type=int, default=20)
  p.add_argument(
      "--restart",
      action="store_true",
      help="自动寻找步数最大的旧轨迹文件并继承继续跑",
  )
  args = p.parse_args()

  latest_traj_file = run_nvt(
      xyz_path=args.xyz,
      temperature_K=args.temp,
      timestep_fs=args.timestep,
      friction=args.friction,
      n_steps=args.steps,
      log_every=args.log_every,
      restart=args.restart,
      model=args.model,
      device=args.device,
      dtype=args.dtype,
  )

  # 分析当前最新完整轨迹的 MSD
  analyze_msd(latest_traj_file, args.timestep, args.log_every)
