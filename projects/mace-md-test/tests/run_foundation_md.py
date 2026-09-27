#!/usr/bin/env python3
"""
run_foundation_md.py

Template script for reproducing the *shape* of Figure 2b in
"Migration as a Probe..." (arXiv:2509.00090), using an off-the-shelf MACE
foundation model -- NO training or fine-tuning required.

What this script does:
  1. Load the 81-atom Cr-doped Sb2Te3 slab (sb2te3_cr_81atom.xyz).
  2. Attach a pretrained MACE foundation-model calculator (mace.calculators.mace_mp).
  3. Run an NVT MD trajectory (Langevin thermostat) at a target temperature.
  4. Log potential energy, temperature, and (virial) pressure every step,
     the same three quantities plotted in Fig. 2b.
  5. Dump the trajectory to disk so you can later compute MSD / diffusivity
     (see the `analyze_msd` helper at the bottom).

Notes on what is intentionally NOT here:
  - No DFT, no QE. This is a pure MLFF-driven classical MD.
  - No training data, no fine-tuning. `mace_mp(...)` downloads a public,
    already-trained checkpoint the first time you run it and caches it.
  - The structure below is one endpoint image pulled from the repo's NEB
    dataset (data/raw_compact/revision1/81_atom_2D_421_Diffusion_traj/neb_1),
    with the DFT cell from templates/qe_81_atom_engine_template_clean.in
    stitched back in. It is chemically the same system (Cr:Sb2Te3, 81 atoms)
    used throughout the paper, but it is NOT guaranteed to be the exact
    equilibrated 600 K snapshot that seeded Fig. 2 -- the repo does not ship
    that raw MD trajectory (see accompanying note). Treat this as a
    reasonable starting geometry for your own equilibration run, not as a
    byte-for-byte reproduction input.

Usage:
    python run_foundation_md.py --model medium-mpa-0 --temp 600 --steps 20000
"""


import argparse
import os
from pathlib import Path

import numpy as np
from ase.io import read, Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase import units

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

output_dir = Path("results/raw")
output_dir.mkdir(parents=True, exist_ok=True)

def build_atoms(xyz_path: str):
    atoms = read(xyz_path)
    return atoms


def attach_calculator(atoms, model: str, device: str = "", dtype: str = "float32"):
    """
    model: one of the MACE-MP foundation model keys, e.g.
        "small", "medium", "large"          -> original MACE-MP-0
        "small-0b", "medium-0b"              -> MACE-MP-0b
        "small-0b2", "medium-0b2", "large-0b2" -> MACE-MP-0b2
        "medium-0b3"                          -> MACE-MP-0b3
        "medium-mpa-0"                        -> MACE-MPA-0 (current mace_mp default)
        "small-omat-0", "medium-omat-0"       -> MACE-OMAT-0 (ASL license)
        "mace-matpes-pbe-0", "mace-matpes-r2scan-0"
        "mh-0", "mh-1"
    See https://github.com/ACEsuit/mace-foundations for the full, current list.
    """
    from mace.calculators import mace_mp

    calc = mace_mp(model=model, device=device, default_dtype=dtype)
    atoms.calc = calc
    return atoms


def run_nvt(
    atoms,
    temperature_K: float = 600.0,
    timestep_fs: float = 1.0,
    friction: float = 0.02,
    n_steps: int = 20000,
    log_every: int = 20,
    traj_path: str = "md_foundation.traj",
    log_path: str = "md_foundation_log.csv",
    seed: int = 42,
):
    MaxwellBoltzmannDistribution(
        atoms,
        temperature_K=temperature_K,
        rng=np.random.default_rng(seed)
        )
    Stationary(atoms)

    dyn = Langevin(
        atoms,
        timestep=timestep_fs * units.fs,
        temperature_K=temperature_K,
        friction=friction,
    )

    traj = Trajectory(traj_path, "w", atoms)
    dyn.attach(traj.write, interval=log_every)

    log_rows = []

    def log_step():
        epot = atoms.get_potential_energy()
        ekin = atoms.get_kinetic_energy()
        temp = ekin / (1.5 * units.kB * len(atoms))
        try:
            stress = atoms.get_stress(voigt=True)  # eV/A^3, requires calculator stress support
            pressure_GPa = -np.mean(stress[:3]) / units.GPa
        except Exception:
            pressure_GPa = np.nan
        log_rows.append((dyn.get_time() / units.fs, epot, temp, pressure_GPa))

    dyn.attach(log_step, interval=log_every)
    dyn.run(n_steps)

    with open(log_path, "w") as f:
        f.write("time_fs,potential_energy_eV,temperature_K,pressure_GPa\n")
        for row in log_rows:
            f.write("{:.3f},{:.6f},{:.3f},{:.4f}\n".format(*row))

    print(f"Wrote trajectory to {traj_path} and thermo log to {log_path}")
    return log_rows


def analyze_msd(traj_path: str, timestep_fs: float, log_every: int, dt_between_frames_fs=None):
    """
    Very simple MSD -> diffusion coefficient estimate from the saved Trajectory.
    D = slope(MSD vs t) / (2 * dim), dim=3 for 3D diffusion (use dim=2 if you
    only care about in-plane diffusion for a 2D slab).
    """
    from ase.io.trajectory import Trajectory as TrajReader

    traj = TrajReader(traj_path)
    positions = [f.get_positions() for f in traj]
    positions = np.array(positions)  # (n_frames, n_atoms, 3)
    disp = positions - positions[0]
    msd = (disp ** 2).sum(axis=2).mean(axis=1)  # mean over atoms

    dt = dt_between_frames_fs if dt_between_frames_fs else timestep_fs * log_every
    times_fs = np.arange(len(msd)) * dt

    # Linear fit over the (typically) diffusive later portion of the run
    fit_start = len(times_fs) // 4
    slope, intercept = np.polyfit(times_fs[fit_start:], msd[fit_start:], 1)
    D_cm2_s = slope * 1e-16 / 1e-15 / 6.0  # convert A^2/fs -> cm^2/s, /6 for 3D MSD=6Dt
    print(f"Estimated 3D self-diffusion coefficient D ~= {D_cm2_s:.3e} cm^2/s")
    return times_fs, msd, D_cm2_s


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--xyz", default="sb2te3_cr_81atom.xyz")
    p.add_argument("--model", default="medium-mpa-0",
                   help="MACE foundation model key, see attach_calculator() docstring")
    p.add_argument("--device", default="")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--temp", type=float, default=600.0)
    p.add_argument("--timestep", type=float, default=1.0)
    p.add_argument("--friction", type=float, default=0.02)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--log-every", type=int, default=20)
    args = p.parse_args()

    atoms = build_atoms("data/raw/sb2te3_cr_81atom.xyz")
    atoms = attach_calculator(atoms, model=args.model, device=args.device, dtype=args.dtype)

    tag = args.model
    run_nvt(
        atoms,
        temperature_K=args.temp,
        timestep_fs=args.timestep,
        friction=args.friction,
        n_steps=args.steps,
        log_every=args.log_every,
        traj_path=output_dir/f"md_{tag}_{int(args.temp)}K.traj",
        log_path=output_dir/f"md_{tag}_{int(args.temp)}K_log.csv",
    )

    analyze_msd(output_dir/f"md_{tag}_{int(args.temp)}K.traj", args.timestep, args.log_every)
