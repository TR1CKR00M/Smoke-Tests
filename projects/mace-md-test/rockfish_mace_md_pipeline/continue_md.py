#!/usr/bin/env python
"""Self-contained, non-destructive ASE/MACE Rockfish continuation driver."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from ase import units
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from mace.calculators import MACECalculator


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    inp = Path(cfg["input_xyz"]).expanduser().resolve()
    model = Path(cfg["model_path"]).expanduser().resolve()
    out = Path(cfg["output_dir"]).expanduser().resolve()
    segment = str(cfg["segment"])
    out.mkdir(parents=True, exist_ok=True)
    segment_path = out / f"segment_{segment}.extxyz"
    checkpoint_path = out / "latest_restart.extxyz"
    status_path = out / "status.json"
    if not inp.is_file():
        raise FileNotFoundError(inp)
    if not model.is_file():
        raise FileNotFoundError(model)
    if segment_path.exists():
        raise FileExistsError(f"Refusing to overwrite {segment_path}")

    atoms = read(inp, index=-1)
    expected_atoms = cfg.get("expected_atoms")
    if expected_atoms is not None and len(atoms) != int(expected_atoms):
        raise RuntimeError(f"expected {expected_atoms} atoms, got {len(atoms)}")
    momenta = atoms.get_momenta()
    if cfg.get("initialize_momenta", False):
        MaxwellBoltzmannDistribution(
            atoms,
            temperature_K=float(cfg["temperature_K"]),
            rng=np.random.default_rng(int(cfg["seed"])),
        )
        momenta = atoms.get_momenta()
    if momenta is None or not np.isfinite(momenta).all():
        raise RuntimeError(
            "last frame has no finite momenta; "
            "set initialize_momenta=true only for a new run"
        )
    if not np.isfinite(atoms.get_positions()).all():
        raise RuntimeError("last frame has non-finite positions")

    atoms.calc = MACECalculator(model_path=str(model), device=cfg.get("device", "cuda"))
    atoms.set_momenta(momenta)
    dyn = Langevin(
        atoms,
        timestep=float(cfg["timestep_fs"]) * units.fs,
        temperature_K=float(cfg["temperature_K"]),
        friction=float(cfg["friction"]),
        rng=np.random.default_rng(int(cfg["seed"])),
    )
    start_step = int(cfg["start_step"])
    interval = int(cfg["interval"])
    segment_path_str = str(segment_path)

    def annotate() -> None:
        absolute_step = start_step + dyn.nsteps
        atoms.info["absolute_step"] = absolute_step
        atoms.info["time_fs"] = absolute_step * float(cfg["timestep_fs"])
        atoms.info["segment"] = segment
        atoms.info["source_input_sha256"] = sha256(inp)
        atoms.info["model_sha256"] = sha256(model)

    def write_state() -> None:
        annotate()
        write(segment_path_str, atoms, format="extxyz", append=True)

    def checkpoint() -> None:
        annotate()
        write(checkpoint_path, atoms, format="extxyz")
        status_path.write_text(json.dumps({
            "segment": segment,
            "absolute_step": int(start_step + dyn.nsteps),
            "time_fs": float((start_step + dyn.nsteps) * float(cfg["timestep_fs"])),
            "temperature_K": float(atoms.get_temperature()),
            "input_xyz": str(inp),
            "input_sha256": sha256(inp),
            "model_path": str(model),
            "model_sha256": sha256(model),
            "output_segment": str(segment_path),
        }, indent=2) + "\n")

    write_state()
    dyn.attach(write_state, interval=interval)
    dyn.attach(checkpoint, interval=int(cfg["checkpoint_steps"]))
    dyn.run(int(cfg["steps"]))
    checkpoint()
    print(json.dumps({
        "segment": segment,
        "final_step": int(start_step + dyn.nsteps),
        "final_time_ps": float((start_step + dyn.nsteps) * float(cfg["timestep_fs"]) / 1000.0),
        "segment_path": str(segment_path),
        "checkpoint_path": str(checkpoint_path),
    }, indent=2))


if __name__ == "__main__":
    main()
