#!/usr/bin/env python3
"""Independently relax every reference torsion grid point with a model.

Each input is a QCSchema TorsionDrive JSON file produced by
``run_torsiondrive.py``. Every optimization starts from that grid point's
converged DFT geometry; no geometry is carried over between grid points.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms, units
from ase.constraints import FixInternals
from ase.io import write
from ase.optimize import BFGS


ROOT = Path(__file__).resolve().parent.parent
DATASETS = {
    "tnet500": ROOT / "data" / "TNet500_minimal",
    "biaryl": ROOT / "data" / "biaryl_minimal",
}
MACE_MODELS = {
    "mace-off23-s": ROOT / "models/mace_off23/MACE-OFF23_small.model",
    "mace-off23-m": ROOT / "models/mace_off23/MACE-OFF23_medium.model",
    "mace-off23-l": ROOT / "models/mace_off23/MACE-OFF23_large.model",
}
MODEL_GROUPS = {
    "tnet500": ("mace-off23-s", "mace-off23-m", "mace-off23-l"),
    "biaryl": ("mace-off23-s", "mace-off23-m", "mace-off23-l"),
}


def calculator_for(name: str, device: str):
    """Create the requested MACE-OFF23 calculator."""
    if name in MACE_MODELS:
        from mace.calculators import MACECalculator

        path = MACE_MODELS[name]
        if not path.is_file():
            raise FileNotFoundError(f"MACE checkpoint not found: {path}")
        return MACECalculator(model_paths=str(path), device=device,
                              default_dtype="float64")
    raise ValueError(f"Unsupported model {name!r}; choose one of {', '.join(MACE_MODELS)}")


def qcschema_atoms(molecule: dict[str, Any]) -> Atoms:
    # QCSchema geometry uses bohr; ASE uses Angstrom.
    xyz = np.asarray(molecule["geometry"], dtype=float).reshape((-1, 3)) * units.Bohr
    return Atoms(symbols=molecule["symbols"], positions=xyz)


def point_angle(key: str) -> float:
    return float(key.split(",")[0])


def relax_file(path: Path, dataset: str, model: str, calc, outdir: Path,
               fmax: float, max_steps: int, overwrite: bool) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("success"):
        raise ValueError(f"Reference torsiondrive failed: {path}")
    dihedrals = data["keywords"]["dihedrals"]
    if len(dihedrals) != 1:
        raise ValueError(f"Expected exactly one scanned dihedral in {path}")
    dihedral = [int(i) for i in dihedrals[0]]
    result_rows = []
    trajectory = outdir / f"{path.stem}__{model}.extxyz"
    rows_path = outdir / f"{path.stem}__{model}.json"
    if rows_path.exists() and not overwrite:
        print(f"Skipping completed output: {rows_path}", flush=True)
        return []
    images = []
    for grid_key in sorted(data["final_energies"], key=point_angle):
        # Fresh DFT reference geometry at each grid point; never use the prior
        # model-relaxed geometry as the next starting structure.
        atoms = qcschema_atoms(data["final_molecules"][grid_key])
        angle = point_angle(grid_key)
        if max(dihedral) >= len(atoms):
            raise ValueError(f"Dihedral index outside atom range in {path}")
        atoms.set_constraint(FixInternals(dihedrals_deg=[[angle, dihedral]]))
        atoms.calc = calc
        opt = BFGS(atoms, logfile=None)
        converged = opt.run(fmax=fmax, steps=max_steps)
        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        max_force = float(np.linalg.norm(forces, axis=1).max())
        row = {
            "dataset": dataset, "input_file": path.name, "model": model,
            "grid_angle_deg": angle, "dihedral_atoms_zero_based": dihedral,
            "energy_eV": energy, "converged": bool(converged),
            "max_force_eV_per_angstrom": max_force,
            "fmax_eV_per_angstrom": fmax, "steps": opt.nsteps,
            "symbols": atoms.get_chemical_symbols(),
            "positions_angstrom": atoms.get_positions().tolist(),
        }
        atoms.info.update({k: v for k, v in row.items()
                           if k not in ("symbols", "positions_angstrom")})
        images.append(atoms)
        result_rows.append(row)
        print(f"{path.name} {angle:7.1f}° {model}: E={energy:.9f} eV "
              f"max|F|={max_force:.5g} eV/Å converged={converged}", flush=True)
    outdir.mkdir(parents=True, exist_ok=True)
    write(trajectory, images, format="extxyz")
    rows_path.write_text(json.dumps(result_rows, indent=2) + "\n", encoding="utf-8")
    return result_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=DATASETS,
                        help="TNet500 or biaryl minimal QCSchema dataset")
    parser.add_argument("--model", choices=sorted(MACE_MODELS),
                        help="one model; default runs the dataset's full comparison group")
    parser.add_argument("--input-dir", type=Path, help="override the minimal data folder")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "torsion_relax")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fmax", type=float, default=0.01,
                        help="maximum force convergence threshold in eV/Å")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--limit", type=int, help="process at most N molecules per model")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.fmax <= 0 or args.max_steps < 1 or (args.limit is not None and args.limit < 1):
        parser.error("fmax, max-steps, and limit must be positive")
    input_dir = args.input_dir or DATASETS[args.dataset]
    files = sorted(input_dir.glob("*.json"))
    if not files:
        parser.error(f"no JSON torsiondrive results found in {input_dir}")
    if args.limit is not None:
        files = files[:args.limit]
    models = [args.model] if args.model else MODEL_GROUPS[args.dataset]
    for model in models:
        calc = calculator_for(model, args.device)
        for path in files:
            relax_file(path, args.dataset, model, calc, args.output_dir,
                       args.fmax, args.max_steps, args.overwrite)


if __name__ == "__main__":
    main()
