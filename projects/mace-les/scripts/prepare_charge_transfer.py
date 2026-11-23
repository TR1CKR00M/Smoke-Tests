#!/usr/bin/env python3
"""Prepare leave-one-charge-out folds from a frozen trajectory-group split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from ase.io import read, write

Q_VALUES = (-3, -2, -1, 0, 1)
SPIN_MULTIPLICITY = {-3: 1, -2: 2, -1: 1, 0: 2, 1: 1}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def charge_counts(frames):
    return dict(sorted(Counter(int(a.info["total_charge"]) for a in frames).items()))


def trajectories(frames):
    return sorted({str(a.info.get("trajectory_id", "")) for a in frames})


def geometry_key(atoms) -> str:
    fractional = atoms.get_scaled_positions(wrap=True)
    atom_rows = sorted(
        (int(z), *np.round(frac, 8).tolist())
        for z, frac in zip(atoms.numbers, fractional)
    )
    h = hashlib.sha256()
    h.update(np.round(np.asarray(atoms.cell), 8).tobytes())
    h.update(np.asarray(atom_rows, dtype=np.float64).tobytes())
    return h.hexdigest()


def validate_frames(name: str, frames) -> None:
    for index, atoms in enumerate(frames):
        q = int(atoms.info.get("total_charge", 999))
        spin = int(atoms.info.get("total_spin", 0))
        if q not in SPIN_MULTIPLICITY:
            raise ValueError(f"{name} frame {index}: unexpected total_charge={q}")
        if spin != SPIN_MULTIPLICITY[q]:
            raise ValueError(f"{name} frame {index}: q={q} expects total_spin={SPIN_MULTIPLICITY[q]}, found {spin}")
        if int(atoms.info.get("charge", 999)) != q:
            raise ValueError(f"{name} frame {index}: charge alias differs from total_charge")
        if int(atoms.info.get("spin", 0)) != spin:
            raise ValueError(f"{name} frame {index}: spin alias differs from total_spin")
        if len(atoms) != 59:
            raise ValueError(f"{name} frame {index}: expected 59 atoms, found {len(atoms)}")
        if not np.isfinite(float(atoms.info.get("REF_energy", np.nan))):
            raise ValueError(f"{name} frame {index}: missing or nonfinite REF_energy")
        forces = np.asarray(atoms.arrays.get("REF_forces", []), dtype=float)
        if forces.shape != (len(atoms), 3) or not np.isfinite(forces).all():
            raise ValueError(f"{name} frame {index}: missing, malformed, or nonfinite REF_forces")
        if not str(atoms.info.get("trajectory_id", "")) or not str(atoms.info.get("source_outcar", "")):
            raise ValueError(f"{name} frame {index}: missing trajectory/source provenance")


def q_label(q: int) -> str:
    return f"m{abs(q)}" if q < 0 else f"p{q}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--valid", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--test-secondary", type=Path)
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()

    input_paths = {name: getattr(args, name) for name in ("train", "valid", "test")}
    if args.test_secondary is not None:
        input_paths["test_secondary"] = args.test_secondary
    source = {name: read(path, index=":") for name, path in input_paths.items()}
    for name, frames in source.items():
        if not frames:
            raise ValueError(f"{name} split is empty")
        missing = [i for i, atoms in enumerate(frames) if "total_charge" not in atoms.info]
        if missing:
            raise ValueError(f"{name} frames missing total_charge: {missing[:5]}")
        validate_frames(name, frames)
    all_charges = {
        int(atoms.info["total_charge"])
        for frames in source.values()
        for atoms in frames
    }
    if all_charges != set(Q_VALUES):
        raise ValueError(f"Expected charge states {Q_VALUES}; found {sorted(all_charges)}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": "leave-one-charge-state-out from frozen trajectory-disjoint split",
        "charge_values": list(Q_VALUES),
        "spin_multiplicity_by_charge": {str(q): SPIN_MULTIPLICITY[q] for q in Q_VALUES},
        "charge_spin_independent_effects_identifiable": False,
        "source_files": {
            name: {"path": f"{path.parent.name}/{path.name}", "sha256": sha256(path)}
            for name, path in input_paths.items()
        },
        "folds": {},
        "limitations": [
            "held-out charge is absent from both training and validation",
            "test trajectories are the two paths per charge from each frozen C1 test split; charge-state frames are absent from train/validation",
            "no validation energy offset exists for the unseen charge; evaluate raw energy errors, forces, ranking, and selection regret",
            "same 59-atom V_Sb1 cell; this does not test defect-type or supercell-size transfer",
        ],
    }

    for q in Q_VALUES:
        train = [a for a in source["train"] if int(a.info["total_charge"]) != q]
        valid = [a for a in source["valid"] if int(a.info["total_charge"]) != q]
        test_sources = [source["test"]]
        if "test_secondary" in source:
            test_sources.append(source["test_secondary"])
        test = [a for frames in test_sources for a in frames if int(a.info["total_charge"]) == q]
        if not train or not valid or not test:
            raise ValueError(f"Empty train/valid/test fold for q={q}")
        train_q = {int(a.info["total_charge"]) for a in train}
        valid_q = {int(a.info["total_charge"]) for a in valid}
        if q in train_q or q in valid_q:
            raise AssertionError(f"Held-out q={q} leaked into train/valid")
        if train_q != valid_q or train_q != set(Q_VALUES) - {q}:
            raise ValueError(f"Unexpected train/valid charges in q={q} fold")
        expected_paths = 4 if "test_secondary" in source else 2
        if len(trajectories(test)) != expected_paths:
            raise ValueError(f"Expected {expected_paths} held-out paths for q={q}; got {trajectories(test)}")
        if "test_secondary" in source:
            primary_paths = {str(a.info.get("trajectory_id", "")) for a in source["test"] if int(a.info["total_charge"]) == q}
            secondary_paths = {str(a.info.get("trajectory_id", "")) for a in source["test_secondary"] if int(a.info["total_charge"]) == q}
            if len(primary_paths) != 2 or len(secondary_paths) != 2 or primary_paths & secondary_paths:
                raise ValueError(f"Primary/secondary test paths are not two-per-split and disjoint for q={q}")

        other_q_training_keys = {
            geometry_key(a)
            for a in source["train"] + source["valid"]
            if int(a.info["total_charge"]) != q
        }
        geometry_matches = [a for a in test if geometry_key(a) in other_q_training_keys]

        fold_dir = args.output_root / f"q_{q_label(q)}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        outputs = {}
        for split_name, frames in (("train", train), ("valid", valid), ("test", test)):
            path = fold_dir / f"{split_name}.xyz"
            write(path, frames, format="extxyz")
            outputs[split_name] = {
                "path": os.path.relpath(path, args.output_root),
                "frames": len(frames),
                "charge_counts": charge_counts(frames),
                "trajectory_ids": trajectories(frames),
                "sha256": sha256(path),
            }
        manifest["folds"][str(q)] = {
            "heldout_charge": q,
            "train": outputs["train"],
            "validation": outputs["valid"],
            "test": outputs["test"],
            "test_cross_charge_geometry_matches": {
                "matching_test_frames": len(geometry_matches),
                "test_frames": len(test),
                "matched_trajectory_ids": sorted({str(a.info.get("trajectory_id", "")) for a in geometry_matches}),
                "criterion": "same cell and sorted atomic numbers/fractional coordinates rounded to 1e-8",
                "interpretation": "matched geometry across Q is allowed; the held-out Q labels are absent from train/validation",
            },
        }

    manifest_path = args.output_root / "charge_transfer_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "sha256": sha256(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
