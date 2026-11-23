#!/usr/bin/env python3
"""Evaluate a model on one wholly unseen charge state and report path metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from ase.io import read
from mace.calculators import MACECalculator


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--charge", type=int, choices=(-3, -2, -1, 0, 1), required=True)
    ap.add_argument("--training-seed", type=int, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("charge-transfer evaluation requires an allocated CUDA GPU")
    frames = read(args.test, index=":")
    if not frames:
        raise ValueError("empty test file")
    if any(int(a.info["total_charge"]) != args.charge for a in frames):
        raise ValueError("test file contains a charge other than the held-out charge")
    calculator = MACECalculator(
        model_paths=str(args.model),
        device="cuda",
        default_dtype="float32",
        info_keys={"total_charge": "total_charge", "total_spin": "total_spin"},
    )

    rows = []
    for index, atoms in enumerate(frames):
        atoms.calc = calculator
        pred_energy = float(atoms.get_potential_energy())
        pred_forces = np.asarray(atoms.get_forces(), dtype=float)
        ref_energy = float(atoms.info["REF_energy"])
        ref_forces = np.asarray(atoms.arrays["REF_forces"], dtype=float)
        if not np.isfinite(pred_energy) or not np.isfinite(pred_forces).all():
            raise ValueError(f"nonfinite prediction at test frame {index}")
        rows.append({
            "frame_index": index,
            "charge": args.charge,
            "total_spin": int(atoms.info["total_spin"]),
            "trajectory_id": str(atoms.info.get("trajectory_id", "")),
            "distortion_label": str(atoms.info.get("distortion_label", "")),
            "source_outcar": str(atoms.info.get("source_outcar", "")),
            "source_frame_index": int(atoms.info.get("source_frame_index", -1)),
            "ref_energy_eV": ref_energy,
            "pred_energy_eV": pred_energy,
            "energy_error_eV": pred_energy - ref_energy,
            "ref_forces": ref_forces,
            "pred_forces": pred_forces,
            "force_component_rmse_eV_A": float(np.sqrt(np.mean((pred_forces - ref_forces) ** 2))),
        })

    ref_energy = np.asarray([r["ref_energy_eV"] for r in rows])
    pred_energy = np.asarray([r["pred_energy_eV"] for r in rows])
    energy_error = pred_energy - ref_energy
    force_error = np.concatenate([
        (r["pred_forces"] - r["ref_forces"]).reshape(-1) for r in rows
    ])
    dft_best = min(rows, key=lambda r: r["ref_energy_eV"])
    model_best = min(rows, key=lambda r: r["pred_energy_eV"])
    per_trajectory = []
    by_trajectory = defaultdict(list)
    for row in rows:
        by_trajectory[row["trajectory_id"]].append(row)
    for trajectory, path_rows in sorted(by_trajectory.items()):
        dft_path_best = min(path_rows, key=lambda r: r["ref_energy_eV"])
        model_path_best = min(path_rows, key=lambda r: r["pred_energy_eV"])
        ref_path_energy = np.asarray([r["ref_energy_eV"] for r in path_rows])
        pred_path_energy = np.asarray([r["pred_energy_eV"] for r in path_rows])
        ref_path_relative = ref_path_energy - np.mean(ref_path_energy)
        pred_path_relative = pred_path_energy - np.mean(pred_path_energy)
        relative_error = pred_path_relative - ref_path_relative
        per_trajectory.append({
            "trajectory_id": trajectory,
            "frames": len(path_rows),
            "spearman": spearman(
                ref_path_energy,
                pred_path_energy,
            ),
            "selection_regret_eV": float(model_path_best["ref_energy_eV"] - dft_path_best["ref_energy_eV"]),
            "trajectory_centered_energy_mae_eV": float(np.mean(np.abs(relative_error))),
            "trajectory_centered_energy_rmse_eV": float(np.sqrt(np.mean(relative_error ** 2))),
        })

    metrics = {
        "heldout_charge": args.charge,
        "test_frames": len(rows),
        "test_trajectory_count": len(by_trajectory),
        "test_trajectory_ids": sorted(by_trajectory),
        "test_spin_multiplicities": sorted({int(a.info["total_spin"]) for a in frames}),
        "raw_energy_mae_eV": float(np.mean(np.abs(energy_error))),
        "raw_energy_rmse_eV": float(np.sqrt(np.mean(energy_error ** 2))),
        "force_component_mae_eV_A": float(np.mean(np.abs(force_error))),
        "force_component_rmse_eV_A": float(np.sqrt(np.mean(force_error ** 2))),
        "within_charge_energy_spearman": spearman(ref_energy, pred_energy),
        "whole_test_selection_regret_eV": float(model_best["ref_energy_eV"] - dft_best["ref_energy_eV"]),
        "dft_selected_trajectory": dft_best["trajectory_id"],
        "model_selected_trajectory": model_best["trajectory_id"],
        "per_trajectory": per_trajectory,
        "energy_offset_calibration": "none; held-out charge is absent from validation",
        "interpretation_scope": "joint charge/multiplicity transfer; spin multiplicity is fixed by q in this dataset and its independent effect is not identifiable",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "test_predictions.csv"
    with predictions_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=(
            "frame_index", "charge", "total_spin", "trajectory_id", "distortion_label",
            "source_outcar", "source_frame_index", "ref_energy_eV", "pred_energy_eV",
            "energy_error_eV", "force_component_rmse_eV_A",
        ))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})
    report = {
        "model_path": str(args.model),
        "model_sha256": sha256(args.model),
        "model_family": "MACE or MACELES with continuous total_charge and categorical total_spin embeddings",
        "calculator_info_keys": {"total_charge": "total_charge", "total_spin": "total_spin"},
        "training_seed": args.training_seed,
        "test_path": str(args.test),
        "test_sha256": sha256(args.test),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "metrics": metrics,
        "predictions_path": str(predictions_path),
        "predictions_sha256": sha256(predictions_path),
        "limitations": [
            "held-out charge is absent from training and validation; no test-fitted charge offset is applied",
            f"{len(by_trajectory)} held-out distortion trajectories from two frozen test splits; all are drawn from the same source pool",
            "fixed-cell, same V_Sb1 defect family and PBE+D3 labels; not charge-state formation energy or transition-level evidence",
            "minimum selection is among held-out single-point frames, not a relaxed ground-state search",
        ],
    }
    report_path = args.output_dir / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"metrics": metrics, "metrics_path": str(report_path), "predictions_path": str(predictions_path)}, indent=2))


if __name__ == "__main__":
    main()
