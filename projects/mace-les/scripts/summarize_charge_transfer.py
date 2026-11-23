#!/usr/bin/env python3
"""Summarize charge-transfer held-out metrics without pooling test folds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

MODEL_RE = re.compile(r"^(?:c1_)?(mace|maceles)_ct_q(m3|m2|m1|p0|p1)_seed(42|43)$")
Q_MAP = {"m3": -3, "m2": -2, "m1": -1, "p0": 0, "p1": 1}
Q_LABEL = {-3: "m3", -2: "m2", -1: "m1", 0: "p0", 1: "p1"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fmt(value, digits=4):
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    records = []
    trajectory_records = []
    test_hashes = {}
    for model_dir in sorted(p for p in args.runs_root.iterdir() if p.is_dir()):
        match = MODEL_RE.match(model_dir.name)
        if not match:
            continue
        family, q_tag, seed_text = match.groups()
        q, seed = Q_MAP[q_tag], int(seed_text)
        metrics_path = model_dir / "heldout" / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(metrics_path)
        report = json.loads(metrics_path.read_text())
        metrics = report["metrics"]
        if int(metrics["heldout_charge"]) != q:
            raise ValueError(f"held-out charge mismatch in {metrics_path}")
        test_hash = report["test_sha256"]
        expected_test_hash = test_hashes.setdefault(q, test_hash)
        if expected_test_hash != test_hash:
            raise ValueError(f"test split mismatch for q={q}")
        predictions_path = model_dir / "heldout" / "test_predictions.csv"
        if sha256(predictions_path) != report["predictions_sha256"]:
            raise ValueError(f"prediction hash mismatch in {predictions_path}")
        records.append({
            "charge": q,
            "charge_regime": "extrapolation" if q in (-3, 1) else "interpolation",
            "family": "MACE-Q-continuous" if family == "mace" else "MACELES-Q-continuous",
            "seed": seed,
            "test_frames": metrics["test_frames"],
            "test_trajectory_count": metrics["test_trajectory_count"],
            "raw_energy_mae_eV": metrics["raw_energy_mae_eV"],
            "raw_energy_rmse_eV": metrics["raw_energy_rmse_eV"],
            "force_component_mae_eV_A": metrics["force_component_mae_eV_A"],
            "force_component_rmse_eV_A": metrics["force_component_rmse_eV_A"],
            "within_charge_energy_spearman": metrics["within_charge_energy_spearman"],
            "whole_test_selection_regret_eV": metrics["whole_test_selection_regret_eV"],
            "test_sha256": test_hash,
            "metrics_sha256": sha256(metrics_path),
            "predictions_sha256": sha256(predictions_path),
        })
        for path in metrics["per_trajectory"]:
            trajectory_records.append({
                "charge": q,
                "family": "MACE-Q-continuous" if family == "mace" else "MACELES-Q-continuous",
                "seed": seed,
                "trajectory_id": path["trajectory_id"],
                "frames": path["frames"],
                "spearman": path["spearman"],
                "selection_regret_eV": path["selection_regret_eV"],
                "trajectory_centered_energy_mae_eV": path["trajectory_centered_energy_mae_eV"],
                "trajectory_centered_energy_rmse_eV": path["trajectory_centered_energy_rmse_eV"],
            })

    expected = {(q, family, seed) for q in Q_MAP.values() for family in ("mace", "maceles") for seed in (42, 43)}
    observed = set()
    for model_dir in args.runs_root.iterdir():
        match = MODEL_RE.match(model_dir.name) if model_dir.is_dir() else None
        if match:
            family, q_tag, seed = match.groups()
            observed.add((Q_MAP[q_tag], family, int(seed)))
    if observed != expected:
        raise ValueError(f"Expected 20 model evaluations; missing={sorted(expected-observed)}, extra={sorted(observed-expected)}")

    records.sort(key=lambda r: (r["charge"], r["family"], r["seed"]))
    trajectory_records.sort(key=lambda r: (r["charge"], r["family"], r["seed"], r["trajectory_id"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = args.output_dir / "charge_transfer_metrics.csv"
    trajectory_csv = args.output_dir / "charge_transfer_trajectory_metrics.csv"
    for path, rows in ((metrics_csv, records), (trajectory_csv, trajectory_records)):
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# C1 leave-one-charge-out results",
        "",
        "All metrics are reported per held-out charge; folds are not pooled.",
        "The q=−3 and q=+1 folds are charge-range extrapolation; q=−2, −1, and 0 are interpolation.",
        "Each q test contains four held-out trajectories from two frozen splits. There is no test-fitted energy offset.",
        "The benchmark transfers joint (charge, spin multiplicity) pairs; spin is fixed by q in this dataset, so their independent effects are not identifiable.",
        "Some test geometries have exact matches at other q in training/validation. This is matched-geometry transfer, not geometric OOD; see the fold manifest.",
        "MACELES energy/force scores do not establish physical latent-charge conservation or long-range electrostatic accuracy.",
        "",
        "| q | Regime | Family | Spearman, seeds 42/43 | Force RMSE range (eV/Å) | Raw energy MAE range (eV) | Whole-test regret range (eV) | Per-path centered energy RMSE range (eV) |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for q in (-3, -2, -1, 0, 1):
        for family in ("MACE-Q-continuous", "MACELES-Q-continuous"):
            group = [r for r in records if r["charge"] == q and r["family"] == family]
            spears = ", ".join(fmt(r["within_charge_energy_spearman"], 3) for r in group)
            force = [r["force_component_rmse_eV_A"] for r in group]
            energy = [r["raw_energy_mae_eV"] for r in group]
            regret = [r["whole_test_selection_regret_eV"] for r in group]
            centered = [
                t["trajectory_centered_energy_rmse_eV"]
                for t in trajectory_records
                if t["charge"] == q and t["family"] == family
            ]
            regime = "extrapolation" if q in (-3, 1) else "interpolation"
            lines.append(
                f"| {q:+d} | {regime} | {family} | {spears} | "
                f"{fmt(min(force),3)}–{fmt(max(force),3)} | "
                f"{fmt(min(energy),3)}–{fmt(max(energy),3)} | "
                f"{fmt(min(regret),4)}–{fmt(max(regret),4)} | "
                f"{fmt(min(centered),3)}–{fmt(max(centered),3)} |"
            )
    lines.extend([
        "",
        "The frame-level and per-trajectory outputs are in the adjacent CSV files. These are fixed-cell single-point metrics, not formation energies, transition levels, or relaxed minima.",
        "",
        f"Input test SHA256 by q: `{json.dumps({str(k): v for k, v in sorted(test_hashes.items())}, sort_keys=True)}`",
        "",
        f"Metrics CSV SHA256: `{sha256(metrics_csv)}`  ",
        f"Trajectory CSV SHA256: `{sha256(trajectory_csv)}`",
    ])
    report = args.output_dir / "CHARGE_TRANSFER_RESULTS.md"
    report.write_text("\n".join(lines) + "\n")
    print(json.dumps({"report": str(report), "metrics_csv": str(metrics_csv), "trajectory_csv": str(trajectory_csv)}, indent=2))


if __name__ == "__main__":
    main()
