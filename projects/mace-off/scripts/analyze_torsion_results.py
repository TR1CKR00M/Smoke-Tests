#!/usr/bin/env python3
"""Summarize MACE-OFF torsion profiles and aligned TNet500 geometry RMSD.

Expected model output files are ``results/torsion_relax/<stem>__<model>.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase import units
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parent.parent
DATASETS = {
    "tnet500": ROOT / "data" / "TNet500_minimal",
    "biaryl": ROOT / "data" / "biaryl_minimal",
}
MODELS = ("mace-off23-s", "mace-off23-m", "mace-off23-l")
KCAL_PER_EV = 23.060548867
KCAL_PER_HARTREE = 627.5094740631
COLORS = {"mace-off23-s": "#4C78A8", "mace-off23-m": "#F58518",
          "mace-off23-l": "#54A24B"}
LABELS = {"mace-off23-s": "MACE-OFF23(S)", "mace-off23-m": "MACE-OFF23(M)",
          "mace-off23-l": "MACE-OFF23(L)"}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_angles(angles):
    return np.mod(np.asarray(angles, dtype=float), 360.0)


def profile(values, scale):
    values = np.asarray(values, dtype=float)
    return (values - np.nanmin(values)) * scale


def kabsch_rmsd(reference, predicted):
    reference = np.asarray(reference, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if reference.shape != predicted.shape:
        raise ValueError(f"coordinate shape mismatch: {reference.shape} vs {predicted.shape}")
    ref = reference - reference.mean(axis=0)
    pred = predicted - predicted.mean(axis=0)
    u, _, vt = np.linalg.svd(pred.T @ ref)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    aligned = pred @ rotation
    return float(np.sqrt(np.mean(np.sum((aligned - ref) ** 2, axis=1))))


def read_reference_file(dataset: str, stem: str):
    ref = read_json(DATASETS[dataset] / f"{stem}.json")
    keys = sorted(ref["final_energies"], key=lambda key: float(key.split(",")[0]))
    angles = normalized_angles([float(key.split(",")[0]) for key in keys])
    energies_hartree = np.asarray([ref["final_energies"][key] for key in keys], dtype=float)
    molecules = [ref["final_molecules"][key] for key in keys]
    geometries = [np.asarray(mol["geometry"], dtype=float).reshape((-1, 3)) * units.Bohr
                  for mol in molecules]
    return angles, energies_hartree * KCAL_PER_HARTREE, geometries


def gather(results_dir: Path):
    profiles = defaultdict(list)
    rmsds = defaultdict(list)
    counts = defaultdict(lambda: {"molecules": set(), "points": 0, "nonconverged": 0,
                                  "bad_molecules": set()})
    reference_profiles = {}
    files = sorted(results_dir.glob("*__mace-off23-*.json"))
    if not files:
        raise FileNotFoundError(f"No relaxation result JSON files in {results_dir}")

    for path in files:
        rows = read_json(path)
        if not rows:
            continue
        first = rows[0]
        dataset, model, stem = first["dataset"], first["model"], Path(first["input_file"]).stem
        if dataset not in DATASETS:
            continue
        angles = np.asarray([row["grid_angle_deg"] for row in rows], dtype=float) % 360.0
        energies_kcal = np.asarray([row["energy_eV"] for row in rows], dtype=float) * KCAL_PER_EV
        order = np.argsort(angles)
        angles, energies_kcal = angles[order], energies_kcal[order]
        if len(rows) != 24:
            print(f"Warning: {path.name}: expected 24 points, found {len(rows)}")
            counts[(dataset, model)]["bad_molecules"].add(stem)
        if not all(row["converged"] for row in rows):
            counts[(dataset, model)]["nonconverged"] += sum(not row["converged"] for row in rows)

        ref_angles, ref_energy, ref_geometries = read_reference_file(dataset, stem)
        reference_profiles[(dataset, stem)] = (ref_angles, profile(ref_energy, 1.0))
        profiles[(dataset, model)].append((stem, angles, profile(energies_kcal, 1.0)))
        counts[(dataset, model)]["molecules"].add(stem)
        counts[(dataset, model)]["points"] += len(rows)

        if dataset == "tnet500":
            row_by_angle = {round(float(row["grid_angle_deg"]) % 360.0, 6): row for row in rows}
            for angle, ref_xyz in zip(ref_angles, ref_geometries, strict=True):
                row = row_by_angle.get(round(float(angle), 6))
                if row is None:
                    continue
                rmsds[model].append(kabsch_rmsd(ref_xyz, row["positions_angstrom"]))

    return profiles, reference_profiles, rmsds, counts


def draw_profiles(ax, dataset, profiles, references):
    data_models = [model for model in MODELS if profiles.get((dataset, model))]
    for model in data_models:
        group = profiles[(dataset, model)]
        curves = []
        grid = np.arange(0.0, 360.0, 15.0)
        for _, angles, energies in group:
            curve = np.interp(grid, np.r_[angles, angles[0] + 360.0],
                              np.r_[energies, energies[0]])
            curves.append(curve)
            smooth_x = np.linspace(0.0, 360.0, 361)
            smooth_y = CubicSpline(np.r_[grid, 360.0], np.r_[curve, curve[0]],
                                   bc_type="periodic")(smooth_x)
            ax.plot(smooth_x, smooth_y, color=COLORS[model], alpha=0.09, linewidth=0.55)
        mean_curve = np.mean(curves, axis=0)
        smooth_x = np.linspace(0.0, 360.0, 721)
        smooth_y = CubicSpline(np.r_[grid, 360.0], np.r_[mean_curve, mean_curve[0]],
                               bc_type="periodic")(smooth_x)
        ax.plot(smooth_x, smooth_y, color=COLORS[model], linewidth=2.2,
                label=f"{LABELS[model]} mean (n={len(group)})")
    refs = [(stem, vals) for (ds, stem), vals in references.items() if ds == dataset]
    if refs:
        grid = np.arange(0.0, 360.0, 15.0)
        curves = []
        for angles, energies in (item[1] for item in refs):
            angles = normalized_angles(angles)
            order = np.argsort(angles)
            angles, energies = angles[order], np.asarray(energies)[order]
            curves.append(np.interp(grid, np.r_[angles, angles[0] + 360.0],
                                    np.r_[energies, energies[0]]))
        mean_curve = np.mean(curves, axis=0)
        smooth_x = np.linspace(0.0, 360.0, 721)
        smooth_y = CubicSpline(np.r_[grid, 360.0], np.r_[mean_curve, mean_curve[0]],
                               bc_type="periodic")(smooth_x)
        ax.plot(smooth_x, smooth_y, color="#222222", linewidth=2.0, linestyle="--",
                label=f"DFT reference mean (n={len(refs)})")
    ax.set_title(f"{dataset.upper()} torsion profiles")
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 60))
    ax.set_xlabel("Dihedral angle (°)")
    ax.set_ylabel("Relative energy (kcal/mol)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, frameon=False)


def make_figure(profiles, references, rmsds, output: Path):
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.45, 0.95))
    ax_tnet = fig.add_subplot(grid[0, 0])
    ax_biaryl = fig.add_subplot(grid[1, 0])
    draw_profiles(ax_tnet, "tnet500", profiles, references)
    draw_profiles(ax_biaryl, "biaryl", profiles, references)

    ax_rmsd = fig.add_subplot(grid[:, 1])
    rmsd_means = [np.mean(rmsds[m]) if rmsds[m] else np.nan for m in MODELS]
    xs = np.arange(len(MODELS))
    ax_rmsd.bar(xs, rmsd_means, color=[COLORS[m] for m in MODELS])
    ax_rmsd.set_xticks(xs, [LABELS[m].replace("MACE-OFF23", "") for m in MODELS], rotation=20)
    ax_rmsd.set_ylabel("Mean aligned RMSD (Å)")
    ax_rmsd.set_title("TNet500 geometry RMSD")
    ax_rmsd.grid(axis="y", alpha=0.2)

    fig.suptitle("MACE-OFF23 constrained torsion evaluation", fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def write_summary(path, counts, rmsds):
    fields = ["dataset", "model", "n_molecules", "n_grid_points", "n_nonconverged",
              "mean_tnet500_rmsd_angstrom"]
    rows = []
    for dataset in ("tnet500", "biaryl"):
        for model in MODELS:
            count = counts[(dataset, model)]
            rows.append({
                "dataset": dataset, "model": LABELS[model],
                "n_molecules": len(count["molecules"]), "n_grid_points": count["points"],
                "n_nonconverged": count["nonconverged"],
                "mean_tnet500_rmsd_angstrom": (float(np.mean(rmsds[model]))
                                                if dataset == "tnet500" and rmsds[model] else ""),
            })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "torsion_relax")
    parser.add_argument("--figure", type=Path,
                        default=ROOT / "results" / "figures" / "fig2_torsion.png")
    parser.add_argument("--summary", type=Path,
                        default=ROOT / "results" / "metrics" / "fig2_metrics.csv")
    parser.add_argument("--allow-partial", action="store_true",
                        help=("create a partial figure when not all molecules/points "
                              "are present"))
    args = parser.parse_args()
    profiles, references, rmsds, counts = gather(args.results_dir)
    missing = []
    for dataset, input_dir in DATASETS.items():
        expected_stems = {path.stem for path in input_dir.glob("*.json")}
        for model in MODELS:
            count = counts[(dataset, model)]
            actual = count["molecules"]
            absent = expected_stems - actual
            bad_points = count["points"] != 24 * len(actual)
            if absent or bad_points or count["bad_molecules"]:
                missing.append(f"{dataset}/{model}: {len(actual)}/{len(expected_stems)} molecules, "
                               f"{count['points']} grid points, "
                               f"{len(count['bad_molecules'])} malformed point files")
    if missing and not args.allow_partial:
        parser.error("incomplete relaxation results; " + "; ".join(missing)
                     + ". Finish all molecule/grid-point relaxations or pass --allow-partial.")
    if missing:
        print("WARNING: partial Figure 2 input: " + "; ".join(missing), flush=True)
    make_figure(profiles, references, rmsds, args.figure)
    write_summary(args.summary, counts, rmsds)
    for (dataset, model), count in sorted(counts.items()):
        print(f"{dataset} {LABELS[model]}: {len(count['molecules'])} molecules, "
              f"{count['points']} grid points, {count['nonconverged']} nonconverged")
    print(f"Figure: {args.figure}\nMetrics: {args.summary}")


if __name__ == "__main__":
    main()
