#!/usr/bin/env python3
"""Reproduce the MACE-MD figures from committed CSV data.

The plotting grammar follows the Plot Atlas ``timecourse`` panel: one
independent ordered trajectory per model, semantic colours, explicit units,
heavy axes, bold ticks, and no decorative grid.  The implementation is kept
self-contained so a checkout does not depend on a private Plot Atlas install.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "results" / "raw"
DEFAULT_FIGURES = ROOT / "results" / "figures"
DEFAULT_METADATA = ROOT / "data" / "metadata"

PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9.5,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11,
            "axes.linewidth": 1.25,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "savefig.dpi": 450,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def read_logs(paths: list[Path]) -> dict[str, pd.DataFrame]:
    if not paths:
        raise ValueError("No CSV files were supplied")
    data: dict[str, pd.DataFrame] = {}
    reference_time: np.ndarray | None = None
    required = {"time_fs", "potential_energy_eV", "temperature_K", "pressure_GPa"}
    for path in paths:
        frame = pd.read_csv(path)
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        if frame.empty or not np.isfinite(frame[list(required)].to_numpy(dtype=float)).all():
            raise ValueError(f"{path} contains no finite complete observations")
        time = frame["time_fs"].to_numpy(float)
        if np.any(np.diff(time) <= 0):
            raise ValueError(f"{path} has a non-increasing time axis")
        if reference_time is None:
            reference_time = time
        elif not np.array_equal(reference_time, time):
            raise ValueError(f"time axis mismatch: {path}")
        label = path.stem.removesuffix("_log")
        # Run-directory logs can share the generic name ``thermo_log.csv``.
        # In that case the run id is the only unambiguous plot label.
        if label in data:
            label = f"{path.parent.name}/{label}"
        if label in data:
            raise ValueError(f"duplicate plot label for {path}")
        data[label] = frame
    return data


def output_figure(fig: mpl.figure.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_trajectory(data: dict[str, pd.DataFrame], output: Path, *, raw_alpha: float) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 5.3), sharex=True, constrained_layout=True)
    specs = [
        ("potential_energy_eV", "Potential energy (eV)"),
        ("temperature_K", "Temperature (K)"),
    ]
    for ax, (column, ylabel) in zip(axes, specs, strict=True):
        for i, (label, frame) in enumerate(data.items()):
            colour = PALETTE[i % len(PALETTE)]
            x = frame["time_fs"].to_numpy(float)
            y = frame[column].to_numpy(float)
            window = max(3, min(21, len(y) // 10 * 2 + 1))
            smooth = pd.Series(y).rolling(window, center=True, min_periods=1).mean()
            ax.plot(x, y, color=colour, alpha=raw_alpha, lw=0.55, zorder=1)
            ax.plot(x, smooth, color=colour, lw=1.45, label=label, zorder=2)
        ax.set_ylabel(ylabel)
        ax.tick_params(top=True, right=True)
        ax.grid(False)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="best")
    axes[-1].set_xlabel("Time (fs)")
    output_figure(fig, output)


def plot_pairwise(data: dict[str, pd.DataFrame], output: Path) -> None:
    labels = list(data)
    if len(labels) < 2:
        raise ValueError("Pairwise comparison requires at least two trajectories")
    reference = data[labels[0]]
    x = reference["time_fs"].to_numpy(float)
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 5.3), sharex=True, constrained_layout=True)
    for i, label in enumerate(labels[1:]):
        target_energy = data[label]["potential_energy_eV"].to_numpy(float)
        reference_energy = reference["potential_energy_eV"].to_numpy(float)
        diff = np.abs(target_energy - reference_energy)
        smooth = pd.Series(diff).rolling(21, center=True, min_periods=1).mean()
        cumulative = np.cumsum(diff) / np.arange(1, len(diff) + 1)
        colour = PALETTE[i % len(PALETTE)]
        axes[0].plot(x, smooth, color=colour, lw=1.45, label=f"{label} vs {labels[0]}")
        axes[1].plot(x, cumulative, color=colour, lw=1.45, label=f"{label} vs {labels[0]}")
    axes[0].set_ylabel(r"Smoothed $|\Delta E|$ (eV)")
    axes[1].set_ylabel(r"Cumulative mean $|\Delta E|$ (eV)")
    axes[1].set_xlabel("Time (fs)")
    for ax in axes:
        ax.grid(False)
        ax.tick_params(top=True, right=True)
        ax.legend(frameon=False, fontsize=7.5, loc="best")
    output_figure(fig, output)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA / "group_1.txt")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FIGURES)
    args = parser.parse_args()
    configure_style()

    all_paths = sorted(args.raw_dir.glob("*_log.csv"))
    all_data = read_logs(all_paths)
    plot_trajectory(all_data, args.output_dir / "atlas_multi_file_comparison", raw_alpha=0.16)
    plot_trajectory(
        all_data,
        args.output_dir / "atlas_multi_file_comparison_alpha_adjusted",
        raw_alpha=0.08,
    )

    listed = []
    for line in args.metadata.read_text(encoding="utf-8").splitlines():
        if line.strip():
            path = ROOT / line.strip()
            if not path.is_file():
                raise FileNotFoundError(f"metadata entry does not exist: {path}")
            listed.append(path)
    group_data = read_logs(listed)
    plot_trajectory(group_data, args.output_dir / "atlas_group_1", raw_alpha=0.12)
    plot_pairwise(all_data, args.output_dir / "atlas_time_sequence_analysis")

    manifest = {
        "script": str(Path(__file__).relative_to(ROOT.parent.parent)),
        "inputs": {
            str(path.relative_to(ROOT.parent.parent)): sha256(path)
            for path in all_paths + listed
        },
        "outputs": [
            "atlas_multi_file_comparison.{png,svg,pdf}",
            "atlas_multi_file_comparison_alpha_adjusted.{png,svg,pdf}",
            "atlas_group_1.{png,svg,pdf}",
            "atlas_time_sequence_analysis.{png,svg,pdf}",
        ],
        "grammar": "Plot Atlas timecourse; self-contained implementation",
    }
    (args.output_dir / "atlas_plot_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
