"""Plot the relaxation benchmark purely from ``results/processed/relax_per_charge.csv``.

Usage (from the repository root):

    uv run python projects/polar_benchmark/script/plot_relax_comparison.py

Panel (a): RMSD between the MLIP ground state and the DFT ground state per
charge state (filled marker = same distortion selected, open = miss).
Panel (b): Spearman rank correlation of the distortion energies (MLIP vs DFT).
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "results/processed/relax_per_charge.csv"
DEFAULT_FIGURE = PROJECT_ROOT / "results/figures/relax_ground_state_vs_charge.png"
RMSD_SUCCESS_THRESHOLD = 0.1
MARKERS = ("D", "o", "s", "^", "v", "P", "X")


def _float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_from_csv(csv_path: Path, output: Path, show: bool = False) -> Path:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_rows(csv_path)
    if not rows:
        raise SystemExit(f"No rows in {csv_path}")
    models = list(dict.fromkeys(row["model"] for row in rows))
    charges = sorted({int(row["charge"]) for row in rows})

    fig, (ax_rmsd, ax_rho) = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    ax_rmsd.axhspan(0, RMSD_SUCCESS_THRESHOLD, color="green", alpha=0.12)
    ax_rmsd.axhline(RMSD_SUCCESS_THRESHOLD, color="green", linestyle="--", alpha=0.6)
    offsets = {m: (i - (len(models) - 1) / 2) * 0.08 for i, m in enumerate(models)}
    for index, model in enumerate(models):
        subset = sorted((r for r in rows if r["model"] == model), key=lambda r: int(r["charge"]))
        marker = MARKERS[index % len(MARKERS)]
        color = f"C{index}"
        xs = [int(r["charge"]) + offsets[model] for r in subset]
        rmsd = [_float(r["rmsd_gs_all_A"]) for r in subset]
        hits = [r["ground_state_hit"] == "True" for r in subset]
        ax_rmsd.plot(xs, rmsd, color=color, alpha=0.4, linewidth=1)
        for x, y, hit in zip(xs, rmsd, hits, strict=True):
            ax_rmsd.scatter(
                x,
                y,
                marker=marker,
                s=55,
                zorder=3,
                edgecolors=color,
                facecolors=color if hit else "none",
            )
        ax_rmsd.scatter([], [], marker=marker, color=color, label=model)
        rho = [_float(r["spearman_rho"]) for r in subset]
        ax_rho.plot(xs, rho, marker=marker, color=color, label=model)

    ax_rmsd.scatter([], [], marker="o", facecolors="none", edgecolors="gray", label="GS miss")
    ax_rmsd.set_xlabel("Defect charge state $q$")
    ax_rmsd.set_ylabel("RMSD(MLIP GS, DFT GS) ($\\mathrm{\\AA}$)")
    ax_rmsd.set_xticks(charges)
    ax_rmsd.set_ylim(bottom=0)
    ax_rmsd.set_title("(a) Ground-state structure")
    ax_rmsd.legend(fontsize=8, frameon=True)
    ax_rho.axhline(0, color="gray", linewidth=0.8)
    ax_rho.set_xlabel("Defect charge state $q$")
    ax_rho.set_ylabel("Spearman $\\rho$ (distortion energies)")
    ax_rho.set_xticks(charges)
    ax_rho.set_ylim(-1.05, 1.05)
    ax_rho.set_title("(b) Energy ranking")
    fig.suptitle("$V_{\\mathrm{Sb}(1)}$ in $\\mathrm{Sb_2Se_3}$: MLIP vs. DFT (PBE+D3)")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved figure to {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if not args.csv.is_file():
        raise SystemExit(
            f"{args.csv} not found; run script/mace_relax_comparison.py first to create it."
        )
    plot_from_csv(args.csv, args.output, show=args.show)


if __name__ == "__main__":
    main()
