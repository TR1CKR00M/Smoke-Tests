"""Compare Top-20 SOAP feature overlap and importance across MACE models.

Consumes ``top20_features.csv`` produced by
``2_b_Angular_momentum_distribution.py`` and creates:
1. A symmetric pairwise overlap matrix for the four model Top-20 sets.
2. Four horizontal bar-chart panels showing each model's ranked Top-20 features.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_MODELS = ["mace_k32_L0", "mace_k32_L2", "mace_k128_L0", "mace_k128_L2"]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "results" / "angular_momentum"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "feature_hierarchy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Directory created by analysis script 2_b")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for plots and tables")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Model prefixes, in desired display order")
    parser.add_argument("--top-n", type=int, default=20, help="Number of ranked features per model")
    parser.add_argument("--dopant", default="Cr", help="Element symbol used to mark dopant-related features")
    return parser.parse_args()


def load_top_features(path: Path, models: list[str], top_n: int) -> dict[str, list[dict]]:
    if not path.is_file():
        raise FileNotFoundError(f"Top feature table not found: {path}. Run script 2_b first.")
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        columns = set(reader.fieldnames or [])
    required = {"model", "rank", "feature_name", "mean_abs_shap"}
    missing = required.difference(columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    selected = {model: [] for model in models}
    for row in rows:
        if row["model"] in selected:
            row["rank"] = int(row["rank"])
            row["mean_abs_shap"] = float(row["mean_abs_shap"])
            selected[row["model"]].append(row)
    absent_models = [model for model in models if not selected[model]]
    if absent_models:
        raise ValueError(f"No Top feature data found for models: {', '.join(absent_models)}")

    for model in models:
        selected[model].sort(key=lambda row: row["rank"])
        if len(selected[model]) < top_n:
            count = len(selected[model])
            raise ValueError(f"{model} has only {count} ranked features; cannot plot Top-{top_n}.")
        selected[model] = selected[model][:top_n]
    return selected


def main() -> None:
    args = parse_args()
    if args.top_n < 1:
        raise ValueError("--top-n must be a positive integer")
    if len(args.models) != 4:
        raise ValueError("This comparison figure expects exactly four models")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    top_features = load_top_features(input_dir / "top20_features.csv", args.models, args.top_n)

    # Pairwise set-intersection cardinality. Feature labels are unique SOAP
    # descriptor names, so identical labels indicate the same input feature.
    feature_sets = {
        model: {row["feature_name"] for row in top_features[model]}
        for model in args.models
    }
    overlap = np.zeros((len(args.models), len(args.models)), dtype=int)
    for i, model_i in enumerate(args.models):
        for j, model_j in enumerate(args.models):
            overlap[i, j] = len(feature_sets[model_i] & feature_sets[model_j])

    overlap_path = output_dir / "top20_pairwise_overlap.csv"
    with overlap_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["model", *args.models])
        for model, row in zip(args.models, overlap):
            writer.writerow([model, *row.tolist()])

    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    image = ax.imshow(overlap, cmap="Blues", vmin=0, vmax=args.top_n)
    ax.set_xticks(range(len(args.models)), labels=args.models, rotation=35, ha="right")
    ax.set_yticks(range(len(args.models)), labels=args.models)
    for i in range(len(args.models)):
        for j in range(len(args.models)):
            ax.text(j, i, str(overlap[i, j]), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, label=f"Shared features in Top-{args.top_n}")
    ax.set_title(f"Pairwise Top-{args.top_n} SOAP Feature Overlap")
    ax.set_xlabel("Model")
    ax.set_ylabel("Model")
    fig.savefig(output_dir / "Top20_pairwise_overlap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Export the plotted values and feature-type labels as a convenient table.
    feature_rows = []
    for model in args.models:
        for row in top_features[model]:
            name = row["feature_name"]
            feature_class = (
                f"Dopant-related ({args.dopant})"
                if args.dopant in name.split("_n", 1)[0].split("-")
                else "Host-only"
            )
            row["feature_class"] = feature_class
            feature_rows.append(row)
    with (output_dir / "top20_feature_hierarchy.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["model", "rank", "feature_name", "l", "n1", "n2", "mean_abs_shap", "feature_class"])
        writer.writeheader()
        writer.writerows(feature_rows)

    class_colors = {f"Dopant-related ({args.dopant})": "#E45756", "Host-only": "#4C78A8"}
    fig, axes = plt.subplots(2, 2, figsize=(17, 13), constrained_layout=True)
    for ax, model in zip(axes.flat, args.models):
        subset = sorted(top_features[model], key=lambda row: row["rank"], reverse=True)
        bar_colors = [class_colors[row["feature_class"]] for row in subset]
        ax.barh([row["feature_name"] for row in subset], [row["mean_abs_shap"] for row in subset], color=bar_colors)
        ax.set_title(model)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_ylabel("SOAP feature (ranked; highest at top)")
        ax.grid(axis="x", alpha=0.25)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=label)
        for label, color in class_colors.items()
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(f"Top-{args.top_n} SOAP Feature Importance by Model", y=1.02)
    fig.savefig(output_dir / "Top20_feature_importance_by_model.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Top-{args.top_n} pairwise feature overlap:")
    print("            " + "  ".join(f"{model:>14}" for model in args.models))
    for model, row in zip(args.models, overlap):
        print(f"{model:<12}" + "  ".join(f"{value:>14d}" for value in row))
    print(f"\nSaved overlap matrix, feature ranking table, and figures under: {output_dir}")


if __name__ == "__main__":
    main()
