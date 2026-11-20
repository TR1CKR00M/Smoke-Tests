"""Explain the four SOAP error surrogates with TreeSHAP.

Loads the SOAP matrix and fitted GradientBoostingRegressor objects produced by
2_a_Surrogate_Model_Training.py, computes per-structure TreeSHAP values,
aggregates global importance into six radial/angular categories, and reports
the angular-momentum distribution among each model's Top-20 features.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

DEFAULT_MODELS = ["mace_k32_L0", "mace_k32_L2", "mace_k128_L0", "mace_k128_L2"]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "results" / "surrogate"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "angular_momentum"
FEATURE_PATTERN = re.compile(r"^(.+)-(.+)_n(\d)(\d)_l(\d+)$")
RANGE_NAMES = ["Short Range", "Medium Range", "Long Range"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", default=str(DEFAULT_INPUT_DIR), help="Artifacts created by script 2_a"
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for SHAP tables and figures",
    )
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS, help="Surrogate model name prefixes"
    )
    parser.add_argument(
        "--range-aggregation",
        choices=["sum", "mean"],
        default="sum",
        help="Aggregate mean absolute SHAP across features in each category",
    )
    parser.add_argument(
        "--short-max-mean-n",
        type=float,
        default=2.0,
        help="Short range if (n1+n2)/2 is at or below this threshold",
    )
    parser.add_argument(
        "--medium-max-mean-n",
        type=float,
        default=3.0,
        help=(
            "Medium range if mean radial index is above short threshold "
            "and at or below this threshold"
        ),
    )
    return parser.parse_args()


def parse_feature(name: str) -> tuple[int, int, int]:
    match = FEATURE_PATTERN.match(name)
    if not match:
        raise ValueError(f"Unrecognized SOAP feature name: {name!r}")
    n1, n2, angular_momentum = (int(match.group(i)) for i in (3, 4, 5))
    return n1, n2, angular_momentum


def radial_range(n1: int, n2: int, short_max: float, medium_max: float) -> str:
    # Radial basis indices are not literal distances in angstroms. This bins
    # their mean index into a reproducible low/middle/high radial-order proxy.
    mean_index = (n1 + n2) / 2.0
    if mean_index <= short_max:
        return RANGE_NAMES[0]
    if mean_index <= medium_max:
        return RANGE_NAMES[1]
    return RANGE_NAMES[2]


def extract_shap_matrix(explainer, features: np.ndarray) -> np.ndarray:
    values = explainer.shap_values(features)
    if hasattr(values, "values"):
        values = values.values
    if isinstance(values, list):
        if len(values) != 1:
            raise ValueError(f"Expected a single-output regressor, got {len(values)} SHAP outputs")
        values = values[0]
    matrix = np.asarray(values)
    if matrix.shape != features.shape:
        raise ValueError(f"Unexpected SHAP matrix shape {matrix.shape}; expected {features.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("SHAP matrix contains NaN or infinite values")
    return matrix


def save_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.short_max_mean_n >= args.medium_max_mean_n:
        raise ValueError("--short-max-mean-n must be less than --medium-max-mean-n")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_table = pd.read_csv(input_dir / "soap_features.csv")
    if "frame_index" not in feature_table:
        raise KeyError("soap_features.csv must contain a frame_index column")
    features = feature_table.drop(columns=["frame_index"]).to_numpy(dtype=np.float64)
    feature_names = feature_table.drop(columns=["frame_index"]).columns.tolist()
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError(f"Invalid SOAP feature matrix shape: {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("SOAP feature matrix contains NaN or infinite values")

    metadata = [parse_feature(name) for name in feature_names]
    categories = []
    category_names = [
        "Short Range Spherical",
        "Short Range Angular",
        "Medium Range Spherical",
        "Medium Range Angular",
        "Long Range Spherical",
        "Long Range Angular",
    ]
    for n1, n2, angular_momentum in metadata:
        range_name = radial_range(n1, n2, args.short_max_mean_n, args.medium_max_mean_n)
        angular_name = "Spherical" if angular_momentum == 0 else "Angular"
        categories.append(f"{range_name} {angular_name}")

    importance_by_model: dict[str, np.ndarray] = {}
    for model_name in args.models:
        model_path = input_dir / f"{model_name}_soap_surrogate.joblib"
        if not model_path.is_file():
            raise FileNotFoundError(f"Surrogate model not found: {model_path}")
        surrogate = joblib.load(model_path)
        explainer = shap.TreeExplainer(surrogate)
        shap_matrix = extract_shap_matrix(explainer, features)
        global_importance = np.mean(np.abs(shap_matrix), axis=0)
        importance_by_model[model_name] = global_importance

        shap_path = output_dir / f"{model_name}_shap_values.csv"
        save_csv(
            shap_path,
            ["frame_index", *feature_names],
            [
                [int(frame_idx), *row.tolist()]
                for frame_idx, row in zip(feature_table["frame_index"], shap_matrix, strict=True)
            ],
        )

    save_csv(
        output_dir / "global_feature_importance.csv",
        ["feature_name", *args.models],
        [
            [name, *[importance_by_model[model][i] for model in args.models]]
            for i, name in enumerate(feature_names)
        ],
    )

    category_scores: dict[str, dict[str, float]] = {}
    for model_name, importance in importance_by_model.items():
        category_scores[model_name] = {}
        for category_name in category_names:
            values = importance[np.asarray(categories) == category_name]
            category_scores[model_name][category_name] = (
                float(np.sum(values))
                if args.range_aggregation == "sum"
                else float(np.mean(values))
                if len(values)
                else 0.0
            )

    save_csv(
        output_dir / "six_category_importance.csv",
        ["category", *args.models],
        [
            [category, *[category_scores[model][category] for model in args.models]]
            for category in category_names
        ],
    )

    # Figure 4: six category panels, each with one bar per MACE model.
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    for ax, category_name in zip(axes.flat, category_names, strict=True):
        values = [category_scores[model][category_name] for model in args.models]
        ax.bar(range(len(args.models)), values, color=colors[: len(args.models)])
        ax.set_title(category_name)
        ax.set_xticks(range(len(args.models)))
        ax.set_xticklabels(
            [name.replace("mace_", "") for name in args.models], rotation=35, ha="right"
        )
        ax.set_ylabel(f"{args.range_aggregation.title()} mean |SHAP|")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Global SOAP feature importance by radial and angular category")
    fig.savefig(output_dir / "Figure_4_angular_radial_categories.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Top-20 feature ranks and angular-momentum channel frequencies.
    top_rows = []
    angular_counts: dict[str, dict[int, int]] = {}
    for model_name, importance in importance_by_model.items():
        top_indices = np.argsort(importance)[::-1][: min(20, len(feature_names))]
        counts = {ell: 0 for ell in range(5)}
        for rank, feature_index in enumerate(top_indices, start=1):
            n1, n2, ell = metadata[feature_index]
            counts[ell] += 1
            top_rows.append(
                [
                    model_name,
                    rank,
                    feature_names[feature_index],
                    ell,
                    n1,
                    n2,
                    float(importance[feature_index]),
                ]
            )
        angular_counts[model_name] = counts

    save_csv(
        output_dir / "top20_features.csv",
        ["model", "rank", "feature_name", "l", "n1", "n2", "mean_abs_shap"],
        top_rows,
    )
    save_csv(
        output_dir / "top20_angular_momentum_distribution.csv",
        [
            "model",
            *[f"l={ell}_count" for ell in range(5)],
            *[f"l={ell}_percent" for ell in range(5)],
        ],
        [
            [
                model,
                *[angular_counts[model][ell] for ell in range(5)],
                *[
                    100.0 * angular_counts[model][ell] / sum(angular_counts[model].values())
                    for ell in range(5)
                ],
            ]
            for model in args.models
        ],
    )

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    x = np.arange(len(args.models))
    width = 0.15
    for ell in range(5):
        percentages = [
            100.0 * angular_counts[model][ell] / sum(angular_counts[model].values())
            for model in args.models
        ]
        ax.bar(x + (ell - 2) * width, percentages, width, label=f"l={ell}")
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace("mace_", "") for name in args.models], rotation=20, ha="right")
    ax.set_ylabel("Share of Top-20 features (%)")
    ax.set_title("Angular-momentum composition of each model's Top-20 SOAP features")
    ax.legend(title="SOAP angular channel", ncol=5)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(
        output_dir / "Top20_angular_momentum_distribution.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    print(
        f"Computed SHAP matrices for {features.shape[0]} structures "
        f"and {features.shape[1]} SOAP features."
    )
    print(
        f"Radial grouping: mean(n1,n2) <= {args.short_max_mean_n:g} = short; "
        f"<= {args.medium_max_mean_n:g} = medium; above = long."
    )
    print(f"Category aggregation: {args.range_aggregation} of global mean absolute SHAP.")
    print("\nTop-20 angular-momentum frequencies:")
    print(f"{'Model':<16} | " + " | ".join(f"l={ell}" for ell in range(5)))
    print("-" * 58)
    for model in args.models:
        total = sum(angular_counts[model].values())
        print(
            f"{model:<16} | "
            + " | ".join(f"{100.0 * angular_counts[model][ell] / total:5.1f}%" for ell in range(5))
        )
    print(f"\nSaved SHAP matrices, importance tables, and figures under: {output_dir}")


if __name__ == "__main__":
    main()
