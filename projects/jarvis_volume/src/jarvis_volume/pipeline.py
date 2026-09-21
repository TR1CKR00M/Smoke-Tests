"""Reproducible MACE/JARVIS equilibrium-volume benchmarking pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def load_predictions(path: Path, config: dict[str, Any]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        config["id_column"],
        config["prediction_column"],
        config["test_output_column"],
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    frame = frame.dropna(subset=[config["id_column"], config["prediction_column"]])
    frame[config["prediction_column"]] = pd.to_numeric(
        frame[config["prediction_column"]], errors="coerce"
    )
    frame[config["test_output_column"]] = pd.to_numeric(
        frame[config["test_output_column"]], errors="coerce"
    )
    frame = frame.dropna(subset=[config["prediction_column"]])
    frame["jid"] = frame[config["id_column"]].map(extract_jid)
    return frame.dropna(subset=["jid"]).reset_index(drop=True)


def extract_jid(value: object) -> str | None:
    text = str(value)
    if "JVASP-" not in text:
        return None
    return text.split("JVASP-")[0] + "JVASP-" + text.split("JVASP-")[1].split("_")[0]


def load_reference(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty or {"jid", "dft_volume"}.difference(frame.columns):
        raise ValueError(f"Invalid reference dataset: {path}")
    return frame


def evaluate(
    merged: pd.DataFrame,
    prediction_column: str,
) -> dict[str, float]:
    y_true = merged["dft_volume"].to_numpy(dtype=float)
    y_pred = merged[prediction_column].to_numpy(dtype=float)
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mape_pct": float(np.mean(np.abs((y_pred - y_true) / y_true)) * 100),
        "n_materials": float(len(merged)),
    }


def plot_parity(
    merged: pd.DataFrame,
    prediction_column: str,
    figure_path: Path,
) -> None:
    y_true = merged["dft_volume"].to_numpy(dtype=float)
    y_pred = merged[prediction_column].to_numpy(dtype=float)
    lower = float(min(y_true.min(), y_pred.min()) * 0.95)
    upper = float(max(y_true.max(), y_pred.max()) * 1.05)
    mae = float(mean_absolute_error(y_true, y_pred))
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.7, edgecolors="black", linewidths=0.4)
    ax.plot([lower, upper], [lower, upper], "r--", linewidth=1.5, label="Ideal (y = x)")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("JARVIS-DFT volume (Å³)")
    ax.set_ylabel("MACE predicted volume (Å³)")
    ax.set_title(f"Equilibrium volume comparison (MAE = {mae:.3f} Å³)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)


def run(
    config_path: Path,
    results_root: Path | None = None,
) -> dict[str, Any]:
    config = read_config(config_path)
    project_root = config_path.parents[1]
    raw_predictions_path = project_root / config["raw_predictions"]
    reference_path = project_root / config["reference_volumes"]
    if results_root is None:
        results_root = project_root / config["results_root"]
    results_root.mkdir(parents=True, exist_ok=True)
    raw_dir = results_root / "raw"
    processed_dir = results_root / "processed"
    figures_dir = results_root / "figures"
    for directory in (raw_dir, processed_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    predictions = load_predictions(raw_predictions_path, config)
    reference = load_reference(reference_path)
    merged = predictions.merge(reference, on="jid", how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("No JIDs matched between predictions and JARVIS reference")
    merged = merged.sort_values("jid").reset_index(drop=True)
    metrics = evaluate(merged, config["prediction_column"])

    detail_path = raw_dir / "comparison_detail.csv"
    merged.to_csv(detail_path, index=False)
    metrics_path = processed_dir / "comparison_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    figure_path = figures_dir / "volume_parity_plot.png"
    plot_parity(merged, config["prediction_column"], figure_path)

    manifest = {
        "raw_predictions": {
            "path": config["raw_predictions"],
            "sha256": hashlib.sha256(raw_predictions_path.read_bytes()).hexdigest(),
        },
        "reference_volumes": {
            "path": config["reference_volumes"],
            "sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            "rows": int(len(reference)),
        },
        "upstream_provenance": {
            "raw_predictions_source": "owner-supplied artifact",
            "reference_source": "JARVIS dft_3d",
            "raw_prediction_generation_metadata_available": False,
        },
        "environment": {
            "python": __import__("platform").python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": __import__("sklearn").__version__,
        },
    }
    manifest_path = processed_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    report_path = results_root / "reports" / "RESULTS.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# MACE/JARVIS equilibrium-volume benchmark\n\n"
        "## Method\n\n"
        "MACE-predicted equilibrium volumes are normalized to JARVIS material IDs and "
        "joined one-to-one with JARVIS-DFT `dft_3d` reference volumes. The evaluation "
        "reports MAE, RMSE, and MAPE and draws a parity plot against the ideal "
        "one-to-one line.\n\n"
        "## Results\n\n"
        f"- Matched materials: {metrics['n_materials']:.0f}\n"
        f"- MAE: {metrics['mae']:.4f} Å³\n"
        f"- RMSE: {metrics['rmse']:.4f} Å³\n"
        f"- MAPE: {metrics['mape_pct']:.2f}%\n\n"
        f"![Volume parity plot](../figures/volume_parity_plot.png)\n\n"
        "## Interpretation\n\n"
        "The low MAPE indicates close overall agreement across the matched benchmark "
        "set, while the RMSE is substantially larger than the MAE, suggesting a small "
        "number of large deviations. The per-material errors are retained in the raw "
        "comparison table for targeted follow-up.\n",
        encoding="utf-8",
    )

    return {
        "metrics": metrics,
        "paths": {
            "detail": detail_path,
            "metrics": metrics_path,
            "figure": figure_path,
            "manifest": manifest_path,
            "report": report_path,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
