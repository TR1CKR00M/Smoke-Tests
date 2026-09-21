"""Reproducible training and SHAP pipeline for the house-price project."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.provenance import read_yaml, runtime_metadata, sha256_file, write_json
from common.validation import require_columns, require_unique


def load_and_validate(raw_csv: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    target = config["target"]
    identifier = config.get("identifier")
    frame = pd.read_csv(raw_csv)
    require_columns(frame, [target] + ([identifier] if identifier else []), raw_csv)
    if identifier:
        require_unique(frame, identifier, raw_csv)
    if frame[target].isna().any():
        raise ValueError(f"Target column {target!r} contains missing values")

    exclude = [target] + ([identifier] if identifier else [])
    frame = frame.dropna(axis=0, subset=[target]).reset_index(drop=True)
    feature_frame = frame.drop(columns=exclude).select_dtypes(exclude=["object"])
    if feature_frame.empty:
        raise ValueError("No numeric features were found")
    if feature_frame.isna().all().any():
        raise ValueError("One or more numeric features contain only missing values")
    return feature_frame, frame[target]


def train_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seed = int(config["split"]["random_state"])
    estimators: dict[str, Any] = {
        "xgboost": xgboost.XGBRegressor(random_state=seed),
        "random_forest": RandomForestRegressor(random_state=seed),
        "mlp": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=tuple(config["models"]["mlp"]["hidden_layer_sizes"]),
                max_iter=int(config["models"]["mlp"]["max_iter"]),
                random_state=seed,
            ),
        ),
    }
    for model_name, estimator in estimators.items():
        started = time.perf_counter()
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)
        mse = float(mean_squared_error(y_test, predictions))
        results.append(
            {
                "model": model_name_to_label(model_name),
                "estimator": estimator,
                "predictions": predictions,
                "metrics": {
                    "mae": float(mean_absolute_error(y_test, predictions)),
                    "mse": mse,
                    "rmse": float(np.sqrt(mse)),
                    "r2": float(r2_score(y_test, predictions)),
                    "fit_seconds": time.perf_counter() - started,
                },
            }
        )
    return results


def model_name_to_label(name: str) -> str:
    return {
        "xgboost": "XGBRegressor",
        "random_forest": "RandomForestRegressor",
        "mlp": "MLPRegressor",
    }[name]


def save_model_artifacts(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_results: list[dict[str, Any]],
    raw_dir: Path,
    processed_dir: Path,
) -> Path:
    predictions = pd.DataFrame(
        {
            "row_index": X_test.index,
            "y_true": y_test.to_numpy(),
        }
    )
    for result in model_results:
        predictions[result["model"]] = result["predictions"]
    prediction_path = raw_dir / "model_predictions.csv"
    predictions.to_csv(prediction_path, index=False)

    metric_rows: list[dict[str, Any]] = []
    for result in model_results:
        metric_rows.append({"model": result["model"], **result["metrics"]})
    metric_path = processed_dir / "model_metrics.csv"
    pd.DataFrame(metric_rows).to_csv(metric_path, index=False)
    return prediction_path


def compute_shap(
    model_results: list[dict[str, Any]],
    X: pd.DataFrame,
    X_train: pd.DataFrame,
    config: dict[str, Any],
    processed_dir: Path,
    figures_dir: Path,
) -> None:
    import shap

    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config["split"]["random_state"])
    background = shap.utils.sample(
        X_train,
        nsamples=int(config["shap"]["background_samples"]),
        random_state=seed,
    )
    for result in model_results:
        explainer = shap.Explainer(result["estimator"].predict, background)
        explanation = explainer(X)
        importance = pd.DataFrame(
            {
                "feature": X.columns,
                "mean_abs_shap": np.abs(explanation.values).mean(axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False)
        importance_path = processed_dir / f"{result['model']}_shap_importance.csv"
        importance.to_csv(importance_path, index=False)

        figure_path = figures_dir / f"{result['model']}_shap_bar.png"
        fig = plt.figure(figsize=(9, 7))
        shap.plots.bar(explanation, max_display=12, show=False)
        plt.tight_layout()
        plt.savefig(figure_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        beeswarm_path = figures_dir / f"{result['model']}_shap_beeswarm.png"
        fig = plt.figure(figsize=(9, 7))
        shap.plots.beeswarm(explanation, max_display=12, show=False)
        plt.tight_layout()
        plt.savefig(beeswarm_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_model_metrics(model_results: list[dict[str, Any]], figures_dir: Path) -> None:
    metric_frame = pd.DataFrame(
        [{"model": result["model"], "mae": result["metrics"]["mae"]} for result in model_results]
    )
    figure_path = figures_dir / "model_mae.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=metric_frame, x="mae", y="model", ax=ax)
    ax.set_xlabel("Mean absolute error (USD)")
    ax.set_ylabel("")
    ax.set_title("Test-set MAE by model")
    fig.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)


def write_report(
    model_results: list[dict[str, Any]],
    report_path: Path,
) -> None:
    """Write the reproducible, paper-style results narrative."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metric_rows = "\n".join(
        f"| {result['model']} | {result['metrics']['mae']:.2f} | "
        f"{result['metrics']['rmse']:.2f} | {result['metrics']['r2']:.4f} |"
        for result in model_results
    )
    best = min(model_results, key=lambda result: result["metrics"]["mae"])
    report_path.write_text(
        "# House Prices: model comparison and interpretability\n\n"
        "## Method\n\n"
        "The pipeline uses the numeric columns in the Kaggle House Prices training set, "
        "holds out 30% of the data with a fixed random seed, and compares XGBoost, "
        "RandomForest, and a seeded MLP. The mean absolute error is the primary metric; "
        "RMSE and R² are retained as secondary diagnostics.\n\n"
        "## Results\n\n"
        "| Model | MAE (USD) | RMSE (USD) | R² |\n"
        "| --- | ---: | ---: | ---: |\n"
        f"{metric_rows}\n\n"
        f"![Test-set MAE by model](../figures/model_mae.png)\n\n"
        "## Interpretation\n\n"
        f"{best['model']} gives the lowest test MAE in this fixed-seed run. The raw "
        "test predictions and processed metrics are generated by the pipeline, and each "
        "model has an accompanying SHAP bar and beeswarm figure when SHAP is enabled. "
        "Because only numeric columns are used, these values are not directly comparable "
        "with public leaderboards that include engineered categorical features.\n",
        encoding="utf-8",
    )


def run(
    raw_csv: Path,
    config_path: Path,
    results_root: Path,
    include_shap: bool = True,
) -> dict[str, Any]:
    config = read_yaml(config_path)
    results_root.mkdir(parents=True, exist_ok=True)
    raw_dir = results_root / "raw"
    processed_dir = results_root / "processed"
    figures_dir = results_root / "figures"
    for directory in (raw_dir, processed_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    X, y = load_and_validate(raw_csv, config)
    project_root = raw_csv.parent.parent.parent
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(config["split"]["test_size"]),
        random_state=int(config["split"]["random_state"]),
    )
    model_results = train_models(X_train, X_test, y_train, y_test, config)
    prediction_path = save_model_artifacts(X_test, y_test, model_results, raw_dir, processed_dir)

    manifest = {
        "dataset": {
            "path": str(raw_csv.relative_to(project_root)),
            "sha256": sha256_file(raw_csv),
            "rows": int(len(X)),
            "numeric_features": int(len(X.columns)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "target": config["target"],
        },
        "split": config["split"],
        "data_provenance": config.get("data_provenance", {}),
        "environment": {
            **runtime_metadata(),
            "packages": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": __import__("sklearn").__version__,
                "xgboost": xgboost.__version__,
                "shap": __import__("shap").__version__,
            },
        },
    }
    manifest_path = processed_dir / "run_manifest.json"
    write_json(manifest_path, manifest)

    if include_shap:
        compute_shap(model_results, X, X_train, config, processed_dir, figures_dir)

    plot_model_metrics(model_results, figures_dir)
    report_path = figures_dir.parent / "reports" / "RESULTS.md"
    write_report(model_results, report_path)

    return {
        "config": config,
        "model_results": model_results,
        "paths": {
            "predictions": prediction_path,
            "metrics": processed_dir / "model_metrics.csv",
            "manifest": manifest_path,
            "report": report_path,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skip-shap", action="store_true")
    args = parser.parse_args()
    config = read_yaml(args.config)
    project_root = args.config.parents[1]
    run(
        raw_csv=project_root / config["raw_data"],
        config_path=args.config,
        results_root=project_root / config["results_root"],
        include_shap=not args.skip_shap,
    )


if __name__ == "__main__":
    main()
