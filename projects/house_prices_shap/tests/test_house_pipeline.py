import json

import pandas as pd
import pytest
from house_prices_shap.pipeline import load_and_validate, train_models
from sklearn.model_selection import train_test_split


def make_frame(rows: int = 40) -> pd.DataFrame:
    rng = pd.Series(range(rows), name="Id")
    return pd.DataFrame(
        {
            "Id": rng,
            "OverallQual": rng,
            "GrLivArea": rng * 5,
            "YearBuilt": 1900 + rng,
            "SalePrice": rng * 1000,
        }
    )


def test_load_and_validate_excludes_target_and_identifier(tmp_path):
    frame = make_frame()
    path = tmp_path / "train.csv"
    frame.to_csv(path, index=False)
    config = {"target": "SalePrice", "identifier": "Id"}
    X, y = load_and_validate(path, config)
    assert list(X.columns) == ["OverallQual", "GrLivArea", "YearBuilt"]
    assert len(X) == len(frame)


def test_load_and_validate_rejects_duplicate_ids(tmp_path):
    frame = make_frame(3)
    frame.loc[0, "Id"] = 1
    path = tmp_path / "train.csv"
    frame.to_csv(path, index=False)
    config = {"target": "SalePrice", "identifier": "Id"}
    with pytest.raises(ValueError, match="duplicate"):
        load_and_validate(path, config)


def test_train_models_and_save_artifacts(tmp_path):
    frame = make_frame(30)
    X = frame.drop(columns=["SalePrice"])
    y = frame["SalePrice"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
    config = {
        "split": {"random_state": 7},
        "models": {"mlp": {"hidden_layer_sizes": [2, 2], "max_iter": 100}},
    }
    results = train_models(X_train, X_test, y_train, y_test, config)
    assert [result["model"] for result in results] == [
        "XGBRegressor",
        "RandomForestRegressor",
        "MLPRegressor",
    ]
    assert all("mae" in result["metrics"] for result in results)

    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    predictions_path = raw_dir / "model_predictions.csv"
    metrics_path = processed_dir / "model_metrics.csv"
    predictions_path.parent.mkdir(parents=True)
    metrics_path.parent.mkdir(parents=True)

    from house_prices_shap.pipeline import save_model_artifacts

    save_model_artifacts(X_test, y_test, results, raw_dir, processed_dir)
    assert predictions_path.exists()
    assert metrics_path.exists()
    predictions = pd.read_csv(predictions_path)
    metrics = pd.read_csv(metrics_path)
    assert len(predictions) == len(X_test)
    assert len(metrics) == 3
    assert {"model", "mae", "mse", "rmse", "r2", "fit_seconds"} <= set(metrics.columns)


def test_run_manifest_is_written(tmp_path):
    from house_prices_shap.pipeline import run

    frame = make_frame(30)
    raw = tmp_path / "data" / "raw" / "train.csv"
    results = tmp_path / "results"
    raw.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(raw, index=False)
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "raw_data: data/raw/train.csv\n"
        "results_root: results\n"
        "target: SalePrice\n"
        "identifier: Id\n"
        "split:\n"
        "  test_size: 0.30\n"
        "  random_state: 7\n"
        "models:\n"
        "  mlp:\n"
        "    hidden_layer_sizes: [2, 2]\n"
        "    max_iter: 100\n"
        "shap:\n"
        "  background_samples: 5\n",
        encoding="utf-8",
    )
    output = run(raw, config_path, results, include_shap=False)
    manifest_path = output["paths"]["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset"]["rows"] == len(frame)
    assert manifest["dataset"]["test_rows"] == 9
    assert (results / "processed" / "model_metrics.csv").exists()
