import json

import pandas as pd
import pytest
from jarvis_volume.pipeline import evaluate, extract_jid, load_predictions, run


def test_extract_jid_normalizes_suffixes():
    assert extract_jid("JVASP-1002_mac") == "JVASP-1002"
    assert extract_jid("not-a-jid") is None


def test_load_predictions_drops_malformed_rows(tmp_path):
    path = tmp_path / "predictions.csv"
    path.write_text(
        "ID,Literature_output,Test_output,\n"
        "JVASP-1002,40.5,40.537,0.037\n"
        "JVASP-10036,127.715,127.583,0.132\n"
        ",,0.29\n",
        encoding="utf-8",
    )
    config = {
        "id_column": "ID",
        "prediction_column": "Literature_output",
        "test_output_column": "Test_output",
    }
    predictions = load_predictions(path, config)
    assert len(predictions) == 2
    assert predictions["jid"].tolist() == ["JVASP-1002", "JVASP-10036"]


def test_evaluate_metrics():
    merged = pd.DataFrame(
        {
            "dft_volume": [10.0, 20.0],
            "predicted_volume": [11.0, 19.0],
        }
    )
    metrics = evaluate(merged, "predicted_volume")
    assert metrics["n_materials"] == 2
    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(1.0)
    assert metrics["mape_pct"] == pytest.approx(7.5)


def test_run_creates_reproducible_artifacts(tmp_path):
    project_root = tmp_path / "project"
    raw_predictions = project_root / "data" / "raw" / "predictions.csv"
    reference = project_root / "data" / "raw" / "reference.csv"
    raw_predictions.parent.mkdir(parents=True)
    reference.parent.mkdir(parents=True, exist_ok=True)
    raw_predictions.write_text(
        "ID,Literature_output,Test_output,\n"
        "JVASP-1002,40.5,40.537,0.037\n"
        "JVASP-10036,127.715,127.583,0.132\n",
        encoding="utf-8",
    )
    reference.write_text(
        "jid,dft_volume\nJVASP-1002,40.0\nJVASP-10036,128.0\n",
        encoding="utf-8",
    )
    config_path = project_root / "configs" / "experiment.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "raw_predictions: data/raw/predictions.csv\n"
        "reference_volumes: data/raw/reference.csv\n"
        "prediction_column: Literature_output\n"
        "id_column: ID\n"
        "test_output_column: Test_output\n",
        encoding="utf-8",
    )
    results_root = project_root / "results"
    output = run(config_path, results_root)
    assert output["metrics"]["n_materials"] == 2
    assert output["paths"]["detail"].exists()
    assert output["paths"]["figure"].exists()
    assert output["paths"]["manifest"].exists()
    metrics_path = project_root / "results" / "processed" / "comparison_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    assert metrics["n_materials"] == 2
