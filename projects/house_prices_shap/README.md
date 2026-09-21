# House Prices + SHAP

This project trains three regressors on the Kaggle House Prices dataset and produces reproducible metrics and SHAP explanations.

## Pipeline

```bash
uv run python -m house_prices_shap.pipeline \
  --config projects/house_prices_shap/configs/experiment.yaml
```

The pipeline validates the target and identifier, selects numeric features, performs a 70/30 split with `random_state=7`, trains XGBoost, RandomForest, and a seeded MLP, and writes raw predictions, processed metrics, manifests, figures, and a results report.

## Outputs

- `results/raw/model_predictions.csv`
- `results/processed/model_metrics.csv`
- `results/processed/*_shap_importance.csv`
- `results/processed/run_manifest.json`
- `results/figures/*.png`
- `results/reports/RESULTS.md`

## Data provenance

The original repository did not include `train.csv`. This branch uses a public mirror of the Kaggle House Prices training set. The file has 1461 rows and its SHA-256 is recorded in `data/raw/train_manifest.json`. Canonical Kaggle source and license still need owner confirmation.

## Caveat

The original notebook did not seed the MLP. This project fixes the seed, so the MLP result is reproducible but will not match the unseeded original run exactly.
