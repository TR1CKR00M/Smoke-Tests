# Architecture

The repository separates unrelated experiments so that one project cannot accidentally mutate another project's inputs or outputs.

```text
README.md
pyproject.toml
Makefile
docs/
  ARCHITECTURE.md
  REPRODUCING.md
  PROVENANCE.md
  EVALUATION.md
  GOVERNANCE.md
projects/
  house_prices_shap/
    README.md
    configs/
    data/raw/
    notebooks/original_SHAP_Analysis_Practice.ipynb
    src/house_prices_shap/
    results/{raw,processed,figures,reports}
    tests/
  jarvis_volume/
    README.md
    configs/
    data/raw/
    legacy/
    src/jarvis_volume/
    results/{raw,processed,figures,reports}
    tests/
src/common/
```

## House Prices + SHAP project

The pipeline is:

1. read `projects/house_prices_shap/data/raw/train.csv`
2. validate target/identifier and select numeric columns
3. split with `test_size=0.30` and `random_state=7`
4. train XGBoost, RandomForest, and a seeded MLP
5. save raw test predictions
6. save processed metrics, manifests, and SHAP importances
7. generate figures and a paper-style results report

The original notebook is preserved under `notebooks/` as a historical artifact, not as the production entry point.

## MACE/JARVIS project

The pipeline is:

1. read raw MACE-predicted equilibrium volumes
2. drop malformed rows and normalize JARVIS IDs
3. read the cached JARVIS-DFT `dft_3d` reference volumes
4. join predictions and references one-to-one on JID
5. compute MAE, RMSE, MAPE, and per-material errors
6. save raw comparison detail, processed metrics, manifest, parity plot, and results report

The original `compare_with_jarvis.py` remains under `projects/jarvis_volume/legacy/` as a historical artifact, not as the production entry point.

## Extension pattern

Add a new project with the same layering: `configs/`, `data/raw/`, `src/<package>/`, `results/`, and `tests/`. Keep its data and outputs isolated unless a dependency is explicit.
