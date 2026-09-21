# Projects

Each experiment is an independent project. The only shared code is under `src/common/`.

| Folder | Scope |
| --- | --- |
| [`house_prices_shap`](house_prices_shap) | Kaggle House Prices model comparison and SHAP interpretability |
| [`jarvis_volume`](jarvis_volume) | MACE/JARVIS equilibrium-volume benchmarking |
| [`data_cleaning_practice`](data_cleaning_practice) | Basic data cleaning and preprocessing exercise |


New experiments should be added as a new folder with the same structure:

```text
projects/<new-project>/
  README.md
  configs/
  data/raw/
  src/<package>/
  results/{raw,processed,figures,reports}
  tests/
```

Do not share raw data or results across projects unless the dependency is explicit and documented.
