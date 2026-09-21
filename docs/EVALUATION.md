# Evaluation of the original repository

## What the original repo contained

The initial repository mixed two unrelated experiments in one flat folder:

- `SHAP_Analysis_Practice.ipynb`: house-price model comparison and SHAP explanations
- `compare_with_jarvis.py`: MACE-predicted equilibrium volume comparison against JARVIS-DFT
- `test.csv`: Kaggle house-price test data, unrelated to the JARVIS workflow
- `equilibrium_volume_results.csv`: JARVIS benchmark input

## Strengths

- The notebook demonstrated several model families.
- The JARVIS comparison produced useful downstream metrics.
- The raw JARVIS prediction table had enough information to run evaluation.

## Issues fixed by this branch

1. **Mixed projects** — split into `projects/house_prices_shap/` and `projects/jarvis_volume/`.
2. **No reproducible entry point** — added Make targets and CLI pipelines.
3. **Missing seeds** — MLP is now seeded; all models use fixed seeds.
4. **No dataset provenance** — added raw-data manifests and SHA-256 hashes.
5. **No result layer** — added `results/raw`, `results/processed`, `results/figures`, and `results/reports`.
6. **No validation** — added data checks and automated tests.
7. **No paper-style discussion** — added generated reports.
8. **No CI** — added GitHub Actions.

## Reproduction findings

### House Prices

The original notebook labeled XGBoost's printed metric as “Mean Squared Error,” but the printed value is MAE. This branch computes both MAE and MSE explicitly. Current fixed-seed results are:

| Model | MAE (USD) | RMSE (USD) | R² |
| --- | ---: | ---: | ---: |
| XGBRegressor | 20495.53 | 40170.84 | 0.7781 |
| RandomForestRegressor | 18263.05 | 31265.58 | 0.8656 |
| MLPRegressor | 22725.55 | 43820.97 | 0.7359 |

The restructured pipeline uses the mirror dataset and a seeded MLP, so exact notebook values are not expected.

### JARVIS

The downstream comparison reproduced:

- 104 matched materials
- MAE: 2.8487 Å³
- RMSE: 9.7171 Å³
- MAPE: 1.83%

The processed pipeline preserves these metrics and writes a reproducible parity figure.

## Remaining blockers

These require the repository owner's input:

1. `train.csv` was absent upstream. This branch includes a mirror file and records its hash, but the canonical Kaggle source and license still need confirmation.
2. The original MLP had no `random_state`, so the exact original MLP result cannot be reproduced uniquely.
3. The JARVIS prediction CSV has no upstream generation script or model config, so the source of `Literature_output` and `Test_output` remains unclear.
4. The original notebook mislabeled XGBoost's printed MAE as “Mean Squared Error.”
