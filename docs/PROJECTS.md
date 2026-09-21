# Project boundaries

The repository is a workspace, not a single experiment. Each project owns its own data, config, code, outputs, tests, and report.

| Project | Scope | Production entry point | Historical artifacts |
| --- | --- | --- | --- |
| `house_prices_shap` | Kaggle House Prices model comparison and SHAP interpretation | `projects/house_prices_shap/src/house_prices_shap/pipeline.py` | `projects/house_prices_shap/notebooks/original_SHAP_Analysis_Practice.ipynb` |
| `jarvis_volume` | MACE-predicted equilibrium volumes compared with JARVIS-DFT | `projects/jarvis_volume/src/jarvis_volume/pipeline.py` | `projects/jarvis_volume/legacy/` |

## Isolation rules

1. Do not share raw data between projects unless the dependency is explicit and documented.
2. Do not write one project's outputs into another project's folder.
3. Keep configs and seeds local to the project.
4. Add a new project with its own `configs/`, `data/raw/`, `src/`, `results/`, and `tests/`.
5. Keep common utilities in `src/common/` only when they are genuinely reusable.
