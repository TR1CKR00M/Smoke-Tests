# Reproducing everything

## 1. Environment

```bash
uv sync --extra dev
```

This creates `.venv/` and installs the pinned dependency ranges recorded in `uv.lock`.

## 2. Run a project

See the selected project's README for its reproduction command and expected artifacts.

## 3. Validate

```bash
make test
make lint
```

CI runs the fast tests and lint on every pull request.

## 4. Expected artifacts

Every project should document its generated raw results, processed metrics, figures, reports, and manifests in its own README. Regenerate generated files from code; do not edit results by hand.

## Reproduction results observed in this environment

### House Prices

| Model | MAE (USD) | RMSE (USD) | R² |
| --- | ---: | ---: | ---: |
| XGBRegressor | 20495.53 | 40170.84 | 0.7781 |
| RandomForestRegressor | 18263.05 | 31265.58 | 0.8656 |
| MLPRegressor | 22725.55 | 43820.97 | 0.7359 |

These are fixed-seed results from the restructured pipeline, not exact notebook outputs.

### MACE/JARVIS

- Matched materials: 104
- MAE: 2.8487 Å³
- RMSE: 9.7171 Å³
- MAPE: 1.83%

These match the original downstream comparison.
