# Provenance policy

Every experiment records a chain from input to final artifact:

1. **Raw data**: source URL or database, retrieval note, row count, columns, and SHA-256.
2. **Config**: experiment configuration and fixed random seed.
3. **Code**: Git commit and package versions.
4. **Raw results**: direct model output or joined comparison table.
5. **Processed results**: derived metrics and summaries.
6. **Figures**: generated from processed metrics.
7. **Reports**: paper-style interpretation referencing generated figures.

Raw data and generated reference caches live under each project's `data/raw/`. Run-level provenance lives in each project's `results/processed/run_manifest.json`.

Do not edit processed results directly. Regenerate them with the documented commands.

## Known provenance blockers

1. `projects/house_prices_shap/data/raw/train.csv` is a public mirror; canonical Kaggle source and license need owner confirmation.
2. `projects/house_prices_shap/data/raw/test.csv` is committed for context but is not used by the current training pipeline.
3. `projects/jarvis_volume/data/raw/equilibrium_volume_results.csv` lacks upstream generation metadata.
4. The original notebook's MLP was unseeded, so its exact historical result is not uniquely reproducible.
