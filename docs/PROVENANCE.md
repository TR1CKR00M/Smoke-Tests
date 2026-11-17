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

## Project-specific blockers

Record unresolved source, licensing, generation, or reproducibility issues in the relevant project's README and results report.
