# MACE/JARVIS equilibrium-volume benchmark

This project compares MACE-predicted equilibrium volumes with JARVIS-DFT `dft_3d` reference volumes.

## Pipeline

```bash
uv run python -m jarvis_volume.pipeline \
  --config projects/jarvis_volume/configs/experiment.yaml
```

The pipeline normalizes JARVIS IDs, drops malformed rows, joins predictions to the JARVIS reference one-to-one on JID, computes MAE, RMSE, and MAPE, and writes raw comparison detail, processed metrics, manifests, and a parity figure.

## Reference generation

```bash
uv run python -m jarvis_volume.fetch_reference \
  --raw-predictions projects/jarvis_volume/data/raw/equilibrium_volume_results.csv \
  --reference-csv projects/jarvis_volume/data/raw/jarvis_dft_volumes.csv
```

The current reference cache has 104 rows and its hash is recorded in `data/raw/jarvis_dft_volumes.json`.

## Outputs

- `results/raw/comparison_detail.csv`
- `results/processed/comparison_metrics.json`
- `results/processed/run_manifest.json`
- `results/figures/volume_parity_plot.png`
- `results/reports/RESULTS.md`

## Caveat

The original repository did not include the script or model configuration that generated `equilibrium_volume_results.csv`. The meanings of `Literature_output` and `Test_output` remain ambiguous. Downstream evaluation is reproducible, but upstream prediction provenance must be supplied by the repository owner.
