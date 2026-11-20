# Model registry

`MODEL_REGISTRY.csv` lists every committed `*.model` file with its size and
sha256. It is regenerated (hashes only) by

```bash
uv run python projects/polar_benchmark/script/build_manifest.py
```

The columns `role, source_url, version, license, training_config, training_log, notes`
are filled in **by hand** and are preserved when the script is re-run. Every
`TODO` must be resolved before any result from that model goes into a paper.

| Path | What we believe it is | Status |
| --- | --- | --- |
| `model/mace-mh-1.model` | MACE-MH-1 multi-head foundation model (heads: `matpes_r2scan`, `mp_pbe_refit_add`, `spice_wB97M`, `oc20_usemppbe`, `omol`, `omat_pbe`); neutral baseline | Source URL / version / license: **to confirm** |
| `data/2_global_embedding/Charge_embedded_MACE/MACE_vse1_five_wb_swa_e1000.model` | MACE with a categorical `total_charge` embedding (5 classes), trained with `run.sh` on V_Se1 HSE06 data (`train.xyz`/`test.xyz`) | Inferred from `run.sh`; training log and exact mace version **to confirm** |
| `data/3_MF/MF/MF_MACE/VSe1_neu/Training/MACE_vse1_neu.model` | Multi-head (PBE/HSE) multi-fidelity MACE for neutral V_Se1, trained with `run_mh.sh` | Inferred; **to confirm** |
| `data/3_MF/MF/MF_MACE/VSe1_pl1/MACE_vse1_pl1.model` | Same as above for V_Se1 q=+1 | Inferred; **to confirm** |
| MACE-Polar checkpoint(s) | Not committed yet | **Missing**: add the checkpoint key/URL (e.g. `polar-1-s`) and sha256 |
| Fine-tuned MACE-Polar models (5 runs) | Not committed yet | **Missing**: add configs, logs, seeds and sha256 |
