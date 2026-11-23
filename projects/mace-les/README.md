# MACE-LES charged-defect benchmark

This project contains the MACE-Q and MACELES-Q leave-one-charge-out benchmark for the 59-atom Sb2Se3 V_Sb1 PBE+D3 dataset. It is a separate project parallel to polar_benchmark: MACE-POLAR remains in its existing folder; LES and the matched MACE-Q comparator live here.

## Current status

The initial 20-fit matrix (MACE-Q and MACELES-Q; five held-out charge states; two seeds) and all 20 four-path evaluations completed successfully on Rockfish. Results are in results/processed/CHARGE_TRANSFER_RESULTS.md with per-charge and per-trajectory CSVs. All model, metric, and prediction hashes matched the Rockfish outputs at archival time.

The repository contains protocols, scripts, manifests, and compact processed results. Full model weights and Slurm/training logs remain in the Rockfish archive at /scratch16/pclancy3/ycao73/charged_defect_lr_20261120_r01/runs/; model files are not committed. The frozen fold data are present on Rockfish under data/raw/ but ignored by Git because they contain raw structures and labels. Their hashes and split metadata are tracked in manifests/charge_transfer_manifest.json.

## Interpretation

This is same-defect, same-cell, fixed-geometry joint charge/multiplicity transfer, not geometric OOD, formation-energy accuracy, or proof that LES learned physical long-range electrostatics. Spin is fixed by charge in this dataset and cannot be identified independently. q=+1 extrapolation is especially unstable across seeds; see the report and CSVs.

## Reproduction on Rockfish

The archived run used the MACE commit and runtime recorded in manifests/source_lock.json. For future runs from this checkout:

1. Create a project-local environment at .venv with Python 3.11, PyTorch/CUDA, the pinned MACE version, and the les_fit extension at the commit recorded in source_lock.json. Do not use the old campaign directory's environment.
2. Ensure the frozen data/raw/charge_transfer_loo/ and data/raw/charge_transfer_loo_4path/ folds are available. Use the tracked manifest to verify each input hash.
3. Create results/raw/slurm/, then submit the 20-fit array: sbatch scripts/train_charge_transfer.slurm.
4. Submit evaluation only after all fit tasks succeed: sbatch --dependency=afterok:<fit-array-job-id> scripts/evaluate_charge_transfer.slurm.
5. Regenerate compact tables: python scripts/summarize_charge_transfer.py --runs-root results/raw --output-dir results/processed.

The Slurm scripts default to this project directory and write run products under results/raw/. Adjust account/partition directives only when needed for the allocation. Never fit offsets on held-out charge tests.

## Layout

- configs/CHARGE_TRANSFER_PROTOCOL.md: frozen experiment and metrics protocol.
- data/raw/: ignored Rockfish fold files; manifests are tracked separately.
- manifests/: source lock and split manifest; model and result hashes are in source_lock.json and the processed metrics table.
- scripts/: training/evaluation Slurm entry points and result processing code.
- results/processed/: tracked report and compact CSV outputs.
- results/raw/: ignored checkpoints, logs, and model files produced by future runs.
