# MACE-OFF torsion and SPICE evaluation

This project evaluates MACE-OFF23 models on SPICE data and independent
constrained torsion scans. Input datasets, model checkpoints, and per-grid
optimization outputs are kept out of Git.

## Repository layout

```text
.
├── data/                      # Local datasets and scans (not tracked)
├── models/mace_off23/         # Local model checkpoints (not tracked)
├── results/
│   ├── figures/               # Committed preview and generated figures
│   ├── metrics/               # Generated metric CSV files (not tracked)
│   └── torsion_relax/         # Per-molecule results (not tracked)
├── scripts/
│   ├── check_data.py
│   ├── create_dataset.py      # Convert TorsionDrive JSON to SPICE-style HDF5
│   ├── evaluate_models.py     # SPICE force evaluation
│   ├── run_torsiondrive.py    # Generate DFT torsion reference scans
│   ├── relax_torsion_grid.py  # Independent constrained MACE relaxations
│   ├── analyze_torsion_results.py
│   └── training/              # MACE training command templates
```

## Requirements

Python 3.10 or newer is recommended. Runtime dependencies include NumPy, ASE,
Matplotlib, SciPy, h5py, QCEngine, QCElemental, geomeTRIC, and MACE. PSI4 is
needed for DFT reference scans and is typically installed separately through
Conda. Use versions compatible with your PyTorch and accelerator setup.

```bash
pip install numpy ase matplotlib scipy h5py qcengine qcelemental geometric mace-torch
```

## Input files

Provide the SPICE extxyz files locally:

- `data/test_large_neut_all.xyz` is the default evaluation and force-check input.
- `data/train_large_neut_no_bad_clean.xyz` is the training set referenced by this project; it is not read by the evaluation script.

The files and the `TNet500_minimal/` and `biaryl_minimal/` folders are
intentionally excluded from Git because of size and data-use terms. Obtain the
SPICE datasets from their appropriate source. The minimals should contain the
QCSchema torsion-drive JSON files consumed by `relax_torsion_grid.py`.

Place the MACE-OFF23 checkpoints at:

```text
models/mace_off23/MACE-OFF23_small.model
models/mace_off23/MACE-OFF23_medium.model
models/mace_off23/MACE-OFF23_large.model
```

The model files are also excluded from Git. The evaluation script currently targets these three MACE-OFF23 variants.

## Usage

Scan one or more extxyz files for reference-force issues:

```bash
python scripts/check_data.py data/test_large_neut_all.xyz
```

The default report path is `data/force_quality_issues.csv`. Frames are flagged if any atomic force exceeds 15 eV/Å or the whole-configuration net force exceeds 0.1 eV/Å. Molecular-fragment net force is reported as a diagnostic and does not itself trigger exclusion.

Evaluate the three models and write a metrics CSV and plot:

```bash
python scripts/evaluate_models.py --device cpu
```

By default, this reads `data/test_large_neut_all.xyz` and writes
`results/metrics/fig1_metrics.csv` and `results/figures/fig1.png`. Use
`--limit N` for a short inference check, `--device cuda` for a CUDA-enabled
setup, or `--xyz`, `--csv`, and `--figure` to override paths.

## Training templates

The shell files under `scripts/training/` contain MACE training command templates. Replace the `/path/to/...` dataset and statistics placeholders with paths valid in your environment before running them. The templates share a common configuration and currently use the same experiment name; set a distinct `--name` for separate runs.

## Reference torsion scans

Run `scripts/run_torsiondrive.py` with a literature-optimized XYZ geometry and four zero-based atom indices defining the target dihedral. It uses QCEngine/TorsionDrive, geomeTRIC TRIC, and PSI4 wB97M-D3BJ/def2-TZVPPD, then saves the QCSchema result as JSON. See the script's `--help` for options.

Convert a folder of successful TorsionDrive JSON results to a SPICE-style HDF5
file with:

```bash
python scripts/create_dataset.py data/biaryl_minimal data/Biaryl-SPICE.hdf5
```

## Independent constrained torsion-grid relaxations

`scripts/relax_torsion_grid.py` reads the QCSchema results in
`data/TNet500_minimal/` and `data/biaryl_minimal/`. For every model and grid
point it initializes a new structure from that point's converged DFT geometry,
fixes the scanned dihedral at its grid angle, and independently optimizes all
other coordinates. It records the final energy, coordinates, force, and
convergence state in JSON, and also writes optimized structures as extxyz.
The default force threshold is 0.01 eV/Å.

Run all three MACE-OFF23 variants for one dataset:

```bash
python scripts/relax_torsion_grid.py tnet500 --device cpu
python scripts/relax_torsion_grid.py biaryl --device cpu
```

Run one model or a small input subset with `--model mace-off23-m --limit 1`.
Results go to `results/torsion_relax/` by default. Existing per-molecule
outputs are protected; pass `--overwrite` to replace them. The MACE checkpoints
use the paths in `models/mace_off23/` above.

## Torsion profiles and geometry RMSD

After all molecules and grid points have relaxation outputs, create the relative
torsion-profile and TNet500 aligned-RMSD plots with:

```bash
python scripts/analyze_torsion_results.py
```

This writes `results/figures/fig2_torsion.png` and
`results/metrics/fig2_metrics.csv`. The
script checks that every input molecule has 24 points for all three models;
`--allow-partial` is available for interim plots. It aligns TNet500 structures
with the matching DFT reference geometries before calculating mean RMSD.
Barrier-height errors and cumulative barrier-error distributions are not
calculated.

The currently included `results/figures/fig2_torsion_partial_n10.png` is a
partial preview based on 10 molecules per dataset. Its legend shows the sample
size; it is not a full-dataset benchmark figure.

## Notes

- Input energies are assumed to be in eV and forces in eV/Å, as represented by the extxyz data.
- Evaluation subsets are selected using the `config_type` metadata in each frame.
- `results/figures/fig1.png` is a checked-in example figure. Re-running
  evaluation overwrites the default figure and generates a metrics CSV that is
  ignored by Git.
