# MoS₂:Cr Defect Potential Model Comparison

This repository generates a Cr-substituted MoS₂ dataset, trains a four-model MACE grid, and analyzes prediction errors and learned representations. The model grid varies hidden-channel width (`k=32` or `128`) and the hidden equivariant channels (`L=0` or `L=2`).

## Repository layout

```text
.
├── data/
│   └── processed/                 # Generated train, validation, and test XYZ files
├── docs/
│   └── diagnosis_new.md           # Detailed model-diagnosis methodology
├── models/                        # Trained MACE model files
├── results/
│   ├── surrogate/                 # SOAP features, error targets, surrogate models
│   ├── angular_momentum/           # TreeSHAP matrices and angular/radial summaries
│   ├── feature_hierarchy/          # Top-feature overlap and importance figures
│   └── attention_flow/             # Feature matrices, singular spectra, effective ranks
├── scripts/
│   ├── generate_defect_data.py
│   ├── train_grid.sh
│   ├── analysis/                   # Ordered analysis scripts
│   └── legacy/                     # Original diagnosis script and IDE sample
├── .gitignore
├── README.md
└── requirements.txt
```

Analysis scripts resolve their default inputs and outputs relative to the repository root, so they can be launched from the root directory without manually changing paths.

## Environment

Use Python with the packages listed in [`requirements.txt`](requirements.txt). MACE installation and supported PyTorch/CUDA combinations depend on the target machine; install a compatible MACE environment before generating reference labels or training. CUDA is optional. The data generator and training script fall back to CPU when CUDA is unavailable.

Install the Python requirements in an activated environment:

```bash
pip install -r requirements.txt
```

## Workflow

### 1. Generate the defect datasets

```bash
python scripts/generate_defect_data.py
```

The script builds a 4×4×1 2H-MoS₂ supercell with 48 atoms, replaces two neighboring Mo sites with Cr, and applies random displacements. MACE-MP-0 Large supplies each frame's reference total energy and forces. The script writes these datasets to `data/processed/`:

| Dataset | Frames | Displacement standard deviation | Role |
| --- | ---: | ---: | --- |
| `defect_train_300K.xyz` | 150 | 0.04 Å | Model training |
| `defect_val_300K.xyz` | 30 | 0.04 Å | Model validation |
| `defect_test_300K.xyz` | 50 | 0.04 Å | Same-distribution test set |
| `defect_test_1200K.xyz` | 50 | 0.12 Å | Larger-displacement test set |

The temperature labels refer to displacement regimes in static structures. The generator does not perform molecular dynamics or thermal equilibration. The two reference fields are `REF_energy` and `REF_forces`.

### 2. Train the MACE model grid

```bash
bash scripts/train_grid.sh
```

The training script uses the same train, validation, and 1200 K test files and shared settings for each model. It detects CUDA with `nvidia-smi`, otherwise selects CPU. After each successful training run, it moves the final `_stagetwo.model` file into `models/`.

| Model | Hidden irreps | Comparison |
| --- | --- | --- |
| `mace_k32_L0` | `32x0e` | Narrow scalar model |
| `mace_k32_L2` | `32x0e + 32x1o + 32x2e` | Narrow model with higher-order equivariant channels |
| `mace_k128_L0` | `128x0e` | Wide scalar model |
| `mace_k128_L2` | `128x0e + 128x1o + 128x2e` | Wide model with higher-order equivariant channels |

The shared configuration is defined in `scripts/train_grid.sh`: two interactions, `r_max=5.5 Å`, `max_ell=3`, correlation order 3, batch size 10, up to 120 epochs, SWA starting at epoch 90, energy/force weights 1/10, and seed 42.

### 3. Compare energy and force RMSE

```bash
python scripts/analysis/1_a_RMSE_comparison.py
```

The script prints energy RMSE in meV/atom and force RMSE in meV/Å for train, validation, and the 1200 K test set. It accepts `--train`, `--val`, `--test`, `--models`, and `--device` overrides.

### 4. Train SOAP error surrogates

```bash
python scripts/analysis/2_a_Surrogate_Model_Training.py
```

For each 1200 K test structure, the script computes an inner-averaged global SOAP power spectrum (`r_cut=5.0 Å`, `n_max=4`, `l_max=4`) and each MACE model's absolute total-energy error. It fits one `GradientBoostingRegressor` per MACE model and saves feature data, targets, fitted surrogates, and a JSON report under `results/surrogate/`.

### 5. Explain the surrogate models with TreeSHAP

```bash
python scripts/analysis/2_b_Angular_momentum_distribution.py
```

This computes per-structure TreeSHAP matrices, global mean absolute feature importance, six radial/angular category summaries, and Top-20 angular-momentum frequencies. Outputs are written to `results/angular_momentum/`. The default radial grouping uses the mean of the one-based SOAP radial indices `(n1+n2)/2` as a low/middle/high radial-order proxy; these indices do not directly represent distances in Å. The grouping thresholds and sum-versus-mean category aggregation are configurable.

### 6. Compare feature hierarchy

```bash
python scripts/analysis/2_c_Feature_hierarchy.py
```

This produces the pairwise overlap matrix of the models' Top-20 SOAP features and four ranked horizontal bar charts, with Cr-related features highlighted. Tables and figures are saved under `results/feature_hierarchy/`.

### 7. Analyze representation effective rank

```bash
python scripts/analysis/3_a_Attention_Flow_Analysis.py
```

The script captures node-embedding and final product-block features for the same test configurations across all models, centers channels, and computes spectral effective rank and normalized channel utilization. It writes feature matrices, singular values, a metrics table, and a comparison plot to `results/attention_flow/`. The default minimum is 1,000 pooled atoms. For the current 48-atom structures this requires 21 configurations, one above the preferred 10–20 range, to avoid a sample-count rank cap for `k=128`.

### Optional: detailed diagnosis

```bash
python scripts/analysis/diagnosis_new.py
```

This reports force errors, a defect-region activation score, and a SOAP/GBDT/SHAP summary. See [`docs/diagnosis_new.md`](docs/diagnosis_new.md) for definitions and limitations. The earlier script is retained at `scripts/legacy/diagnosis.py`.

## Reproducibility and interpretation

- Scripts use deterministic seeds where their algorithms support them. Dataset generation and training use seed 42.
- The four-model comparisons should use the same dataset frames and feature definitions.
- SHAP values describe the fitted surrogate's attribution on the supplied test structures; they are not direct causal explanations of MACE physics.
- Effective rank depends on the selected structures, centering, and the chosen layer output. It is a representation statistic, not a standalone accuracy metric.
- Generated datasets, trained model files, and analysis results are included in their named directories as the current project artifacts.
