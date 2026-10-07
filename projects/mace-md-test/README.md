# MACE-MD Test Project

This project runs restartable molecular dynamics simulations with MACE,
analyzes thermodynamic logs, and regenerates plots from CSV trajectories.

## Project layout

```text
mace-md-test/
├── data/
│   ├── metadata/          # Lists of input CSV files used for grouped plots
│   └── raw/               # Initial atomic structures
├── results/
│   ├── figures/            # Generated figures and plot manifest
│   ├── raw/                # Committed trajectory and CSV data
│   └── runs/               # Per-run MD outputs and manifests
├── src/
│   ├── offset_analysis.py  # Statistical comparison of two MD logs
│   ├── plot_comparison.py  # Legacy single-panel plotting entrypoint
│   ├── reproduce_plots.py  # Full figure reproduction command
│   └── run_md.py           # MD execution and restart logic
├── tests/                  # Automated checks
├── requirements.txt        # Portable direct dependency pins
├── requirements-dev.txt    # Runtime pins plus pytest
└── run_reproduction.sh     # Regenerate figures and run checks
```

## Install

Use Python 3.10–3.13. `requirements.txt` pins the direct runtime dependencies,
including `mace-torch==0.3.16` and `torch==2.14.0`. It is a portable set of
version constraints, not a full transitive lockfile; pip resolves compatible
transitive packages and wheels for the host platform.

```sh
python -m pip install -r requirements.txt
```

The default PyPI PyTorch build supports CPU execution, including on Apple
Silicon. For an NVIDIA GPU, first install the PyTorch 2.14.0 build matching the
machine's CUDA runtime with the [official PyTorch selector](https://pytorch.org/get-started/locally/),
then install the project requirements. MACE uses CUDA when available and CPU
otherwise; `--device cpu` forces CPU execution. The MACE package version stays
at 0.3.16 across these PyTorch backends.

To install the environment used by the full reproduction command, including
pytest:

```sh
python -m pip install -r requirements-dev.txt
```

## Reproduce figures

From the project root, run:

```sh
bash run_reproduction.sh
```

The script runs `src/reproduce_plots.py`, runs the checks in
`tests/test_reproduce_plots.py`, and verifies the PNG outputs. The plotting
command reads `results/raw/*_log.csv` and `data/metadata/group_1.txt`, then
writes four figure sets in `results/figures/`, each as PNG, SVG, and PDF:

- `atlas_multi_file_comparison`
- `atlas_multi_file_comparison_alpha_adjusted`
- `atlas_group_1`
- `atlas_time_sequence_analysis`

It also writes `atlas_plot_manifest.json`, recording input and generated-file
hashes, plotting parameters, script hash, and relevant library versions. The
plotting implementation is self-contained and does not require a local Plot
Atlas installation. To create the grouped trajectory panel for another
metadata file, run the compatibility entrypoint, for example:

```sh
python src/plot_comparison.py group_2.txt
```

## Run or restart an MD simulation

Run commands from the project root. A new run creates a unique directory under
`results/runs/` containing `trajectory.traj`, `thermo_log.csv`, and
`manifest.yaml`:

```sh
python src/run_md.py --temp 600 --steps 20000 --model medium-mpa-0
```

To continue a particular run, pass its directory explicitly:

```sh
python src/run_md.py --temp 600 --steps 20000 --model medium-mpa-0 \
  --parent-run-dir results/runs/<parent-run-directory>
```

Alternatively, `--restart` selects the latest run matching the requested model
and temperature. The manifest records simulation parameters, software
versions, input and output hashes, run lineage, and MACE model provenance:
requested model, resolved checkpoint path, known source URL or local-file
origin, and checkpoint SHA256. A restart is rejected if the parent's recorded
model hash differs from the current checkpoint hash. Keep the checkpoint file
to reuse the same model; model aliases can resolve differently across
`mace-torch` versions, and matching a model hash alone does not guarantee a
bit-for-bit identical simulation across hardware or software environments.
The Langevin random generator is seeded and its state is stored for continuation
from manifests created by this version. A restart must keep the parent's model,
temperature, timestep, friction, dtype, and logging interval. Earlier manifests
without a saved random state can be continued, but their stochastic sequence
cannot be reproduced exactly. When parent output hashes are present, restart
checks them before using the trajectory or log.

### Interpretation limits

The initial structure is an 81-atom slab with three-dimensional periodic
boundary conditions and vacuum along one cell direction. `pressure_GPa` is the
three-dimensional cell-averaged virial pressure; because it depends on the
chosen vacuum height, do not interpret it as an intrinsic two-dimensional
material pressure or compare it without specifying the cell convention. The
reported temperature currently uses the kinetic energy divided by `3N` degrees
of freedom, so it does not correct for the center-of-mass constraint.

## Compare two MD logs

`src/offset_analysis.py` is an exploratory comparison of pressure or
temperature in two CSV logs. It trims a time-based burn-in and reports
autocorrelation- and block-based uncertainty estimates. It does not validate
that the input time axes match. Its current offset metrics pair rows by index,
while its Welch-style significance test treats the two series as independent;
use its p-value only after validating those assumptions for the experiment.
For example:

```sh
python src/offset_analysis.py \
  --files results/raw/md_medium-mpa-0_600K_log.csv results/raw/md_medium_600K_log.csv \
  --var pressure --burn-in 0.20 --save-dir results/figures --no-show
```
