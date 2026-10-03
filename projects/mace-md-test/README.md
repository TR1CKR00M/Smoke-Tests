# MACE-MD-TEST

Running MD simulations via MACE, with committed raw trajectories, manifests,
and reproducible plotting.

## Directory Structure

```text
mace-md-test/
├── data/
│   └── raw/              # Raw input structures
├── results/              # Output data, MD trajectories and plots
├── tests/                # Scripts for verifying execution and linting
├── README.md             # Project documentation
├── requirements.txt      # Python dependencies
└── run_reproduction.sh   # Bash script to execute the full test pipeline
```

## Usage

### Reproduce committed figures

The checked-in raw CSV files are the source of truth for the figures. This
command regenerates four Atlas-style figures as PNG, SVG, and PDF, writes an
input-hash manifest, and runs schema checks:

```text
bash run_reproduction.sh
```

The outputs are named `atlas_multi_file_comparison`,
`atlas_multi_file_comparison_alpha_adjusted`, `atlas_group_1`, and
`atlas_time_sequence_analysis`. The plotting grammar is Plot Atlas
`timecourse`; the implementation is self-contained so the result does not
depend on an untracked local plotting installation.

### Restartable MD simulation

Run `tests/run_md.py` with an explicit `--parent-run-dir` for a restart. Each
run writes its own trajectory, thermodynamic log, and `manifest.yaml`. The
manifest records parameters, software versions, input/output hashes, and the
parent run. The old `run_md_auto.py` and CSV merge entrypoints were removed in
this PR and should not be referenced by documentation.


