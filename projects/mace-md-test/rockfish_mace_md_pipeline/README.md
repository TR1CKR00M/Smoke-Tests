# Rockfish self-contained MACE-MD continuation package

This folder is a standalone Rockfish runner for the MigrationBench MACE-MD continuation. It is intentionally separate from the existing Smoke-Tests project and does not overwrite an existing trajectory.

## What this runs

- ASE Langevin MD with a compiled MACE model on one Rockfish GPU.
- Exact continuation from the final extxyz frame, including finite momenta.
- `1 fs` timestep, `600 K`, friction `0.01`, and a frame every `200` steps (`0.2 ps`), matching the current MigrationBench Rockfish pipeline.
- Immutable segment files plus `latest_restart.extxyz` and `status.json` checkpoints.
- A fresh output directory per model/run; an existing segment is never overwritten.

The concrete example for the PR is `config.medium.example.json`: the 81-atom Cr:Sb2Te3 input and the MACE `medium` foundation model at 600 K. The same runner also supports the larger 2050-atom Rockfish continuation by changing the config.

This is a reproducibility/continuation runner. It does not claim that a trajectory is scientifically accepted merely because Slurm reports `COMPLETED`; inspect the final status, finite coordinates/momenta, frame count, and model/input provenance before using the result.

## Files

```text
rockfish_mace_md_pipeline/
├── README.md
├── config.example.json
├── config.medium.example.json
├── config.medium.smoke.cpu.example.json
├── config.medium.production.cpu.example.json
├── config.medium.production.gpu.example.json
├── continue_md.py
├── run_rockfish.slurm          # GPU
├── run_rockfish_cpu.slurm      # CPU fallback
├── verify_setup.py
├── monitor.sh
└── requirements-rockfish.txt
```

## CPU/GPU choice and run modes

Foundation-model MD is inference, so training on a GPU is not required. The same compiled MACE model can run with PyTorch on CPU; GPU is the preferred production path because CPU can be much slower.

| Purpose | Config | Slurm script | Meaning |
|---|---|---|---|
| Smoke, CPU | `config.medium.smoke.cpu.example.json` | `run_rockfish_cpu.slurm` | 2,000 steps = 2 ps; checks environment, input, model, checkpoint and output |
| Production, CPU fallback | `config.medium.production.cpu.example.json` | `run_rockfish_cpu.slurm` | 200,000 steps = 200 ps; use only if CPU wall time is acceptable |
| Production, GPU | `config.medium.production.gpu.example.json` | `run_rockfish.slurm` | 200,000 steps = 200 ps; preferred path |

Smoke output is only a pipeline check. It is not evidence for a 200 ps/1 ns scientific claim.

## Rockfish setup

Use the existing Rockfish environment that already contains the tested MACE/ASE stack. If a new environment is required, install `requirements-rockfish.txt` into a Rockfish-compatible Python environment; do not use the macOS `pip freeze` from the original repository.

```bash
module load gcc/9.3.0 pytorch/2.10.0-b200-py3.11
python -m pip install --user -r requirements-rockfish.txt
```

If MACE is already provided by your module/virtual environment, skip the pip command and run the setup check below.

## Prepare the `medium` example run

Copy this folder to a Rockfish project/scratch location, then make a private copy of the medium config. For a brand-new 81-atom run, the input must be the intended Smoke-Tests structure and must contain finite momenta before MD starts. For a continuation, use the prior run's final extxyz/checkpoint instead. The model must be the exact compiled `medium` model used for the benchmark.

```bash
cp config.medium.example.json config.medium.rockfish.json
vi config.medium.rockfish.json
python verify_setup.py --config config.medium.rockfish.json
```

Replace the three `/REPLACE/...` values before running. In particular, do not use the output directory as the input path. For the first smoke run, leave `initialize_momenta: true`; for an exact continuation, set it to `false` and provide a checkpoint containing momenta.

The example is deliberately not executable until the three absolute paths are replaced:

- `input_xyz`: `sb2te3_cr_81atom.xyz` or a prior final checkpoint, never the output path;
- `model_path`: the compiled MACE `medium` model;
- `output_dir`: a new run directory under your own Rockfish scratch/project area.

## Submit

Set the run-specific config in `run_rockfish.slurm` and submit:

```bash
sbatch --export=ALL,CONFIG=/absolute/path/config.medium.rockfish.json run_rockfish.slurm
```

For the recommended GPU production run:

```bash
cp config.medium.production.gpu.example.json config.medium.production.gpu.json
vi config.medium.production.gpu.json
python verify_setup.py --config config.medium.production.gpu.json
sbatch --export=ALL,CONFIG="$PWD/config.medium.production.gpu.json" run_rockfish.slurm
```

For the CPU fallback, use the CPU config and CPU batch script:

```bash
cp config.medium.smoke.cpu.example.json config.medium.smoke.cpu.json
vi config.medium.smoke.cpu.json
python verify_setup.py --config config.medium.smoke.cpu.json
sbatch --export=ALL,CONFIG="$PWD/config.medium.smoke.cpu.json" run_rockfish_cpu.slurm
```

After the 2 ps CPU smoke succeeds, change only the config to `config.medium.production.cpu.example.json` for the 200 ps CPU run. Use a new output directory; do not overwrite the smoke result.

Rockfish requires these Slurm settings in a non-interactive shell:

```bash
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
S=/cm/shared/apps/slurm/current/bin
"$S/sbatch" ...
```

The batch script requests one GPU. It does not request or assume a particular GPU UUID.

## Monitor

```bash
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
S=/cm/shared/apps/slurm/current/bin
"$S/squeue" -u "$USER"
"$S/sacct" -j JOBID --format=JobID,State,Elapsed,MaxRSS,AllocTRES,ExitCode
bash monitor.sh /absolute/path/config.medium.rockfish.json JOBID
```

`monitor.sh` reports scheduler state, the latest `status.json`, the latest restart frame, segment count, and the end of the Slurm log. It is diagnostic only; completion still requires output validation.

## Restart after interruption

The driver writes `latest_restart.extxyz` at checkpoint intervals and at normal exit. To continue, create a new config with:

- `input_xyz` set to the existing run's `latest_restart.extxyz`;
- a new `output_dir` (recommended) or the same run directory only when the next segment name is guaranteed to be unused;
- `start_step` equal to the `absolute_step` in the input run's `status.json`;
- `segment` set to a new name, for example `from_0002000`.

Never delete or replace an old segment. The restart preserves momenta and does not resample velocities.

## Acceptance before reporting results

```bash
python verify_setup.py --config config.medium.rockfish.json --check-output
```

Record the following in the PR/run ledger:

1. Slurm job ID and final state;
2. absolute input/model paths and SHA256 hashes;
3. `status.json` final step and time;
4. number of extxyz frames and segment names;
5. whether coordinates, momenta, energies/temperatures are finite;
6. any interruption/retry and the exact restart source.

`COMPLETED`, file existence, or an intermediate step is not by itself scientific convergence.
