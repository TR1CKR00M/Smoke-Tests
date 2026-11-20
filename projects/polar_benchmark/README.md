# Polar benchmark: MACE foundation models for charged point defects

**Main line of the repository.**

## Question

Can MACE-Polar foundation models, zero-shot and after fine-tuning on public DFT
data, describe *charged* point defects reliably? "Reliably" means: (1) structures,
energies and forces, (2) discrimination between charge states, (3) localisation
of the extra charge near the defect, (4) stability at finite temperature.

Model ladder (same test set and metrics for all):
plain MACE (MACE-MH-1, no charge input: neutral baseline / control) →
MACE-Polar zero-shot → MACE-Polar fine-tuned → (later) other models.

The test system is the antimony vacancy V_Sb(1) in Sb2Se3 in charge states
q = −3 … +1 with ShakeNBreak-style bond distortions. The target system of the
wider project is Sb2Te3, so the Sb2Se3 → Sb2Te3 transfer must be discussed
explicitly.

## Data layout (`data/`)

| Folder | Meaning |
| --- | --- |
| `1_FM_benchmark/V_Sb1/DFT_PBE/` | Foundation-model benchmark: V_Sb1 relaxations, PBE+D3(zero damping, IVDW=11), Γ-only, VASP 6.4.3 |
| `2_global_embedding/V_Se1_DFT_HSE06/` | V_Se1 (note: **Se** vacancy) relaxations with HSE06+D3(zero), Γ-only, VASP 6.2.0; training data for the charge-embedded MACE |
| `2_global_embedding/Charge_embedded_MACE/` | Charge-embedded MACE (`total_charge` categorical embedding, 5 classes): `run.sh`, `train.xyz`, `test.xyz`, model |
| `3_MF/MF/` | Multi-fidelity (PBE/HSE multi-head) MACE for V_Se1 neu / pl1; DFT settings templates; one MF-MACE prediction re-optimised by DFT |

Inside each stage:

* **Charge-state folders** `mi3, mi2, mi1, neu, pl1, pl2` = q = −3, −2, −1, 0, +1, +2.
  Verified for every OUTCAR: q = (neutral valence electrons) − NELECT, e.g.
  V_Sb1 neutral NELECT = 331 (23 Sb × 5 + 36 Se × 6), `mi3` = 334, `pl1` = 330.
  (The commented `# NELECT = 329` in `INCAR_neu` is stale.)
* **Distortion folders** `D_±N` = ShakeNBreak bond distortion of the vacancy
  neighbours by ±N % (`D_0` = rattled, undistorted). `Unperturbed` and `Rattled`
  are the ShakeNBreak reference structures (only present for some charges).
  The POSCAR comment line holds the approximate fractional defect site,
  e.g. `~[0.4,0.5,0.4]` — *meaning inferred from ShakeNBreak conventions, to confirm*.
* `POSCAR*` = initial structure, `CONTCAR*` = DFT-relaxed structure, `OUTCAR*` = DFT log.

`data/DATA_MANIFEST.csv` lists **every** data file with sha256, and for each OUTCAR
the NELECT, NELECT-derived charge (and whether it agrees with the folder label),
convergence, ionic steps, final energy, final magnetisation, functional,
dispersion, k-point grid, ENCUT and VASP version. Regenerate with

```bash
uv run python projects/polar_benchmark/script/build_manifest.py
```

Models: see [`MODEL_REGISTRY.md`](MODEL_REGISTRY.md) / `MODEL_REGISTRY.csv`.

## Static relaxation benchmark (stage 1)

`script/mace_relax_comparison.py` relaxes each initial POSCAR with a MACE model
(BFGS, `fmax` 0.05 eV/Å, ≤ 200 steps) and compares with DFT:

* **RMSD** uses minimum-image displacements (`ase.geometry.find_mic`) after checking
  identical species order; no rigid alignment (same cell, vacancy fixes the frame).
  The **defect-local RMSD** uses atoms within `--local-cutoff` (default 4 Å, minimum
  image) of the vacancy site. The site is parsed from the POSCAR comment
  (`~[x,y,z]`, rounded to 0.1 in fractional coordinates, i.e. up to ~0.6 Å off) or
  given explicitly with `--defect-site FX FY FZ`.
* **Ground state** per (model, charge) is the lowest-energy distortion, chosen
  independently for MLIP and DFT among distortions where both are usable.
  Reported: ground-state hit (same distortion), RMSD(MLIP GS, DFT GS), DFT energy
  cost of the MLIP choice, Spearman ρ of the distortion energies.
  `diag_min_rmsd_*` (the old "minimum RMSD over distortions") is kept only as a
  labelled diagnostic.
* **DFT references** that did not reach the required accuracy are excluded from
  energy ranking and logged (`dft_excluded_reason`); `--include-unconverged-refs`
  overrides this.
* BFGS convergence and step count are recorded; only converged relaxations enter
  the ranking. Failures are written to the `error` column, never dropped.
* `atoms.info["total_charge"] = q` is set for every structure (read by charge-aware
  models, ignored by MH-1; `--no-total-charge` disables it).
* `--head` (default `omat_pbe`) is applied only to multi-head models; an unknown head
  is an error (mace-torch would otherwise silently fall back to the last head).
* `--d3` / `--dispersion d3-zero` adds DFT-D3 (zero damping, PBE) via torch-dftd to
  match IVDW=11. **Default is no dispersion** (backward compatible); the `omat_pbe`
  head is plain PBE, so the default is *not* like-for-like with the PBE+D3 reference.
* MACE-Polar: `--model NAME=polar:<key-or-path>`, e.g. `--model Polar-1-S=polar:polar-1-s`.

Outputs: `results/processed/relax_per_structure.csv`, `relax_per_charge.csv`,
`relax_run_metadata.json` (command, versions, model sha256), relaxed structures in
`results/raw/<model>/<charge>_<distortion>.extxyz`. The figure is regenerated from
the CSV only:

```bash
uv run python projects/polar_benchmark/script/plot_relax_comparison.py
```

### Reproduce

```bash
python3.11 -m venv .venv-mace && .venv-mace/bin/pip install -r projects/polar_benchmark/requirements.txt
.venv-mace/bin/python projects/polar_benchmark/script/mace_relax_comparison.py --device cpu
uv run python projects/polar_benchmark/script/plot_relax_comparison.py
```

The exact command and package versions of the committed results are in
`results/processed/relax_run_metadata.json`. ~30 s per relaxation on a laptop CPU.

`fig1a_mace_mh1_smoke.png` is the earlier smoke-test figure from the previous version
of the script (non-minimum-image RMSD, min over distortions); it cannot be regenerated
from a CSV and is superseded by `results/figures/`.

## Tests

`projects/polar_benchmark/tests/` (collected by the root `pytest`) covers the
minimum-image RMSD, OUTCAR parsing on committed OUTCARs (converged `neu/D_0`,
unconverged `pl1/D_-60`), ground-state / hit / Spearman logic and the manifest
builder. They need only the root environment (no mace/torch).

## Known issues (for the paper and future runs)

* **Unconverged reference**: `1_FM_benchmark/.../pl1/D_-60` stopped after 135 ionic
  steps (NSW = 500, so probably wall time) without reaching the required accuracy, and
  its energy is ~4 eV above the other pl1 distortions. Excluded by default; rerun or drop.
* **Mixed levels of theory**: stage 1 PBE+D3(zero), Γ-only; stage 2 HSE06+D3(zero),
  Γ-only; stage 3 training settings mix `INCAR_PBE_opt_neu` with **IVDW=12 (D3BJ)** and
  `INCAR_HSE06_single-point_neu` with IVDW=11, both with a **2×2×2** k-grid. State which
  settings define the reference.
* **Spin**: no MAGMOM is set; final magnetisations differ within a charge state
  (e.g. V_Sb1 `pl1` and `mi1`: 0 and 2 μB). Check that the spin state is consistent
  before comparing energies.
* **No finite-size / charge corrections** are applied to the DFT energies (only
  distortions *within* one charge state are compared here, where they cancel).
* **Target system**: data are Sb2Se3 (V_Sb1, V_Se1), the project target is Sb2Te3.
* **Dataset citation / license** of the public Sb2Se3 data is missing — add DOI and license.
* **MACE-Polar results and the five fine-tuning runs** (configs, logs, seeds) are not
  in the repository yet.
* **E0 mismatch**: `Charge_embedded_MACE/run.sh` uses E0(Se) = −2.05295446 eV, while
  `3_MF/.../e0s_hse.json` has E0(Se) = −2.05791249 eV (Sb agrees to 2e-7 eV). Confirm
  which isolated-atom energies are correct.
* **Job scripts**: `run.sh` / `run_mh.sh` use SGE directives (`#$`) but redirect output
  to `out_$PBS_JOBID.txt` (a PBS variable, empty under SGE). The redirection sits in the
  middle of the argument list (bash still passes the later flags, but it is easy to
  misread), `run.sh` repeats `--weight_decay`/`--patience`, and the mace path is a
  personal conda path. Left unchanged here; tidy when re-running.
* `mi3/D_0` and `neu/D_0` POSCARs are byte-identical. Because MH-1 has no charge input,
  its energies and relaxed structures are identical for all q given the same start
  (verified: setting `total_charge` does not change the MH-1 energy). This is the
  expected control, not a model property.
* Large binaries (`*.model`, `*.xyz`) are still tracked in Git; moving them to
  external storage is pending.
