# C2 leave-one-charge-state-out benchmark protocol

Status: protocol and five leave-one-charge-out folds prepared locally and on
Rockfish with identical manifest SHA256. Continuous-Q interface preflight
`33260426` completed with exit code 0. Training array `33260438` and its
afterok evaluation array `33260439` are submitted and awaiting Rockfish
scheduling. This stage uses only frozen V_Sb1 PBE+D3 data and creates no new
DFT labels.

## Question and interpretation

Can a model trained on four charge/multiplicity pairs predict energies and
forces for a fifth, wholly unseen pair in the same 59-atom cell? This is a
controlled joint charge/multiplicity transfer test. Spin multiplicity is
deterministic for each charge in this dataset, so its independent causal
contribution cannot be separated from Q. This does not establish defect-type
transfer, supercell-size transfer, or formation energies over a Fermi-level
range.

The completed C1 models use a categorical five-class Q embedding. In a
leave-one-charge-out test, the held-out class vector would receive no training
gradient, so those C1 checkpoints cannot answer whether Q interpolation works.
The pinned MACE code supports a continuous feature MLP through
`GenericJointEmbedding`; use a numerical `total_charge` input and a categorical
`total_spin` input. The same conditioning is used for from-scratch MACE-Q and
MACELES-Q. Including spin makes the input complete, but each Q has only one
spin multiplicity in this data, so Q and spin remain perfectly confounded.

## Frozen data and folds

- Source train/validation split:
  `data/split_seed_20261122/{train,valid}.xyz`.
- Test sources: q trajectories from the primary C1 test split
  (`data/split_seed_20261120/test.xyz`) and the independent split-2 test
  (`data/split_seed_20261122/test.xyz`).
- Fold (q): remove every q frame from train and validation; test only q
  trajectories from the two frozen test splits. Do not use test frames from
  other charges.
- All five charge states are tested once. Train and validation contain the
  other four states only. Each q has four held-out paths: two from each C1
  test split, disjoint by trajectory ID. No q-state frame from either test
  set is used for training or validation, and the test is never used for
  early stopping or calibration.
- The C1 split assigns trajectories separately within each q. Some test
  distortion geometries have exact matches in train/validation at another q.
  This is intentional matched-geometry charge transfer, not geometric OOD;
  exact cross-q geometry matches are counted in the fold manifest.
- The preparation script writes charge counts, trajectory IDs, and SHA256s for
  every generated file under `data/charge_transfer_loo/`.

## Models and run matrix

Train MACE-Q and MACELES-Q from scratch for all five held-out charges at seeds
42 and 43 (20 fits). Keep the C1 backbone, optimizer, loss weights, epoch cap,
and early-stopping policy unchanged. Use continuous scalar Q
(`in_dim=1`, `emb_dim=16`) plus categorical spin (2 classes, `emb_dim=8`).
Record exact pinned source/runtime and each fold manifest hash. MACELES latent
charges remain response variables; this experiment does not treat them as
physical atomic charges.

The current Polar C1 continuation and held-out evaluation should finish first.
Only add leave-one-charge-out Polar fine-tunes after the primary C1 held-out
score is reviewed; that prevents spending five long fine-tunes before knowing
whether the transferred PBE model is competitive.

## Metrics

For each held-out q and each seed/model, report raw energy MAE/RMSE, force
component RMSE/MAE, within-q energy Spearman, whole-test selection regret, and
per-trajectory ranking/regret plus trajectory-centered energy MAE/RMSE. There
is no validation offset for an unseen q; do not fit a correction on the test
set. A held-out minimum is among fixed single-point frames, not a relaxed
ground state. Preserve every test prediction and the exact model/test hashes.

Compare seed and trajectory variation explicitly. Each charge has four
held-out paths, all drawn from the same 66-path source pool. If a charge appears
difficult or a model-family ordering changes, add a new path-disjoint source
set before making a general claim.
After retrieving model and evaluation outputs, run
`summarize_c1_charge_transfer.py` to write the split-specific per-charge and
per-trajectory CSVs and the compact results report.

## Gates and reporting

1. Run the continuous-Q one-epoch interface preflight before the 20-fit array.
2. Verify every fold contains four train/validation charges and only the held
   out q in test; verify all output hashes locally against Rockfish.
3. Interpret energy ranking/regret and force transfer separately from raw
   total-energy error. Do not claim transition-level or defect formation
   energy accuracy from these tests.
4. If neither continuous-Q model transfers, inspect charge/spin conditioning
   and energy references before extending to defect type. If one family is
   competitive, repeat promising folds on a larger path-disjoint set and only
   then compare Polar transfer.

## Expected limitations

The folds use one defect family, one cell size, one functional, and four
held-out paths per q from the same source pool. Some test geometries are
matched across q in training, so this is not a geometry-OOD test. Even good
scores would show transfer to unseen charge/multiplicity pairs for V_Sb1
geometries, not universal electrostatic transfer. V_Se PBE+D3 data and
charge-density grids are still absent; defect-type transfer, physical charge
localization, and D2 remain separate data-gated stages. LES energy/force
scores on a fixed cell do not by themselves prove that LES learned the correct
long-range electrostatic physics; latent-charge conservation/localization and
size or pair diagnostics remain separate checks.
