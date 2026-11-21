# Projects

Each experiment is an independent project. The only shared code is under `src/common/`.

| Folder | Role | Scope |
| --- | --- | --- |
| [`polar_benchmark`](polar_benchmark) | **Main line** | Benchmark of MACE foundation models (MH-1 neutral baseline, MACE-Polar zero-shot and fine-tuned) against public Sb2Se3 charged-defect (V_Sb1, q = -3..+1) DFT relaxations |
| [`mace-md-test`](mace-md-test) | Auxiliary | MACE molecular-dynamics tests on Cr:Sb2Te3 with per-run manifests; template for the planned charged-defect MD pilot |
| [`mace-off`](mace-off) | Practice | Reproduction exercise: MACE-OFF23 evaluation on SPICE data and torsion scans |
| [`defect_comparison`](defect_comparison) | Practice / archived | Cr-doped MoS2 defect comparison of MACE architectures with SOAP/SHAP diagnostics (labels from MACE-MP-0, not DFT) |
| [`data_cleaning_practice`](data_cleaning_practice) | Practice | Data-cleaning notebook exercise with a raw-data manifest |

New experiments should be added as a new folder with the same structure:

```text
projects/<new-project>/
  README.md
  configs/
  data/raw/
  src/<package>/
  results/{raw,processed,figures,reports}
  tests/
```

Tests placed in `projects/<project>/tests/` are collected automatically by the root
`pytest` configuration. They must run with the root dependencies only (CI does not
install `mace-torch`/`torch`), so guard heavy imports.

Do not share raw data or results across projects unless the dependency is explicit and documented.
