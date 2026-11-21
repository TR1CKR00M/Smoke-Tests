# Project boundaries

The repository is a workspace, not a single experiment. Each project owns its own data, config, code, outputs, tests, and report.

| Project | Role | Scope |
| --- | --- | --- |
| `polar_benchmark` | **Main line** | MACE foundation models (MH-1 baseline, MACE-Polar zero-shot / fine-tuned) vs. public Sb2Se3 charged V_Sb1 DFT data |
| `mace-md-test` | Auxiliary | Cr:Sb2Te3 MACE MD tests; manifest format reused for the charged-defect MD pilot |
| `mace-off` | Practice | MACE-OFF23 SPICE / torsion reproduction exercise |
| `defect_comparison` | Practice / archived | MoS2:Cr MACE architecture comparison with SOAP/SHAP diagnostics (MACE-MP-0 labels) |
| `data_cleaning_practice` | Practice | Data-cleaning notebook exercise |

See [`projects/README.md`](../projects/README.md) for links and details.

## Isolation rules

1. Do not share raw data between projects unless the dependency is explicit and documented.
2. Do not write one project's outputs into another project's folder.
3. Keep configs and seeds local to the project.
4. Add a new project with its own `configs/`, `data/raw/`, `src/`, `results/`, and `tests/`.
5. Keep common utilities in `src/common/` only when they are genuinely reusable.
