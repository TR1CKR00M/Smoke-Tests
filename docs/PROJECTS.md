# Project boundaries

The repository is a workspace, not a single experiment. Each project owns its own data, config, code, outputs, tests, and report.

| Project | Scope |
| --- | --- |
| `defect_comparison` | Defect-data comparison workflow |
| `mace-md-test` | MACE molecular-dynamics tests |
| `polar_benchmark` | Polar benchmark experiments |

## Isolation rules

1. Do not share raw data between projects unless the dependency is explicit and documented.
2. Do not write one project's outputs into another project's folder.
3. Keep configs and seeds local to the project.
4. Add a new project with its own `configs/`, `data/raw/`, `src/`, `results/`, and `tests/`.
5. Keep common utilities in `src/common/` only when they are genuinely reusable.
