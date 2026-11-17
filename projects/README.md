# Projects

Each experiment is an independent project. The only shared code is under `src/common/`.

| Folder | Scope |
| --- | --- |
| [`defect_comparison`](defect_comparison) | Defect-data comparison workflow |
| [`mace-md-test`](mace-md-test) | MACE molecular-dynamics tests |
| [`polar_benchmark`](polar_benchmark) | Polar benchmark experiments |


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

Do not share raw data or results across projects unless the dependency is explicit and documented.
