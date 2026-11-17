# Architecture

The repository separates unrelated experiments so that one project cannot accidentally mutate another project's inputs or outputs.

```text
README.md
pyproject.toml
Makefile
docs/
  ARCHITECTURE.md
  REPRODUCING.md
  PROVENANCE.md
  EVALUATION.md
  GOVERNANCE.md
projects/
  <project>/
    README.md
    configs/
    data/raw/
    src/<package>/
    results/{raw,processed,figures,reports}
    tests/
src/common/
```

## Extension pattern

Add a new project with the same layering: `configs/`, `data/raw/`, `src/<package>/`, `results/`, and `tests/`. Keep its data and outputs isolated unless a dependency is explicit.
