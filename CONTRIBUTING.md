# Contributing

1. Create a feature branch.
2. Add or update the relevant project under `projects/`.
3. Run `uv sync --extra dev`.
4. Run `make lint` and `make test`.
5. Regenerate outputs with `make reproduce` if outputs are part of the change.
6. Open a PR with a clear description and result links.

## Adding a new experiment

Create:

```text
projects/<new-project>/
  README.md
  configs/experiment.yaml
  data/raw/
  src/<package>/
  results/{raw,processed,figures,reports}
  tests/
```

Do not share experiment data across projects unless the data dependency is explicit.
