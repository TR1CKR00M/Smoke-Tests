# Governance

## Commit standards

Use conventional, descriptive commit titles, for example:

- `feat(house-prices): add reproducible training pipeline`
- `fix(jarvis): handle malformed rows`
- `docs: add provenance and reproduction guide`

Each commit should describe one logical change and leave tests passing.

## Issue policy

Open an issue for:

- missing or ambiguous data provenance,
- unavailable or non-deterministic experiments,
- reproducibility blockers,
- dependency/license ambiguity,
- unclear metric definitions.

Reproducibility blockers should include command attempted, expected result, observed result, input SHA-256, row count, and package versions.

## Pull request policy

Each PR should:

- state what changed and why
- list reproduction commands run
- link relevant issues
- include generated artifacts when they are part of the result
- update the relevant project README and report
