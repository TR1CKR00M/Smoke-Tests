# Summary

<!-- What changed and why -->

# Reproduction

```bash
make setup
make test
make lint
# Project-specific command(s) that regenerate the changed outputs, e.g.
# uv run python projects/<project>/script/<entrypoint>.py --help
```

<!-- Paste the exact command(s) used to produce any committed CSV/figure. -->

# Results

<!-- Metrics or artifacts changed -->

# Checklist

- [ ] Tests pass (`make test`)
- [ ] Lint passes (`make lint`)
- [ ] Project README/report updated
- [ ] Outputs regenerated rather than hand-edited

## Provenance

- [ ] Every new or changed model/dataset has a recorded source (URL/DOI), version, license and sha256 (project manifest or registry)
- [ ] Every committed figure can be regenerated from a committed CSV with one recorded command
- [ ] DFT/reference calculations used as ground truth are converged, and their settings (functional, dispersion, k-points, spin) are recorded
- [ ] No large binaries (`*.model`, `*.traj`, `*.xyz`, ...) added to Git; external location + sha256 documented instead
