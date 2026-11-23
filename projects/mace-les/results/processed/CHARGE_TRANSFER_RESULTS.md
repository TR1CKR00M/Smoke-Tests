# C1 leave-one-charge-out results

All metrics are reported per held-out charge; folds are not pooled.
The q=−3 and q=+1 folds are charge-range extrapolation; q=−2, −1, and 0 are interpolation.
Each q test contains four held-out trajectories from two frozen splits. There is no test-fitted energy offset.
The benchmark transfers joint (charge, spin multiplicity) pairs; spin is fixed by q in this dataset, so their independent effects are not identifiable.
Some test geometries have exact matches at other q in training/validation. This is matched-geometry transfer, not geometric OOD; see the fold manifest.
MACELES energy/force scores do not establish physical latent-charge conservation or long-range electrostatic accuracy.

| q | Regime | Family | Spearman, seeds 42/43 | Force RMSE range (eV/Å) | Raw energy MAE range (eV) | Whole-test regret range (eV) | Per-path centered energy RMSE range (eV) |
|---:|---|---|---:|---:|---:|---:|---:|
| -3 | extrapolation | MACE-Q-continuous | 0.983, 0.975 | 0.064–0.070 | 0.282–3.527 | 0.0024–0.0048 | 0.056–0.503 |
| -3 | extrapolation | MACELES-Q-continuous | 0.986, 0.966 | 0.061–0.074 | 0.950–3.392 | 0.0041–0.0048 | 0.055–0.511 |
| -2 | interpolation | MACE-Q-continuous | 0.880, 0.771 | 0.054–0.059 | 1.090–1.440 | 0.0000–0.0005 | 0.125–0.214 |
| -2 | interpolation | MACELES-Q-continuous | 0.822, 0.843 | 0.052–0.056 | 0.205–0.902 | 0.0002–0.0002 | 0.171–0.222 |
| -1 | interpolation | MACE-Q-continuous | 0.959, 0.993 | 0.055–0.060 | 2.811–4.660 | 0.0000–0.0000 | 0.056–0.170 |
| -1 | interpolation | MACELES-Q-continuous | 0.963, 0.957 | 0.058–0.062 | 2.753–3.082 | 0.0000–0.0000 | 0.082–0.143 |
| +0 | interpolation | MACE-Q-continuous | 0.722, 0.668 | 0.068–0.072 | 2.432–2.824 | 0.0024–0.0024 | 0.043–0.151 |
| +0 | interpolation | MACELES-Q-continuous | 0.220, 0.192 | 0.074–0.078 | 1.547–6.400 | 0.2549–0.2570 | 0.045–0.319 |
| +1 | extrapolation | MACE-Q-continuous | -0.244, 0.780 | 3.251–61.934 | 20.998–612.484 | 0.1021–2.2188 | 1.500–94.287 |
| +1 | extrapolation | MACELES-Q-continuous | -0.433, -0.610 | 0.695–26.151 | 7.250–349.194 | 2.5215–18.7068 | 0.729–47.144 |

The frame-level and per-trajectory outputs are in the adjacent CSV files. These are fixed-cell single-point metrics, not formation energies, transition levels, or relaxed minima.

Input test SHA256 by q: `{"-1": "f2302ce727d121cb8b17b7dffa61307ecd155642ef58b3b97a23f53078024d77", "-2": "bd0b4a7c4f70f4c690811d83da67b5b9acf72d8b034031eed97b156e7f323788", "-3": "29b3e78fbd830df44df0950c91f5ec5a9f9a0eb742cedb3a03c19548b1840700", "0": "1f52a9319818a2e08ae5822423fdc0a396fb187320976f190ede4ca04f37029c", "1": "3d81e66a0dee582e79742a4c9ead0f2134b9cc8b710622e6174ce969ee2b7454"}`

Metrics CSV SHA256: `dad3e94e26cf56283de508d3483b46f5be62ae5dd853c6b58e65ac04490f2008`
Trajectory CSV SHA256: `810549e8c75a658e99f13c2ea222bb7bdac688570cce19d7b76cdbdbd68db68d`
