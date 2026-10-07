#!/usr/bin/env bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPT_DIR"

python src/reproduce_plots.py
python -m pytest -q tests/test_reproduce_plots.py

for figure in \
    results/figures/atlas_multi_file_comparison.png \
    results/figures/atlas_multi_file_comparison_alpha_adjusted.png \
    results/figures/atlas_group_1.png \
    results/figures/atlas_time_sequence_analysis.png; do
    test -s "$figure"
done

echo "=== Plot reproduction verification success! ==="
