#!/usr/bin/env bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPT_DIR"

#pip install -r requirements.txt

python tests/run_foundation_md.py --model medium-mpa-0 --temp 600 --steps 2000
python tests/run_foundation_md.py --model medium-0b3 --temp 600 --steps 2000
python tests/plot_temp_pressure.py

if [ ! -f "results/figures/multi_file_comparison.png" ]; then
    echo "Error: no expected figures generated"
    exit 1
fi

echo "=== Verification success! ==="