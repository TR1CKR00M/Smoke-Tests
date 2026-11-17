#!/usr/bin/env bash
#SBATCH --job-name=mb-mace-md
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
: "${CONFIG:?submit with --export=ALL,CONFIG=/absolute/path/config.json}"

module load gcc/9.3.0 pytorch/2.10.0-b200-py3.11
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/continue_md.py" --config "$CONFIG"
