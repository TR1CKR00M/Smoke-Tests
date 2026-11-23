#!/usr/bin/env bash
#SBATCH --job-name=cd-ct-eval
#SBATCH --partition=l40s
#SBATCH --account=pclancy3_gpu
#SBATCH --qos=qos_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --array=0-19%8
#SBATCH --output=/scratch16/pclancy3/yi/Smoke-Tests/projects/mace-les/results/raw/slurm/c1-ct-eval-%A_%a.out
#SBATCH --error=/scratch16/pclancy3/yi/Smoke-Tests/projects/mace-les/results/raw/slurm/c1-ct-eval-%A_%a.err

set -euo pipefail
set +u
PROJECT_ROOT=/scratch16/pclancy3/yi/Smoke-Tests/projects/mace-les
RUN_ROOT="$PROJECT_ROOT/results/raw"
if ! type module >/dev/null 2>&1; then
  export ENABLE_LMOD="${ENABLE_LMOD:-}"
  source /etc/profile.d/modules.sh
fi
module load python/3.11.9 pytorch/2.4.1
set -u
source "$PROJECT_ROOT/.venv/bin/activate"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
cd "$PROJECT_ROOT"

TASK=${SLURM_ARRAY_TASK_ID}
Q_INDEX=$((TASK / 4))
REMAINDER=$((TASK % 4))
FAMILY_INDEX=$((REMAINDER / 2))
SEED_INDEX=$((REMAINDER % 2))
Q_VALUES=(-3 -2 -1 0 1)
Q_LABELS=(m3 m2 m1 p0 p1)
FAMILIES=(mace maceles)
SEEDS=(42 43)
Q=${Q_VALUES[$Q_INDEX]}
Q_LABEL=${Q_LABELS[$Q_INDEX]}
FAMILY=${FAMILIES[$FAMILY_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}
MODEL_ID="${FAMILY}_ct_q${Q_LABEL}_seed${SEED}"
MODEL="$RUN_ROOT/c1_${MODEL_ID}/${MODEL_ID}.model"
TEST_FILE="$PROJECT_ROOT/data/raw/charge_transfer_loo_4path/q_${Q_LABEL}/test.xyz"
OUTPUT_DIR="$RUN_ROOT/c1_${MODEL_ID}/heldout"

python "$PROJECT_ROOT/scripts/evaluate_charge_transfer.py" \
  --model "$MODEL" \
  --test "$TEST_FILE" \
  --charge "$Q" \
  --training-seed "$SEED" \
  --output-dir "$OUTPUT_DIR"
