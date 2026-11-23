#!/usr/bin/env bash
#SBATCH --job-name=cd-ct-fit
#SBATCH --partition=l40s
#SBATCH --account=pclancy3_gpu
#SBATCH --qos=qos_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-19%8
#SBATCH --output=/scratch16/pclancy3/yi/Smoke-Tests/projects/mace-les/results/raw/slurm/c1-ct-fit-%A_%a.out
#SBATCH --error=/scratch16/pclancy3/yi/Smoke-Tests/projects/mace-les/results/raw/slurm/c1-ct-fit-%A_%a.err

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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
cd "$PROJECT_ROOT"

TASK=${SLURM_ARRAY_TASK_ID}
Q_INDEX=$((TASK / 4))
REMAINDER=$((TASK % 4))
FAMILY_INDEX=$((REMAINDER / 2))
SEED_INDEX=$((REMAINDER % 2))
Q_VALUES=(-3 -2 -1 0 1)
Q_LABELS=(m3 m2 m1 p0 p1)
MODELS=(MACE MACELES)
FAMILIES=(mace maceles)
SEEDS=(42 43)
Q=${Q_VALUES[$Q_INDEX]}
Q_LABEL=${Q_LABELS[$Q_INDEX]}
MODEL=${MODELS[$FAMILY_INDEX]}
FAMILY=${FAMILIES[$FAMILY_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}
MODEL_ID="${FAMILY}_ct_q${Q_LABEL}_seed${SEED}"
DATA_ROOT="$PROJECT_ROOT/data/raw/charge_transfer_loo/q_${Q_LABEL}"
RUN_DIR="$RUN_ROOT/c1_${MODEL_ID}"
mkdir -p "$RUN_DIR/checkpoints" "$RUN_DIR/logs" "$RUN_DIR/results"

CHARGE_ARGS=(--embedding_specs="{'total_charge': {'type': 'continuous', 'per': 'graph', 'in_dim': 1, 'emb_dim': 16}, 'total_spin': {'type': 'categorical', 'per': 'graph', 'num_classes': 2, 'emb_dim': 8, 'offset': -1}}")
python -c 'import torch; assert torch.cuda.is_available(); print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0))'

mace_run_train \
  --name="$MODEL_ID" \
  --model="$MODEL" \
  --train_file="$DATA_ROOT/train.xyz" \
  --valid_file="$DATA_ROOT/valid.xyz" \
  --energy_key=REF_energy \
  --forces_key=REF_forces \
  --E0s=average \
  "${CHARGE_ARGS[@]}" \
  --num_interactions=2 \
  --num_channels=64 \
  --correlation=3 \
  --max_ell=2 \
  --max_L=2 \
  --r_max=5.5 \
  --num_radial_basis=8 \
  --energy_weight=1 \
  --forces_weight=100 \
  --stress_weight=0 \
  --batch_size=4 \
  --valid_batch_size=4 \
  --max_num_epochs=100 \
  --patience=15 \
  --eval_interval=1 \
  --ema \
  --default_dtype=float32 \
  --device=cuda \
  --seed="$SEED" \
  --save_cpu \
  --work_dir="$RUN_DIR" \
  --model_dir="$RUN_DIR" \
  --checkpoints_dir="$RUN_DIR/checkpoints" \
  --log_dir="$RUN_DIR/logs" \
  --results_dir="$RUN_DIR/results"
