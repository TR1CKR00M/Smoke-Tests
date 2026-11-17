#!/bin/bash
# step2_train_grid.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p models results/training

# MACE requires an available CUDA device when --device=cuda is selected.
# Fall back to CPU on machines without CUDA (including Apple Silicon).
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    DEVICE="cuda"
else
    DEVICE="cpu"
fi

COMMON_ARGS="--train_file=data/processed/defect_train_300K.xyz \
             --valid_file=data/processed/defect_val_300K.xyz \
             --test_file=data/processed/defect_test_1200K.xyz \
             --E0s=average \
             --model=MACE \
             --num_interactions=2 \
             --r_max=5.5 \
             --max_ell=3 \
             --correlation=3 \
             --batch_size=10 \
             --max_num_epochs=120 \
             --swa --start_swa=90 \
             --energy_weight=1.0 --forces_weight=10.0 \
             --energy_key=REF_energy --forces_key=REF_forces \
             --device=${DEVICE} --seed=42"

echo "1/4: Training wide scalar model (k=128, L=0)"
mace_run_train --name="mace_k128_L0" --hidden_irreps="128x0e" $COMMON_ARGS 2>&1 | tee results/training/mace_k128_L0.log
mv "mace_k128_L0_stagetwo.model" models/

echo "2/4: Training narrow scalar model (k=32, L=0)"
mace_run_train --name="mace_k32_L0"  --hidden_irreps="32x0e"  $COMMON_ARGS 2>&1 | tee results/training/mace_k32_L0.log
mv "mace_k32_L0_stagetwo.model" models/

echo "3/4: Training wide equivariant model (k=128, L=2)"
mace_run_train --name="mace_k128_L2" --hidden_irreps="128x0e + 128x1o + 128x2e" $COMMON_ARGS 2>&1 | tee results/training/mace_k128_L2.log
mv "mace_k128_L2_stagetwo.model" models/

echo "4/4: Training narrow equivariant model (k=32, L=2)"
mace_run_train --name="mace_k32_L2"  --hidden_irreps="32x0e + 32x1o + 32x2e"  $COMMON_ARGS 2>&1 | tee results/training/mace_k32_L2.log
mv "mace_k32_L2_stagetwo.model" models/
