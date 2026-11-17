#!/usr/bin/env bash
#SBATCH --job-name=mace-md-short
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
## NVIDIA GPU 集群按平台要求启用或替换下面两行：
##SBATCH --partition=gpu
##SBATCH --gres=gpu:1

set -euo pipefail

cd /path/to/mace-md-test
mkdir -p logs results/runs

# 按集群实际环境选择：module、conda 或 venv。
# 例如：
# module load CUDA/<集群提供的版本>
source /pclancy3/elliot/mace-md-test/slurm/venv/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python -u src/run_md.py \
  --xyz data/raw/sb2te3_cr_81atom.xyz \
  --model medium-mpa-0 \
  --device cuda \
  --temp 600 \
  --steps 1000 \
  --log-every 20 \
  --seed 42 \
  --output-dir results/runs