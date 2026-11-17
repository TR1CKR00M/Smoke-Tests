#!/bin/bash -l

# Submit a job to the GPU nodes by adding a request for a number of GPUs per node
#$ -l gpu=1

# Only Free jobs are available at present. Use your normal projects
#$ -P Free
#$ -A Imperial_Mat
#$ -N MACE

module unload compilers mpi
module unload gcc-libs
module load compilers/gnu/10.2.0
module load python
module load cuda/12.2.2/gnu-10.2.0
conda activate mace_chg

# Force bash as the executing shell.
#$ -S /bin/bash

# Request wallclock time (format hours:minutes:seconds).
#$ -l h_rt=04:00:00


# Set the Current Working Directory
#$ -cwd

ulimit -s unlimited
ulimit -m unlimited

python /home/mmm0798/miniconda3/envs/mace_chg/lib/python3.11/site-packages/mace/cli/run_train.py \
    --name="MACE" \
    --loss='universal' \
    --train_file="train.xyz" \
    --valid_file="test.xyz" \
    --energy_key="REF_energy" \
    --forces_key="REF_forces" \
    --compute_stress=True \
    --stress_key="REF_stress" \
    --eval_interval=1 \
    --error_table='PerAtomMAE' \
    --model="ScaleShiftMACE" \
    --pair_repulsion \
    --E0s="{34:-2.05295446, 51: -2.54939309}" \
    --scaling="rms_forces_scaling" \
    --num_interactions=2 \
    --correlation=3 \
    --max_ell=3 \
    --r_max=5.5 \
    --max_L=3 \
    --num_channels=128 \
    --num_radial_basis=10 \
    --energy_weight=1 \
    --forces_weight=100 \
    --stress_weight=10 \
    --lr=0.005 \
    --batch_size=4 \
    --max_num_epochs=150 \
    --weight_decay=1e-8 \
    --patience=40 \
    --start_swa=120 \
    --swa_energy_weight=1000.0 \
    --swa_forces_weight=10.0 \
    --swa_stress_weight=10.0 \
    --weight_decay=1e-8 \
    --patience=40 \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --default_dtype="float64" \
    --device='cuda' > out_$PBS_JOBID.txt 2>err_$PBS_JOBID.txt \
    --seed=3 \
    --save_cpu \
    --restart_latest \
    --total_charge_key="total_charge" \
    --embedding_specs="{'total_charge': {'type': 'categorical', 'per': 'graph', 'num_classes': 5, 'emb_dim': 64}}" \
    --use_embedding_readout=True \
