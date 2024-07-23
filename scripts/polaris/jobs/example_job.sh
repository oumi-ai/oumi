#!/bin/bash

#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:10:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -A community_ai
#PBS -o /eagle/community_ai/jobs/logs/
#PBS -e /eagle/community_ai/jobs/logs/

set -e

# Change to the directory where the job was submitted.
cd ${PBS_O_WORKDIR}

# Run several checks and export "LEMA_*" env vars.
source ./scripts/polaris/polaris_init.sh

# Set up default modules.
module use /soft/modulefiles

# Set up conda.
module load conda

# Activate the LeMa Conda environment.
conda activate /home/$USER/miniconda3/envs/lema

TRAIN_DATASETS="data.train.datasets=
- dataset_name: \"/eagle/community_ai/datasets/fineweb-edu/sample-10BT\"
  subset: \"default\"
  split: \"train\"
"

# Each batch should be 512 examples. With 4 GPUS and batch size 32 per GPU, we need
# 4 gradient accumulation steps.
torchrun \
    --nnodes=${LEMA_NUM_NODES} \
    --node-rank=${PBS_NODENUM} \
    --nproc-per-node=4 \
    --master-addr=${LEMA_MASTER_ADDR} \
    --master-port=8007 \
    -m lema.train \
    -c configs/lema/gpt2.pt.yaml \
    "training.run_name='gpt2.pt.${PBS_JOBID}'" \
    "$TRAIN_DATASETS" \
    "training.max_steps=100" \
    "training.include_performance_metrics=true" \
    "training.ddp_find_unused_parameters=false" \
    "training.dataloader_num_workers=2" \
    "training.dataloader_prefetch_factor=4" \
    "training.per_device_train_batch_size=32" \
    "training.gradient_accumulation_steps=4"
