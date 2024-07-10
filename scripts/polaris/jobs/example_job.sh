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
conda activate base

# Set up a virtual python environment.
mkdir -p ./worker_venv/example_environment
python3 -m venv ./worker_venv/example_environment --system-site-packages
source ./worker_venv/example_environment/bin/activate

python3 -m pip install -e '.[train]'

torchrun \
    --nnodes=${LEMA_NUM_NODES} \
    --node-rank=${PBS_NODENUM} \
    --nproc-per-node=4 \
    --master-addr=${LEMA_MASTER_ADDR} \
    --master-port=8007 \
    -m lema.train \
    -c configs/lema/gpt2.pt.yaml \
    "training.run_name='gpt2.pt.${PBS_JOBID}'" \
    "training.ddp_find_unused_parameters=false" \
    "training.dataloader_num_workers=2" \
    "training.dataloader_prefetch_factor=4" \
