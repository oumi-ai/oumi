#!/bin/bash

#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:40:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -A community_ai
#PBS -o /eagle/community_ai/jobs/logs
#PBS -e /eagle/community_ai/jobs/logs

set -e

# Change to the directory where the job was submitted.
echo "Changing directory to ${PBS_O_WORKDIR} ..."
cd ${PBS_O_WORKDIR}

NRANKS=1  # Number of MPI ranks to spawn per node (1 worker per node)
NDEPTH=64 # Number of hardware threads per rank (Polaris has 64 CPU cores per node)
export POLARIS_GPUS_PER_NODE=4

# Run several checks and export "LEMA_*" env vars.
source ./scripts/polaris/polaris_init.sh

# Set up default modules.
module use /soft/modulefiles

# Set up conda.
module load conda

# Activate the LeMa Conda environment.
conda activate /home/$USER/miniconda3/envs/lema
echo "Conda path:"
echo $CONDA_PREFIX

export SHARED_DIR=/eagle/community_ai
export HF_HOME="${SHARED_DIR}/.cache/huggingface"

REPO="meta-llama"
MODEL="Meta-Llama-3.1-70B-Instruct"

MODEL_REPO="${REPO}/${MODEL}"
export SNAPSHOT_DIR="${REPO}--${MODEL}"
export SNAPSHOT=$(ls "${HF_HOME}/hub/models--${SNAPSHOT_DIR}/snapshots")

echo "Setting up vLLM inference with ${LEMA_NUM_NODES} node(s)..."

set -x  # Print command with expanded variables

# # Start worker nodes
mpiexec --verbose \
    --np $LEMA_NUM_NODES \
    --ppn ${NRANKS} \
    --depth ${NDEPTH} \
    --cpu-bind depth \
    ./scripts/polaris/jobs/vllm_worker.sh

echo "Polaris job is all done!"
