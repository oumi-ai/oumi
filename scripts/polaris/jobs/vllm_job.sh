#!/bin/bash

#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:40:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -A community_ai
#PBS -o /eagle/community_ai/jobs/logs
#PBS -e /eagle/community_ai/jobs/logs

# Various setup for running on Polaris.
source ./scripts/polaris/polaris_init.sh

export SHARED_DIR=/eagle/community_ai
export HF_HOME="${SHARED_DIR}/.cache/huggingface"

REPO="meta-llama"
MODEL="Meta-Llama-3.1-70B-Instruct"

export SNAPSHOT_DIR="${REPO}--${MODEL}"
export SNAPSHOT=$(ls "${HF_HOME}/hub/models--${SNAPSHOT_DIR}/snapshots")

echo "Setting up vLLM inference with ${LEMA_NUM_NODES} node(s)..."

set -x  # Print command with expanded variables

# Start worker nodes
mpiexec --verbose \
    --np ${LEMA_NUM_NODES} \
    --ppn ${NRANKS} \
    --depth ${NDEPTH} \
    --cpu-bind ${CPU_BIND} \
    ./scripts/polaris/jobs/vllm_worker.sh

echo "Polaris job is all done!"
