#!/bin/bash

#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:40:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -A community_ai
#PBS -o /eagle/community_ai/jobs/logs
#PBS -e /eagle/community_ai/jobs/logs

set -e

# Various setup for running on Polaris.
source ${PBS_O_WORKDIR}/scripts/polaris/polaris_init.sh

export SHARED_DIR=/eagle/community_ai
export HF_HOME="${SHARED_DIR}/.cache/huggingface"

export OPENAI_API_KEY="EMPTY"

export SNAPSHOT_DIR="${REPO}--${MODEL}"
SNAPSHOTS=$(ls "${HF_HOME}/hub/models--${SNAPSHOT_DIR}/snapshots")
readarray -t SNAPSHOT_ARRAY <<<"$SNAPSHOTS"
export SNAPSHOT=${SNAPSHOT_ARRAY[-1]}

echo "Setting up vLLM inference with ${OUMI_NUM_NODES} node(s)..."

set -x

# Start worker nodes
mpiexec --verbose \
    --np ${OUMI_NUM_NODES} \
    --ppn ${NRANKS} \
    --depth ${NDEPTH} \
    --cpu-bind ${CPU_BIND} \
    ./scripts/polaris/jobs/vllm_worker.sh

echo "Polaris job is all done!"
