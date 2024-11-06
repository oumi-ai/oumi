#!/bin/bash

#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=001:00:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -A community_ai
#PBS -o /eagle/community_ai/jobs/logs
#PBS -e /eagle/community_ai/jobs/logs

set -e

export MPICH_GPU_SUPPORT_ENABLED=1

export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

export SHARED_DIR=/eagle/community_ai
export HF_HOME="${SHARED_DIR}/.cache/huggingface"
REPO="neuralmagic"
MODEL="Meta-Llama-3.1-70B-Instruct-quantized.w8a8"
MODEL_REPO="${REPO}/${MODEL}"

# Various setup for running on Polaris.
source ${PBS_O_WORKDIR}/scripts/polaris/polaris_init.sh

set -x

huggingface-cli download "${MODEL_REPO}"

echo "Polaris job is all done!"
