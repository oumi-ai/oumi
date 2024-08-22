#!/bin/bash

#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=001:00:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -A community_ai
#PBS -o /eagle/community_ai/jobs/logs
#PBS -e /eagle/community_ai/jobs/logs

export MPICH_GPU_SUPPORT_ENABLED=1

set -e

# Change to the directory where the job was submitted.
echo "Changing directory to ${PBS_O_WORKDIR} ..."
cd ${PBS_O_WORKDIR}

export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

export SHARED_DIR=/eagle/community_ai
export HF_HOME="${SHARED_DIR}/.cache/huggingface"
REPO="neuralmagic"
MODEL="Meta-Llama-3.1-70B-Instruct-quantized.w8a8"
MODEL_REPO="${REPO}/${MODEL}"

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

set -x  # Print command with expanded variables

huggingface-cli download "${MODEL_REPO}"

echo "Polaris job is all done!"
