#!/bin/bash

#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:10:00
#PBS -l filesystems=home:eagle
#PBS -l singularity_fakeroot=true
#PBS -q debug
#PBS -A community_ai
#PBS -o /eagle/community_ai/jobs/logs
#PBS -e /eagle/community_ai/jobs/logs

set -e

# Change to the directory where the job was submitted.
echo "Changing directory to ${PBS_O_WORKDIR} ..."
cd ${PBS_O_WORKDIR}

export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

export SHARED_DIR=/eagle/community_ai

# Run several checks and export "LEMA_*" env vars.
source ./scripts/polaris/polaris_init.sh

# Set up default modules.
module use /soft/modulefiles

module use /soft/spack/gcc/0.6.1/install/modulefiles/Core

# Set up apptainer (docker-equivalent)
module load apptainer

set -x  # Print commands
export APPTAINER_TMPDIR="/home/$USER/lema/temp"
apptainer -v build --fakeroot "${SHARED_DIR}/apptainer/vllm_vllm_openai_v0.5.4.sif" docker://vllm/vllm-openai:v0.5.4

echo "Polaris job is all done!"
