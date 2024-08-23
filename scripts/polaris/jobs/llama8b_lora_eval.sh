#!/bin/bash

#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:60:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -A community_ai
#PBS -o /eagle/community_ai/jobs/logs/
#PBS -e /eagle/community_ai/jobs/logs/

set -e

# Change to the directory where the job was submitted.
echo "Changing directory to ${PBS_O_WORKDIR} ..."
cd ${PBS_O_WORKDIR}

# Run several checks and export "LEMA_*" env vars.
source ./scripts/polaris/polaris_init.sh

# Set up default modules.
module use /soft/modulefiles

# Set up conda.
module load conda

# Activate the LeMa Conda environment.
conda activate /home/$USER/miniconda3/envs/lema

# NOTE: Update this variable to point to your own LoRA adapter:
EVAL_CHECKPOINT_DIR="/eagle/community_ai/models/meta-llama/Meta-Llama-3.1-8B-Instruct/sample_lora_adapters/2072919/"

echo "Starting evaluation for ${EVAL_CHECKPOINT_DIR} ..."

set -x # Enable command tracing.
python -m lema.evaluate \
      -c configs/lema/llama8b.lora.eval.yaml \
      "model.adapter_model=${EVAL_CHECKPOINT_DIR}"

echo "Polaris job is all done!"
