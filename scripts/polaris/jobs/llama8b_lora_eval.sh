#!/bin/bash

#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -A community_ai
#PBS -o /eagle/community_ai/jobs/logs/
#PBS -e /eagle/community_ai/jobs/logs/

set -e

# Various setup for running on Polaris.
source ${PBS_O_WORKDIR}/scripts/polaris/polaris_init.sh

# NOTE: Update this variable to point to your own LoRA adapter:
EVAL_CHECKPOINT_DIR="/eagle/community_ai/models/meta-llama/Meta-Llama-3.1-8B-Instruct/sample_lora_adapters/2073171/"

if test ${LEMA_NUM_NODES} -ne 1; then
    echo "Evaluation can only run on 1 Polaris node. Actual: ${LEMA_NUM_NODES} nodes."
    exit 1
fi

echo "Starting evaluation for ${EVAL_CHECKPOINT_DIR} ..."

set -x # Enable command tracing.

accelerate launch \
      --num_processes=4 \
      --num_machines=1  \
      -m lema.evaluate  \
      -c configs/lema/llama8b.eval.yaml \
      "model.adapter_model=${EVAL_CHECKPOINT_DIR}"

echo -e "Finished eval on ${LEMA_NUM_NODES} node(s):\n$(cat $PBS_NODEFILE)"
echo "Polaris job is all done!"
