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

# Various setup for running on Polaris.
source ${PBS_O_WORKDIR}/scripts/polaris/polaris_init.sh

# NOTE: Update this variable to point to your own LoRA adapter:
EVAL_CHECKPOINT_DIR="/eagle/community_ai/models/meta-llama/Meta-Llama-3.1-8B-Instruct/sample_lora_adapters/2073171/"

echo "Starting evaluation for ${EVAL_CHECKPOINT_DIR} ..."

NRANKS=4  # Spawn 4 MPI ranks per Polaris node (1 `lema.evaluate` for each GPU)
NDEPTH=16 # Number of threads per rank
CPU_BIND="numa"

set -x # Enable command tracing.
# python -m lema.evaluate \
#     -c configs/lema/llama8b.lora.eval.yaml
#     "model.adapter_model=${EVAL_CHECKPOINT_DIR}"

accelerate launch \
      --num_processes=4 \
      -m lema.evaluate \
      -c configs/lema/llama8b.lora.eval.yaml \
      "model.adapter_model=${EVAL_CHECKPOINT_DIR}"

# mpiexec --verbose \
#    --np $((${LEMA_NUM_NODES} * ${NRANKS})) \
#    -ppn ${NRANKS} \
#    -d ${NDEPTH} --cpu-bind "${CPU_BIND}" \
#    ./scripts/polaris/jobs/llama8b_lora_eval_worker.sh

echo -e "Finished eval on ${LEMA_NUM_NODES} node(s):\n$(cat $PBS_NODEFILE)"
echo "Polaris job is all done!"
