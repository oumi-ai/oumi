#!/bin/bash

POLARIS_NODE_RANK=${PMI_RANK:=0}
POLARIS_GPUS_PER_NODE=4
# Reversing GPUs order to match Polaris CPU affinities:
# https://docs.alcf.anl.gov/polaris/hardware-overview/machine-overview/#polaris-device-affinity-information
export CUDA_VISIBLE_DEVICES=3,2,1,0
LOG_PREFIX="Node: ${POLARIS_NODE_RANK}:"

echo "${LOG_PREFIX} ***ENV BEGIN***"
echo "${LOG_PREFIX} PBS_JOBID: $PBS_JOBID"
echo "${LOG_PREFIX} USER: ${USER}"
echo "${LOG_PREFIX} LEMA_MASTER_ADDR: $LEMA_MASTER_ADDR"
echo "${LOG_PREFIX} LEMA_MASTER_PORT: $LEMA_MASTER_PORT"
echo "${LOG_PREFIX} LEMA_NUM_NODES: $LEMA_NUM_NODES"
echo "${LOG_PREFIX} PMI_LOCAL_RANK: $PMI_LOCAL_RANK"
echo "${LOG_PREFIX} PMI_RANK: $PMI_RANK"
echo "${LOG_PREFIX} NCCL_COLLNET_ENABLE: $NCCL_COLLNET_ENABLE"
echo "${LOG_PREFIX} NCCL_NET_GDR_LEVEL: $NCCL_NET_GDR_LEVEL"
echo "${LOG_PREFIX} NCCL_DEBUG: $NCCL_DEBUG"
echo "${LOG_PREFIX} NVIDIA info: $(nvidia-smi -L)"
ORIGINAL_TMPDIR="${TMPDIR}"
export TMPDIR="/tmp/${PBS_JOBID}/rank_${POLARIS_NODE_RANK}/"
echo "${LOG_PREFIX} TMPDIR: ${TMPDIR} ORIGINAL_TMPDIR: ${ORIGINAL_TMPDIR}"
echo "${LOG_PREFIX} CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "${LOG_PREFIX} ***ENV END***"

mkdir -p "$TMPDIR"

echo "${LOG_PREFIX} Starting training..."

# https://github.com/huggingface/tokenizers/issues/899#issuecomment-1027739758
export TOKENIZERS_PARALLELISM=false

set -x  # Print "accelerate launch" command with expanded variables
accelerate launch \
    --num_machines ${LEMA_NUM_NODES} \
    --machine_rank ${POLARIS_NODE_RANK} \
    --num_processes $((${LEMA_NUM_NODES} * ${POLARIS_GPUS_PER_NODE})) \
    --main_process_ip ${LEMA_MASTER_ADDR} \
    --main_process_port 8007 \
    --multi_gpu \
    --config_file configs/accelerate/llama.ddp.yaml \
    -m lema.train \
    -c configs/lema/llama8b.lora.yaml \
    "training.run_name='polaris.llama8b.lora.${PBS_JOBID}'" \
    "training.output_dir=/eagle/community_ai/${USER}/${PBS_JOBID}" \
    "training.per_device_train_batch_size=2" \
    "training.gradient_accumulation_steps=32" \
    "training.max_steps=406" \
    "training.save_steps=0" \
    "training.save_final_model=true"

echo "${LOG_PREFIX} All done!"
