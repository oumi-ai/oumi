#!/bin/bash

POLARIS_NODE_RANK=${PMI_RANK:=0}
POLARIS_NUM_GPUS_PER_NODE=4
# Reversing GPUs order to match Polaris CPU affinities:
# https://docs.alcf.anl.gov/polaris/hardware-overview/machine-overview/#polaris-device-affinity-information
export CUDA_VISIBLE_DEVICES=3,2,1,0
LOG_PREFIX="Node: ${POLARIS_NODE_RANK}:"

echo "${LOG_PREFIX} ***ENV BEGIN***"
echo "${LOG_PREFIX} PBS_JOBID: $PBS_JOBID"
echo "${LOG_PREFIX} USER: ${USER}"
echo "${LOG_PREFIX} OUMI_MASTER_ADDR: $OUMI_MASTER_ADDR"
echo "${LOG_PREFIX} OUMI_MASTER_PORT: $OUMI_MASTER_PORT"
echo "${LOG_PREFIX} OUMI_NUM_NODES: $OUMI_NUM_NODES"
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

ALLOWED_TRAINING_MODES=("sft", "lora")
ALLOWED_MODEL_SIZES=("8b", "70b")

helpFunction() {
    echo ""
    echo "Usage: $0 -m (sft/lora) -s (8b/70b)"
    echo -e "\t-m The training mode: ${ALLOWED_TRAINING_MODES[@]}. Defaults to lora."
    echo -e "\t-s The model size: ${ALLOWED_MODEL_SIZES[@]}. Defaults to 8b."
    exit 1 # Exit script after printing help
}

# Default value.
TRAINING_MODE="lora"
MODEL_SIZE="8b"

# Get values from command line and verify.
while getopts ":m:s:" opt; do
    case "$opt" in
    m) TRAINING_MODE="$OPTARG" ;;
    s) MODEL_SIZE="$OPTARG" ;;
    ?) helpFunction ;; # Print a help message for an unknown parameter.
    esac
done
if [ -z "$TRAINING_MODE" ]; then
    echo "Training mode can't be empty."
    helpFunction
fi
if ! (echo "${ALLOWED_TRAINING_MODES[@]}" | grep -q -w "${TRAINING_MODE}"); then
    echo "Unknown training mode: ${TRAINING_MODE}. Valid values: ${ALLOWED_TRAINING_MODES[@]}"
    helpFunction
fi
if [ -z "$MODEL_SIZE" ]; then
    echo "Model size can't be empty."
    helpFunction
fi
if ! (echo "${ALLOWED_MODEL_SIZES[@]}" | grep -q -w "${MODEL_SIZE}"); then
    echo "Unknown model size: ${MODEL_SIZE}. Valid values: ${ALLOWED_MODEL_SIZES[@]}"
    helpFunction
fi

TOTAL_NUM_GPUS=$((${OUMI_NUM_NODES} * ${POLARIS_NUM_GPUS_PER_NODE}))
# https://github.com/huggingface/tokenizers/issues/899#issuecomment-1027739758
export TOKENIZERS_PARALLELISM=false

# Our config is set to train for one epoch. Each section lists the number of steps
# that it should be equivalent to.
# For shorter debugging runs, set `training.max_steps`.
echo "${LOG_PREFIX} Starting training..."
# "2083804.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov" -> "2083804"
JOBNUM=$(echo $PBS_JOBID | cut -d'.' -f1)
if [ "$MODEL_SIZE" == "8b" ]; then
    # Copy the model to our Polaris machine to avoiding downloading from HF.
    rsync -av \
        /eagle/community_ai/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/ \
        ~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct
    if [ "$TRAINING_MODE" == "lora" ]; then
        # Num nodes: 1
        # Batch size per GPU: 2
        # Gradient accumulation steps (GAS): 32
        # Examples per step: 1 node * 4 GPUs/node * 2 bs * 32 GAS  = 256
        # Num steps for 1 epoch: 51,760 / 256 = 203
        set -x # Print "torchrun" command with expanded variables
        # DDP training with torchrun
        torchrun \
            --nnodes=${OUMI_NUM_NODES} \
            --node-rank=${POLARIS_NODE_RANK} \
            --nproc-per-node=${POLARIS_NUM_GPUS_PER_NODE} \
            --master-addr=${OUMI_MASTER_ADDR} \
            --master-port=8007 \
            -m oumi.train \
            -c configs/oumi/llama8b.lora.yaml \
            "training.run_name='polaris.llama8b.lora.${PBS_JOBID}'" \
            "training.output_dir=/eagle/community_ai/${USER}/runs/llama8b.lora.${JOBNUM}"
    else # SFT
        # Num nodes: 1
        # Batch size per GPU: 2
        # Gradient accumulation steps (GAS): 1
        # Examples per step: 1 node * 4 GPUs/node * 2 bs * 1 GAS  = 8
        # Num steps for 1 epoch: 51,760 / 8 = 6,470
        set -x # Print "accelerate" command with expanded variables
        accelerate launch \
            --num_machines ${OUMI_NUM_NODES} \
            --machine_rank ${POLARIS_NODE_RANK} \
            --num_processes ${TOTAL_NUM_GPUS} \
            --main_process_ip ${OUMI_MASTER_ADDR} \
            --main_process_port 8007 \
            --use_fsdp \
            --config_file configs/accelerate/llama8b.fsdp.yaml \
            -m oumi.train \
            -c configs/oumi/llama8b.sft.yaml \
            "training.run_name='polaris.llama8b.sft.${PBS_JOBID}'" \
            "training.output_dir=/eagle/community_ai/${USER}/runs/llama8b.sft.${JOBNUM}"
    fi
else # 70B
    # Copy the model to our Polaris machine to avoid downloading from HF.
    rsync -av \
        /eagle/community_ai/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-70B-Instruct/ \
        ~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-70B-Instruct
    if [ "$TRAINING_MODE" == "lora" ]; then
        # Num nodes: 2
        # Batch size per GPU: 2
        # Gradient accumulation steps (GAS): 1
        # Examples per step: 2 nodes * 4 GPUs/node * 2 bs * 1 GAS  = 16
        # Num steps for 1 epoch: 51,760 / 16 = 3,235
        set -x # Print "accelerate" command with expanded variables
        accelerate launch \
            --num_machines ${OUMI_NUM_NODES} \
            --machine_rank ${POLARIS_NODE_RANK} \
            --num_processes ${TOTAL_NUM_GPUS} \
            --main_process_ip ${OUMI_MASTER_ADDR} \
            --main_process_port 8007 \
            --use_fsdp \
            --config_file configs/accelerate/llama70b.lora.yaml \
            -m oumi.train \
            -c configs/oumi/llama70b.lora.yaml \
            "training.run_name='polaris.llama70b.lora.${PBS_JOBID}'" \
            "training.output_dir=/eagle/community_ai/${USER}/runs/llama70b.lora.${JOBNUM}"
    else # SFT
        # Num nodes: 4
        # Batch size per GPU: 2
        # Gradient accumulation steps (GAS): 1
        # Examples per step: 4 nodes * 4 GPUs/node * 2 bs * 1 GAS  = 32
        # Num steps for 1 epoch: 51,760 / 32 = 1,618
        set -x # Print "accelerate" command with expanded variables
        accelerate launch \
            --num_machines ${OUMI_NUM_NODES} \
            --machine_rank ${POLARIS_NODE_RANK} \
            --num_processes ${TOTAL_NUM_GPUS} \
            --main_process_ip ${OUMI_MASTER_ADDR} \
            --main_process_port 8007 \
            --use_fsdp \
            --config_file configs/accelerate/llama70b.fsdp.yaml \
            -m oumi.train \
            -c configs/oumi/llama70b.sft.yaml \
            "training.run_name='polaris.llama70b.sft.${PBS_JOBID}'" \
            "training.output_dir=/eagle/community_ai/${USER}/runs/llama70b.sft.${JOBNUM}"
    fi
fi

echo "${LOG_PREFIX} All done!"
