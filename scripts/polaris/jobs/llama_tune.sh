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

ALLOWED_TRAINING_MODES=("sft", "lora", "qlora")
ALLOWED_MODEL_SIZES=("3b", "8b", "70b")

helpFunction() {
    echo ""
    echo "Usage: $0 -m (sft/lora/qlora) -s (3b/8b/70b)"
    echo -e "\t-m The training mode: ${ALLOWED_TRAINING_MODES[@]}. Defaults to lora."
    echo -e "\t-s The model size: ${ALLOWED_MODEL_SIZES[@]}. Defaults to 8b."
    exit 1 # Exit script after printing help
}

# Default value.
TRAINING_MODE="lora"
MODEL_SIZE="8b"
ENABLE_OUMI_TELEMETRY="false"

# Get values from command line and verify.
while getopts ":m:s:t" opt; do
    case "$opt" in
    m) TRAINING_MODE="$OPTARG" ;;
    s) MODEL_SIZE="$OPTARG" ;;
    t) ENABLE_OUMI_TELEMETRY="true" ;;
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

if "${ENABLE_OUMI_TELEMETRY}"; then
    OUMI_TELEMETRY_PARAMS="training.telemetry.collect_telemetry_for_all_ranks=true
    training.telemetry.track_gpu_temperature=true"
    echo "Oumi telemetry enabled!"
fi

TOTAL_NUM_GPUS=$((${OUMI_NUM_NODES} * ${POLARIS_NUM_GPUS_PER_NODE}))
# https://github.com/huggingface/tokenizers/issues/899#issuecomment-1027739758
export TOKENIZERS_PARALLELISM=false


# Training params shared between the different training modes, and likely
# don't need to be modified during experimentation.
SHARED_TRAINING_PARAMS="training.run_name='polaris.llama${MODEL_SIZE}.${TRAINING_MODE}.${OUMI_JOBNUM}'
training.output_dir=/eagle/community_ai/${USER}/runs/llama${MODEL_SIZE}.${TRAINING_MODE}.${OUMI_JOBNUM}
${OUMI_TELEMETRY_PARAMS}"

# For shorter debugging runs, set `training.max_steps`.
echo "${LOG_PREFIX} Starting training..."
if [ "$MODEL_SIZE" == "3b" ]; then
    if [ "$TRAINING_MODE" == "lora" ]; then
        set -x # Print "torchrun" command with expanded variables
        torchrun \
            --nnodes=${OUMI_NUM_NODES} \
            --node-rank=${POLARIS_NODE_RANK} \
            --nproc-per-node=${POLARIS_NUM_GPUS_PER_NODE} \
            --master-addr=${OUMI_MASTER_ADDR} \
            --master-port=8007 \
            -m oumi.train \
            -c configs/oumi/llama3b.lora.yaml \
            $SHARED_TRAINING_PARAMS
    elif [ "$TRAINING_MODE" == "qlora" ]; then
        set -x # Print "torchrun" command with expanded variables
        torchrun \
            --nnodes=${OUMI_NUM_NODES} \
            --node-rank=${POLARIS_NODE_RANK} \
            --nproc-per-node=${POLARIS_NUM_GPUS_PER_NODE} \
            --master-addr=${OUMI_MASTER_ADDR} \
            --master-port=8007 \
            -m oumi.train \
            -c configs/oumi/llama3b.qlora.yaml \
            $SHARED_TRAINING_PARAMS
    else # SFT
        set -x # Print "torchrun" command with expanded variables
        torchrun \
            --nnodes=${OUMI_NUM_NODES} \
            --node-rank=${POLARIS_NODE_RANK} \
            --nproc-per-node=${POLARIS_NUM_GPUS_PER_NODE} \
            --master-addr=${OUMI_MASTER_ADDR} \
            --master-port=8007 \
            -m oumi.train \
            -c configs/oumi/llama3b.sft.yaml \
            $SHARED_TRAINING_PARAMS
    fi
elif [ "$MODEL_SIZE" == "8b" ]; then
    # Copy the model to our Polaris machine to avoiding downloading from HF.
    rsync -av \
        /eagle/community_ai/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/ \
        ~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct
    if [ "$TRAINING_MODE" == "lora" ]; then
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
            $SHARED_TRAINING_PARAMS
    elif [ "$TRAINING_MODE" == "qlora" ]; then
        echo "Llama 8B QLora is currently not supported!"
    else # SFT
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
            $SHARED_TRAINING_PARAMS
    fi
else # 70B
    # Copy the model to our Polaris machine to avoid downloading from HF.
    rsync -av \
        /eagle/community_ai/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-70B-Instruct/ \
        ~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-70B-Instruct
    if [ "$TRAINING_MODE" == "lora" ]; then
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
            $SHARED_TRAINING_PARAMS
    elif [ "$TRAINING_MODE" == "qlora" ]; then
        echo "Llama 70B QLora is currently not supported!"
    else # SFT
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
            $SHARED_TRAINING_PARAMS
    fi
fi

echo "${LOG_PREFIX} All done!"
