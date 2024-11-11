#!/bin/bash

POLARIS_NODE_RANK=${PMI_RANK:=0}

# Reversing GPUs order to match Polaris CPU affinities:
# https://docs.alcf.anl.gov/polaris/hardware-overview/machine-overview/#polaris-device-affinity-information
export CUDA_VISIBLE_DEVICES=3,2,1,0
LOG_PREFIX="Node: ${POLARIS_NODE_RANK}:"

echo "${LOG_PREFIX} ***ENV BEGIN***"
echo "${LOG_PREFIX} PBS_JOBID: $PBS_JOBID"
echo "${LOG_PREFIX} OUMI_JOBNUM: $OUMI_JOBNUM"
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

ALLOWED_TRAINING_MODES=("fft", "lora", "qlora", "pretrain")
ALLOWED_DISTRIBUTION_MODES=("ddp", "fsdp")
ALLOWED_MODEL_SIZES=("3b", "8b", "70b", "405b")

helpFunction() {
    echo ""
    echo "Usage: $0 -m (fft/lora/qlora/pretrain) -d (ddp/fsdp) -s (3b/8b/70b/405b)"
    echo -e "\t-m The training mode: ${ALLOWED_TRAINING_MODES[@]}. Defaults to lora."
    echo -e "\t-d The distribution mode: ${ALLOWED_DISTRIBUTION_MODES[@]}. Defaults to ddp."
    echo -e "\t-s The model size: ${ALLOWED_MODEL_SIZES[@]}. Defaults to 8b."
    exit 1 # Exit script after printing help
}

# Default values.
TRAINING_MODE="lora"
DISTRIBUTION_MODE="ddp"
MODEL_SIZE="8b"
ENABLE_OUMI_TELEMETRY="false"

# Get values from command line and verify.
while getopts ":m:d:s:t" opt; do
    case "$opt" in
    m) TRAINING_MODE="$OPTARG" ;;
    d) DISTRIBUTION_MODE="$OPTARG" ;;
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
if [ -z "$DISTRIBUTION_MODE" ]; then
    echo "Distribution mode can't be empty."
    helpFunction
fi
if ! (echo "${ALLOWED_DISTRIBUTION_MODES[@]}" | grep -q -w "${DISTRIBUTION_MODE}"); then
    echo "Unknown distribution mode: ${DISTRIBUTION_MODE}. Valid values: ${ALLOWED_DISTRIBUTION_MODES[@]}"
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

# https://github.com/huggingface/tokenizers/issues/899#issuecomment-1027739758
export TOKENIZERS_PARALLELISM=false


# Training params shared between the different training modes, and likely
# don't need to be modified during experimentation.
SHARED_TRAINING_PARAMS="training.run_name='polaris.llama${MODEL_SIZE}.${TRAINING_MODE}.${OUMI_JOBNUM}'
training.output_dir=/eagle/community_ai/${USER}/runs/llama${MODEL_SIZE}.${TRAINING_MODE}.${OUMI_JOBNUM}
${OUMI_TELEMETRY_PARAMS}"

if [ "$TRAINING_MODE" == "pretrain" ]; then
# Local copy of "HuggingFaceFW/fineweb-edu" dataset stored on Polaris.
PRETRAIN_DATASETS="data.train.datasets=
- dataset_name: '/eagle/community_ai/datasets/fineweb-edu/sample-10BT'
  subset: 'default'
  split: 'train'
"
fi

# For shorter debugging runs, set `training.max_steps`.
echo "${LOG_PREFIX} Starting training..."
if [ "$MODEL_SIZE" == "3b" ]; then
    if [ "$TRAINING_MODE" == "pretrain" ]; then
        echo "Llama 3B pretraining is currently not supported!"
    elif [ "$DISTRIBUTION_MODE" == "ddp" ]; then
        if [ "$TRAINING_MODE" == "lora" ]; then
            set -x
            torchrun \
                --nnodes=${OUMI_NUM_NODES} \
                --node-rank=${POLARIS_NODE_RANK} \
                --nproc-per-node=${OUMI_POLARIS_NUM_GPUS_PER_NODE} \
                --master-addr=${OUMI_MASTER_ADDR} \
                --master-port=8007 \
                -m oumi.train \
                -c configs/recipes/llama3_2/sft/3b_lora/train.yaml \
                $SHARED_TRAINING_PARAMS
        elif [ "$TRAINING_MODE" == "qlora" ]; then
            set -x
            torchrun \
                --nnodes=${OUMI_NUM_NODES} \
                --node-rank=${POLARIS_NODE_RANK} \
                --nproc-per-node=${OUMI_POLARIS_NUM_GPUS_PER_NODE} \
                --master-addr=${OUMI_MASTER_ADDR} \
                --master-port=8007 \
                -m oumi.train \
                -c configs/recipes/llama3_2/sft/3b_qlora/train.yaml \
                $SHARED_TRAINING_PARAMS
        else # FFT
            set -x
            torchrun \
                --nnodes=${OUMI_NUM_NODES} \
                --node-rank=${POLARIS_NODE_RANK} \
                --nproc-per-node=${OUMI_POLARIS_NUM_GPUS_PER_NODE} \
                --master-addr=${OUMI_MASTER_ADDR} \
                --master-port=8007 \
                -m oumi.train \
                -c configs/recipes/llama3_2/sft/3b_full/train.yaml \
                "model.model_max_length=512" \
                $SHARED_TRAINING_PARAMS
        fi
    else # FSDP
        echo "Llama 3B FSDP is currently not supported!"
    fi
elif [ "$MODEL_SIZE" == "8b" ]; then
    # Copy the model to our Polaris machine to avoiding downloading from HF.
    if [ "$TRAINING_MODE" == "pretrain" ]; then
        rsync -av \
            /eagle/community_ai/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/ \
            ~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B
    else
        rsync -av \
            /eagle/community_ai/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/ \
            ~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct
    fi

    if [ "$DISTRIBUTION_MODE" == "ddp" ]; then
        if [ "$TRAINING_MODE" == "lora" ]; then
            set -x
            # DDP training with torchrun
            torchrun \
                --nnodes=${OUMI_NUM_NODES} \
                --node-rank=${POLARIS_NODE_RANK} \
                --nproc-per-node=${OUMI_POLARIS_NUM_GPUS_PER_NODE} \
                --master-addr=${OUMI_MASTER_ADDR} \
                --master-port=8007 \
                -m oumi.train \
                -c configs/recipes/llama3_1/sft/8b_lora/train.yaml \
                $SHARED_TRAINING_PARAMS
        elif [ "$TRAINING_MODE" == "qlora" ]; then
            echo "Llama 8B QLora DDP is currently not supported!"
        else # FFT
            echo "Llama 8B FFT DDP is currently not supported!"
        fi
    else # FSDP
        if [ "$TRAINING_MODE" == "lora" ]; then
            echo "Llama 8B Lora FSDP is currently not supported!"
        elif [ "$TRAINING_MODE" == "qlora" ]; then
            echo "Llama 8B QLora FSDP is currently not supported!"
        else # FFT
            OUMI_CFG_FILE="configs/recipes/llama3_1/sft/8b_full/train.yaml"
            if [ "$TRAINING_MODE" == "pretrain" ]; then
                OUMI_CFG_FILE="configs/recipes/llama3_1/pretraining/8b/train.yaml"
            fi

            set -x
            torchrun \
                --nnodes=${OUMI_NUM_NODES} \
                --node-rank=${POLARIS_NODE_RANK} \
                --nproc-per-node=${OUMI_POLARIS_NUM_GPUS_PER_NODE} \
                --master-addr=${OUMI_MASTER_ADDR} \
                --master-port=8007 \
                -m oumi.train \
                -c "${OUMI_CFG_FILE}" \
                "$PRETRAIN_DATASETS" \
                $SHARED_TRAINING_PARAMS
        fi
    fi
elif [ "$MODEL_SIZE" == "70b" ]; then
    # Copy the model to our Polaris machine to avoid downloading from HF.
    if [ "$TRAINING_MODE" == "pretrain" ]; then
        rsync -av \
            /eagle/community_ai/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-70B/ \
            ~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-70B
    else
        rsync -av \
            /eagle/community_ai/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-70B-Instruct/ \
            ~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-70B-Instruct
    fi

    if [ "$TRAINING_MODE" == "pretrain" ]; then
        echo "Llama 70B pretraining is currently not supported!"
    elif [ "$DISTRIBUTION_MODE" == "ddp" ]; then
        echo "Llama 70B DDP is not possible!"
    else # FSDP
        if [ "$TRAINING_MODE" == "lora" ]; then
            set -x
            torchrun \
                --nnodes=${OUMI_NUM_NODES} \
                --node-rank=${POLARIS_NODE_RANK} \
                --nproc-per-node=${OUMI_POLARIS_NUM_GPUS_PER_NODE} \
                --master-addr=${OUMI_MASTER_ADDR} \
                --master-port=8007 \
                -m oumi.train \
                -c configs/recipes/llama3_1/sft/70b_lora/train.yaml \
                $SHARED_TRAINING_PARAMS
        elif [ "$TRAINING_MODE" == "qlora" ]; then
            set -x
            torchrun \
                --nnodes=${OUMI_NUM_NODES} \
                --node-rank=${POLARIS_NODE_RANK} \
                --nproc-per-node=${OUMI_POLARIS_NUM_GPUS_PER_NODE} \
                --master-addr=${OUMI_MASTER_ADDR} \
                --master-port=8007 \
                -m oumi.train \
                -c configs/recipes/llama3_1/sft/70b_qlora/train.yaml \
                $SHARED_TRAINING_PARAMS
        else # FFT
            set -x
            torchrun \
                --nnodes=${OUMI_NUM_NODES} \
                --node-rank=${POLARIS_NODE_RANK} \
                --nproc-per-node=${OUMI_POLARIS_NUM_GPUS_PER_NODE} \
                --master-addr=${OUMI_MASTER_ADDR} \
                --master-port=8007 \
                -m oumi.train \
                -c configs/recipes/llama3_1/sft/70b_full/train.yaml \
                $SHARED_TRAINING_PARAMS
        fi
    fi
else # 405B
    # 405B can't fit in the user's `/home` directory, so use Eagle as our cache.`
    mkdir -p /tmp/hf_cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-405B-Instruct/snapshots/be673f326cab4cd22ccfef76109faf68e41aa5f1/
    cp /eagle/community_ai/hf_cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-405B-Instruct/snapshots/be673f326cab4cd22ccfef76109faf68e41aa5f1/* /tmp/hf_cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-405B-Instruct/snapshots/be673f326cab4cd22ccfef76109faf68e41aa5f1/
    cp /home/wizeng/.cache/huggingface/token /tmp/hf_cache/huggingface/token
    export HF_HOME="/tmp/hf_cache/huggingface"
    # export HF_HOME="/eagle/community_ai/hf_cache/huggingface"
    if [ "$TRAINING_MODE" == "pretrain" ]; then
        echo "Llama 405B pretraining is currently not supported!"
    elif [ "$DISTRIBUTION_MODE" == "ddp" ]; then
        echo "Llama 405B DDP is not possible!"
    else # FSDP
        if [ "$TRAINING_MODE" == "lora" ]; then
            set -x
            torchrun \
                --nnodes=${OUMI_NUM_NODES} \
                --node-rank=${POLARIS_NODE_RANK} \
                --nproc-per-node=${OUMI_POLARIS_NUM_GPUS_PER_NODE} \
                --master-addr=${OUMI_MASTER_ADDR} \
                --master-port=8007 \
                -m oumi.train \
                -c configs/recipes/llama3_1/sft/405b_lora/train.yaml \
                $SHARED_TRAINING_PARAMS
        elif [ "$TRAINING_MODE" == "qlora" ]; then
            set -x
            torchrun \
                --nnodes=${OUMI_NUM_NODES} \
                --node-rank=${POLARIS_NODE_RANK} \
                --nproc-per-node=${OUMI_POLARIS_NUM_GPUS_PER_NODE} \
                --master-addr=${OUMI_MASTER_ADDR} \
                --master-port=8007 \
                -m oumi.train \
                -c configs/recipes/llama3_1/sft/405b_qlora/train.yaml \
                $SHARED_TRAINING_PARAMS
        else # FFT
            set -x
            torchrun \
                --nnodes=${OUMI_NUM_NODES} \
                --node-rank=${POLARIS_NODE_RANK} \
                --nproc-per-node=${OUMI_POLARIS_NUM_GPUS_PER_NODE} \
                --master-addr=${OUMI_MASTER_ADDR} \
                --master-port=8007 \
                -m oumi.train \
                -c configs/recipes/llama3_1/sft/405b_full/train.yaml \
                $SHARED_TRAINING_PARAMS
        fi
    fi
fi

echo "${LOG_PREFIX} All done!"
