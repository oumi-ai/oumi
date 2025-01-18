#!/bin/bash

POLARIS_NODE_RANK=${PMI_RANK:=0}
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

ALLOWED_TRAINING_MODES=("ddp", "ddp1gpu", "fsdp")

helpFunction() {
    echo ""
    echo "Usage: $0 -m (ddp|ddp1gpu|fsdp)"
    echo -e "\t-m The training mode: ${ALLOWED_TRAINING_MODES[@]}."
    exit 1 # Exit script after printing help
}

# Default values.
TRAINING_MODE="fsdp"

ENABLE_PYTORCH_PROFILER="false"
ENABLE_OUMI_TELEMETRY="false"

while getopts ":m:p:t" opt; do
    case "$opt" in
    m) TRAINING_MODE="$OPTARG" ;;
    p) ENABLE_PYTORCH_PROFILER="true" ;;
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

MAX_STEPS=20
if "${ENABLE_PYTORCH_PROFILER}"; then
    # Use a smaller number of steps with Profiler to keep traces usable.
    MAX_STEPS=6
    PROFILER_TRAINING_PARAMS="--training.profiler.schedule.enable_schedule true
    --training.profiler.schedule.skip_first 1
    --training.profiler.schedule.warmup 1
    --training.profiler.schedule.active 4
    --training.profiler.enable_cpu_profiling true
    --training.profiler.enable_cuda_profiling true"
    echo "PyTorch profiler enabled!"
fi

if "${ENABLE_OUMI_TELEMETRY}"; then
    OUMI_TELEMETRY_PARAMS="--training.telemetry.collect_telemetry_for_all_ranks true
    --training.telemetry.track_gpu_temperature true"
    echo "Oumi telemetry enabled!"
fi

if "${ENABLE_PYTORCH_PROFILER}" || "${ENABLE_OUMI_TELEMETRY}"; then
    TRAINING_OUTPUT_DIR_PARAM="--training.output_dir /eagle/community_ai/${USER}/${OUMI_JOBNUM}"
fi

# Local copy of "HuggingFaceFW/fineweb-edu" dataset stored on Polaris.
TRAIN_DATASETS="--data.train.datasets=
- dataset_name: '/eagle/community_ai/datasets/fineweb-edu/sample-10BT'
  subset: 'default'
  split: 'train'
"

# Training params shared between the different training modes, and likely
# don't need to be modified during experimentation.
SHARED_TRAINING_PARAMS="--training.max_steps ${MAX_STEPS}
--training.save_steps 0
--training.save_final_model false
--training.dataloader_main_process_only false
--training.dataloader_num_workers 8
--training.log_model_summary false
--training.run_name 'polaris.fineweb.${TRAINING_MODE}.${OUMI_JOBNUM}'
${TRAINING_OUTPUT_DIR_PARAM}
${PROFILER_TRAINING_PARAMS}
${OUMI_TELEMETRY_PARAMS}"

echo "${LOG_PREFIX} Starting training (${TRAINING_MODE})..."
if [ "$TRAINING_MODE" == "ddp" ]; then
    set -x
    oumi distributed torchrun \
        -m oumi train \
        -c configs/examples/fineweb_ablation_pretraining/ddp/train.yaml \
        $SHARED_TRAINING_PARAMS
elif [ "$TRAINING_MODE" == "ddp1gpu" ]; then
    export CUDA_VISIBLE_DEVICES=$((${OUMI_POLARIS_NUM_GPUS_PER_NODE} - 1 - ${PMI_LOCAL_RANK} % ${OUMI_POLARIS_NUM_GPUS_PER_NODE}))
    set -x
    echo "${LOG_PREFIX} CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    # Note the 1 process per node
    torchrun \
        --nnodes=${OUMI_TOTAL_NUM_GPUS} \
        --node-rank=${POLARIS_NODE_RANK} \
        --nproc-per-node=1 \
        --master-addr=${OUMI_MASTER_ADDR} \
        --master-port=8007 \
        -m oumi train \
        -c configs/examples/fineweb_ablation_pretraining/ddp/train.yaml \
        "$TRAIN_DATASETS" \
        $SHARED_TRAINING_PARAMS \
        --training.per_device_train_batch_size 4 \
        --training.gradient_accumulation_steps 64
else # FSDP
    set -x
    oumi distributed torchrun \
        -m oumi train \
        -c configs/examples/fineweb_ablation_pretraining/fsdp/train.yaml \
        "$TRAIN_DATASETS" \
        $SHARED_TRAINING_PARAMS
fi

echo "${LOG_PREFIX} All done!"
