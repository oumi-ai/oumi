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

ALLOWED_TRAINING_MODES=("ddp" "fsdp", "deepspeed")

helpFunction()
{
   echo ""
   echo "Usage: $0 -m (ddp|fsdp|deepspeed)"
   echo -e "\t-m The training mode: ${ALLOWED_TRAINING_MODES[@]}."
   exit 1 # Exit script after printing help
}

# Default values.
TRAINING_MODE="fsdp"

ENABLE_PYTORCH_PROFILER="false"

while getopts "m:p" opt
do
   case "$opt" in
      m ) TRAINING_MODE="$OPTARG" ;;
      p ) ENABLE_PYTORCH_PROFILER="true" ;;
      ? ) helpFunction ;; # Print a help message for an unknown parameter.
   esac
done

if [ -z "$TRAINING_MODE" ]; then
   echo "Empty training mode: ${TRAINING_MODE}";
   helpFunction
fi

if ! (echo "${ALLOWED_TRAINING_MODES[@]}" | grep -q -w "${TRAINING_MODE}"); then
    echo "Unknown training mode: ${TRAINING_MODE}. Valid values: ${ALLOWED_TRAINING_MODES[@]}"
    helpFunction
fi

MAX_STEPS=20
if "${ENABLE_PYTORCH_PROFILER}"; then
   # Use a smaller number of steps with Profiler to keep traces usable.
   MAX_STEPS=4
   PROFILER_TRAINING_PARAMS="training.output_dir=/eagle/community_ai/${USER}/${PBS_JOBID}
   training.profiler.enable_cpu_profiling=true
   training.profiler.enable_cuda_profiling=true"
   echo "PyTorch profiler enabled!"
fi

# Local copy of "HuggingFaceFW/fineweb-edu" dataset stored on Polaris.
TRAIN_DATASETS="data.train.datasets=
- dataset_name: \"/eagle/community_ai/datasets/fineweb-edu/sample-10BT\"
  subset: \"default\"
  split: \"train\"
"

# Training params shared between the different training modes, and likely
# don't need to be modified during experimentation.
SHARED_TRAINING_PARAMS="data.train.experimental_use_async_dataset=true
training.max_steps=${MAX_STEPS}
training.save_steps=0
training.save_final_model=false
training.dataloader_main_process_only=false
training.dataloader_num_workers=8
training.dataloader_prefetch_factor=32
training.log_model_summary=false
training.include_performance_metrics=true
training.ddp_find_unused_parameters=false
training.try_resume_from_last_checkpoint=false
${PROFILER_TRAINING_PARAMS}
training.enable_wandb=true"

echo "${LOG_PREFIX} Starting training (${TRAINING_MODE})..."
if [ "$TRAINING_MODE" == "ddp" ]; then
    set -x  # Print "torchrun" command with expanded variables
    torchrun \
        --nnodes=${LEMA_NUM_NODES} \
        --node-rank=${POLARIS_NODE_RANK} \
        --nproc-per-node=${POLARIS_GPUS_PER_NODE} \
        --master-addr=${LEMA_MASTER_ADDR} \
        --master-port=8007 \
        -m lema.train \
        -c configs/lema/llama2b.pt.yaml \
        "$TRAIN_DATASETS" \
        $SHARED_TRAINING_PARAMS \
        "training.run_name='polaris.llama2b.ddp.${PBS_JOBID}'" \
        "training.optimizer='adafactor'" \
        "training.per_device_train_batch_size=4" \
        "training.gradient_accumulation_steps=64"
elif [ "$TRAINING_MODE" == "deepspeed" ]; then
    set -x  # Print "accelerate" command with expanded variables
    accelerate launch \
      --num_machines ${LEMA_NUM_NODES} \
      --machine_rank ${POLARIS_NODE_RANK} \
      --num_processes $((${LEMA_NUM_NODES} * ${POLARIS_GPUS_PER_NODE})) \
      --main_process_ip ${LEMA_MASTER_ADDR} \
      --main_process_port 8007 \
      --use_deepspeed \
      --config_file configs/accelerate/llama.deepspeed.yaml \
      -m lema.train \
      -c configs/lema/llama2b.pt.yaml \
      "$TRAIN_DATASETS" \
      $SHARED_TRAINING_PARAMS \
      "training.run_name='polaris.llama2b.deepspeed.${PBS_JOBID}'" \
      "training.optimizer='adafactor'" \
      "training.enable_gradient_checkpointing=false" \
      "training.per_device_train_batch_size=4" \
      "training.gradient_accumulation_steps=64" \
      "training.mixed_precision_dtype=BF16"
else
    set -x  # Print "accelerate" command with expanded variables
    accelerate launch \
      --num_machines ${LEMA_NUM_NODES} \
      --machine_rank ${POLARIS_NODE_RANK} \
      --num_processes $((${LEMA_NUM_NODES} * ${POLARIS_GPUS_PER_NODE})) \
      --main_process_ip ${LEMA_MASTER_ADDR} \
      --main_process_port 8007 \
      --use_fsdp \
      --config_file configs/accelerate/llama.fsdp.yaml \
      -m lema.train \
      -c configs/lema/llama2b.pt.yaml \
      "$TRAIN_DATASETS" \
      $SHARED_TRAINING_PARAMS \
      "training.run_name='polaris.llama2b.fsdp.${PBS_JOBID}'" \
      "training.optimizer='adafactor'" \
      "training.per_device_train_batch_size=14" \
      "training.gradient_accumulation_steps=19"
fi

echo "${LOG_PREFIX} All done!"
