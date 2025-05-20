#!/bin/bash

#SBATCH -N 1
#SBATCH -A lrn081
#SBATCH -J example_job
#SBATCH -o /lustre/orion/lrn081/scratch/$USER/jobs/logs/example_job-%j.OU
#SBATCH -e /lustre/orion/lrn081/scratch/$USER/jobs/logs/example_job-%j.ER
#SBATCH -t 01:00:00
#SBATCH -p batch

FRONTIER_NODE_RANK=${PMI_RANK:=0}

set -e

# Various setup for running on Polaris.
source "${SLURM_SUBMIT_DIR}/scripts/frontier/frontier_init.sh"

LOG_PREFIX="Node: ${FRONTIER_NODE_RANK}:"
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
echo "${LOG_PREFIX} ROCM info: $(rocm-smi)"
echo "${LOG_PREFIX} TMPDIR: ${TMPDIR}"
echo "${LOG_PREFIX} CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "${LOG_PREFIX} ROCR_VISIBLE_DEVICES: ${ROCR_VISIBLE_DEVICES}"
echo "${LOG_PREFIX} ***ENV END***"

echo "Using this Python environment: $(which python3)"

HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download --repo-type=dataset "yahma/alpaca-cleaned"


oumi distributed torchrun \
  -m oumi train \
  -c configs/recipes/deepseek_r1/sft/distill_qwen_1_5b/full_train.yaml \
  --training.run_name="deepseek-r1.qwen1.5b.fft.${SLURM_JOBID}" \
  --training.max_steps=5
