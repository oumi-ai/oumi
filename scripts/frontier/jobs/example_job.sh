#!/bin/bash

#SBATCH -N 1
#SBATCH -A lrn081
#SBATCH -J example_job
#SBATCH -o /lustre/orion/lrn081/scratch/$USER/jobs/logs/example_job-%j.OU
#SBATCH -e /lustre/orion/lrn081/scratch/$USER/jobs/logs/example_job-%j.ER
#SBATCH -t 01:00:00
#SBATCH -p batch


set -e

# Various setup for running on Polaris.
source "${SLURM_SUBMIT_DIR}/scripts/frontier/frontier_init.sh"

HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download --repo-type=dataset "yahma/alpaca-cleaned"


oumi distributed torchrun \
  -m oumi train \
  -c configs/recipes/deepseek_r1/sft/distill_qwen_1_5b/full_train.yaml \
  --training.run_name="deepseek-r1.qwen1.5b.fft.${SLURM_JOBID}" \
  --training.max_steps=10
