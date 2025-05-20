#!/bin/bash

#SBATCH -N 1
#SBATCH -A lrn081
#SBATCH -J example_job
#SBATCH -o /lustre/orion/lrn081/scratch/$USER/jobs/logs/example_job-%j.OU
#SBATCH -e /lustre/orion/lrn081/scratch/$USER/jobs/logs/example_job-%j.ER
#SBATCH -t 00:10:00
#SBATCH -p batch


set -e

# Various setup for running on Polaris.
source "${SLURM_SUBMIT_DIR}/scripts/frontier/frontier_init.sh"

TRAIN_DATASETS="--data.train.datasets=
- dataset_name: \"/eagle/community_ai/datasets/fineweb-edu/sample-10BT\"
  subset: \"default\"
  split: \"train\"
"

HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download HuggingFaceFW/ablation-model-fineweb-v1

# Each batch should be 512 examples. With 4 GPUS and batch size 32 per GPU, we need
# 4 gradient accumulation steps.
# oumi distributed torchrun \
#   -m oumi train \
#   -c configs/recipes/gpt2/pretraining/train.yaml \
#   --training.run_name "gpt2.pt.${PBS_JOBID}" \
#   "$TRAIN_DATASETS" \
#   --training.max_steps 100 \
#   --training.include_performance_metrics true \
#   --training.ddp_find_unused_parameters false \
#   --training.dataloader_num_workers 2 \
#   --training.dataloader_prefetch_factor 4 \
#   --training.per_device_train_batch_size 32 \
#   --training.gradient_accumulation_steps 4
