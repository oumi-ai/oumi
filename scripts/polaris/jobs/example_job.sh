#!/bin/bash

#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:10:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -A community_ai
#PBS -o /eagle/community_ai/jobs/logs/
#PBS -e /eagle/community_ai/jobs/logs/

set -e

# Various setup for running on Polaris.
source ${PBS_O_WORKDIR}/scripts/polaris/polaris_init.sh

TRAIN_DATASETS="--data.train.datasets=
- dataset_name: \"/eagle/community_ai/datasets/fineweb-edu/sample-10BT\"
  subset: \"default\"
  split: \"train\"
"

# Each batch should be 512 examples. With 4 GPUS and batch size 32 per GPU, we need
# 4 gradient accumulation steps.
oumi distributed torchrun \
  -m oumi train \
  -c configs/recipes/gpt2/pretraining/train.yaml \
  --training.run_name "gpt2.pt.${PBS_JOBID}" \
  "$TRAIN_DATASETS" \
  --training.max_steps 100 \
  --training.include_performance_metrics true \
  --training.ddp_find_unused_parameters false \
  --training.dataloader_num_workers 2 \
  --training.dataloader_prefetch_factor 4 \
  --training.per_device_train_batch_size 32 \
  --training.gradient_accumulation_steps 4
