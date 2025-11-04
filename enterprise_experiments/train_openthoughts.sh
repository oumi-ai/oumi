#!/bin/bash

# Mlflow environment variables
export DATABRICKS_HOST="xx"
export DATABRICKS_TOKEN="xx"
export MLFLOW_TRACKING_URI="databricks"
export MLFLOW_EXPERIMENT_ID="2151893598744976"


CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master-port=9007 -m oumi train -c "/home/shanghong/oumi/tmp/enterprise_experiments/configs/qwen2.5_7b_openthoughts_small_1k_original.yaml" 2>&1 | tee "/home/shanghong/oumi/tmp/enterprise_experiments/logs/qwen2.5_7b_openthoughts_small_1k_original.log" &
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master-port=9006 -m oumi train -c "/home/shanghong/oumi/tmp/enterprise_experiments/configs/qwen2.5_7b_openthoughts_small_1k.yaml" 2>&1 | tee "/home/shanghong/oumi/tmp/enterprise_experiments/logs/qwen2.5_7b_openthoughts_small_1k.log"

wait
