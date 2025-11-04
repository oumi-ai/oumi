#!/bin/bash

# Mlflow environment variables
export DATABRICKS_HOST="xx"
export DATABRICKS_TOKEN="xx"
export MLFLOW_TRACKING_URI="databricks"
export MLFLOW_EXPERIMENT_ID="2151893598744976"

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export NCCL_ASYNC_ERROR_HANDLING=1
# export CUDA_LAUNCH_BLOCKING=1


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master-port=9007 -m oumi train -c "/home/shanghong/oumi/tmp/enterprise_experiments/configs/qwen3_32b_tatqa_lora_e4_think.yaml" 2>&1 | tee "/home/shanghong/oumi/tmp/enterprise_experiments/logs/qwen3_32b_tatqa_lora_e4_think.log" &
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master-port=9006 -m oumi train -c "/home/shanghong/oumi/tmp/enterprise_experiments/configs/qwen3_32b_tatqa_lora_e4_think2.yaml" 2>&1 | tee "/home/shanghong/oumi/tmp/enterprise_experiments/logs/qwen3_32b_tatqa_lora_e4_think2.log"

wait

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master-port=9007 -m oumi train -c "/home/shanghong/oumi/tmp/enterprise_experiments/configs/qwen3_32b_tatqa_lora_e5_think.yaml" 2>&1 | tee "/home/shanghong/oumi/tmp/enterprise_experiments/logs/qwen3_32b_tatqa_lora_e5_think.log" &
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master-port=9006 -m oumi train -c "/home/shanghong/oumi/tmp/enterprise_experiments/configs/qwen3_32b_tatqa_lora_e5_think2.yaml" 2>&1 | tee "/home/shanghong/oumi/tmp/enterprise_experiments/logs/qwen3_32b_tatqa_lora_e5_think2.log"

wait