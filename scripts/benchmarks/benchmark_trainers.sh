#!/bin/bash
set -xe

# Script to benchmark different trainers and model configurations
# and compare their performance.

# HuggingFace model with huggingface trainer
time accelerate launch \
    --config_file configs/accelerate/llama8b.fsdp.yaml \
    --use_fsdp \
    --num_processes 8 \
    --dynamo_backend inductor \
    --mixed_precision no \
    -m oumi.train \
    -c "configs/oumi/llama8b_longcontext.sft.yaml"

# # HuggingFace model with Oumi trainer
# # time CUDA_VISIBLE_DEVICES="0" python \  # For single GPU, can also be ran directly
# time torchrun --standalone --nproc_per_node 1 \
#     -m oumi.train \
#     -c "configs/oumi/llama2b.pt.yaml" \
#     "training.trainer_type=OUMI" \
#     "training.per_device_train_batch_size=4" \
#     "training.gradient_accumulation_steps=1" \
#     "training.output_dir=output/trainer-oumi/" \
#     "training.include_performance_metrics=true" \
#     "training.dep_log_level=debug" \
#     "training.logging_steps=5" \
#     "training.max_steps=1000" \
#     "training.save_steps=0" \
#     "training.compile=true" \
#     "training.save_final_model=false" \
#     "training.dataloader_prefetch_factor=2" \
#     "training.dataloader_num_workers=1"
