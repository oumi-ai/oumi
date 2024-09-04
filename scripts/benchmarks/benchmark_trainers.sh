#!/bin/bash
set -xe

# Script to benchmark different trainers and model configurations
# and compare their performance.

# HuggingFace model with huggingface trainer
time accelerate launch --num_machines 1 --machine_rank 0 --num_processes 1 --gpu_ids 0 --dynamo_backend inductor --mixed_precision no \
    -m lema.train \
    -c "configs/lema/llama2b.pt.yaml" \
    "training.trainer_type=TRL_SFT" \
    "training.per_device_train_batch_size=3" \
    "training.gradient_accumulation_steps=1" \
    "training.output_dir=output/trainer-trk/" \
    "training.include_performance_metrics=true" \
    "training.dep_log_level=debug" \
    "training.logging_steps=5" \
    "training.max_steps=1000" \
    "training.save_steps=0" \
    "training.compile=true" \
    "training.save_final_model=false"

# HuggingFace model with lema trainer
# time CUDA_VISIBLE_DEVICES="0" python \  # For single GPU, can also be ran directly
time torchrun --standalone --nproc_per_node 1 \
    -m lema.train \
    -c "configs/lema/llama2b.pt.yaml" \
    "training.trainer_type=LEMA" \
    "training.per_device_train_batch_size=3" \
    "training.gradient_accumulation_steps=1" \
    "training.output_dir=output/trainer-lema/" \
    "training.include_performance_metrics=true" \
    "training.dep_log_level=debug" \
    "training.logging_steps=5" \
    "training.max_steps=1000" \
    "training.save_steps=0" \
    "training.compile=true" \
    "training.save_final_model=false" \
    "training.dataloader_prefetch_factor=2" \
    "training.dataloader_num_workers=1"
