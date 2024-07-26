#!/bin/bash
# Script to benchmark different trainers and model configurations
# and compare their performance.

set -xe

# HuggingFace model with huggingface trainer
python -m lema.train -c "configs/lema/gpt2.pt.yaml" \
    "model.model_name=gpt2" \
    "model.tokenizer_name=gpt2" \
    "training.per_device_train_batch_size=16" \
    "training.trainer_type=TRL_SFT" \
    "training.output_dir=output/model-hf_trainer-hf/" \
    "training.include_performance_metrics=True" \
    "training.max_steps=100"

# HuggingFace model with lema trainer
python -m lema.train -c "configs/lema/gpt2.pt.yaml" \
    "model.model_name=gpt2" \
    "model.tokenizer_name=gpt2" \
    "training.per_device_train_batch_size=10" \
    "training.trainer_type=LEMA" \
    "training.output_dir=output/model-hf_trainer-lema/" \
    "training.include_performance_metrics=True" \
    "training.max_steps=100"

# Lema model with lema trainer
python -m lema.train -c "configs/lema/gpt2.pt.yaml" \
    "model.model_name=NanoGPT2" \
    "model.tokenizer_name=gpt2" \
    "training.per_device_train_batch_size=12" \
    "training.trainer_type=LEMA" \
    "training.output_dir=output/model-lema_trainer-lema/" \
    "training.include_performance_metrics=True" \
    "training.max_steps=100"

# Lema model with huggingface trainer
# TODO: fix issue with DDP in this config
CUDA_VISIBLE_DEVICES="0" python -m lema.train -c "configs/lema/gpt2.pt.yaml" \
    "model.model_name=NanoGPT2" \
    "model.tokenizer_name=gpt2" \
    "training.per_device_train_batch_size=16" \
    "training.trainer_type=TRL_SFT" \
    "training.output_dir=output/model-lema_trainer-hf/" \
    "training.include_performance_metrics=True" \
    "training.max_steps=100"
