#!/bin/bash

# Arguments need to be provided in dotfile format
python -m oumi.train \
    "model.model_name=microsoft/Phi-3-mini-4k-instruct" \
    "data.datasets=[{dataset_name: yahma/alpaca-cleaned, preprocessing_function_name: alpaca}]" \
    "data.target_col=prompt" \
    "training.output_dir=train/" \
    "model.trust_remote_code=true"
