#!/bin/bash

# Arguments need to be provided in dotfile format
python -m lema.train \
    "model.model_name=microsoft/Phi-3-mini-4k-instruct" \
    "data.dataset_name=yahma/alpaca-cleaned" \
    "data.preprocessing_function_name=alpaca" \
    "data.trainer_kwargs.dataset_text_field=prompt" \
    "training.output_dir=train/" \
    "model.trust_remote_code=true"
