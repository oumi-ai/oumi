#!/bin/bash

python -m lema.train \
    --model_name "microsoft/Phi-3-mini-4k-instruct" \
    --dataset_name "yahma/alpaca-cleaned" \
    --output_dir tmp/ \
    --trust_remote_code true \
    --preprocessing_function_name alpaca
