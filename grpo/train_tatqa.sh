#!/bin/bash
export OUMI_EXTRA_DEPS_FILE=/data/shanghong/oumi/grpo/requirements.txt

oumi train -c /data/shanghong/oumi/grpo/configs/tatqa_grpo_train.yaml 2>&1 | tee logs/grpo_tatqa_qwen2.5-1.5b.log
