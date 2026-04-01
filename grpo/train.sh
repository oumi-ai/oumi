export OUMI_EXTRA_DEPS_FILE=/data/shanghong/oumi/grpo/requirements.txt

oumi train -c configs/grpo.yaml | 2>&1 tee logs/grpo_countdown_qwen2.5-1.5b-instruct.log
wait
