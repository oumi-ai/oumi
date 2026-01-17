# oumi train -c configs/countdown_tatqa_lambda0.5.yaml | 2>&1 tee logs/train_countdown_tatqa_lambda0.5.log
# wait

# oumi train -c configs/countdown_cross.yaml | 2>&1 tee logs/train_countdown_cross.log
# wait

# oumi train -c configs/countdown_tatqa_lambda0.0.yaml | 2>&1 tee logs/train_countdown_tatqa_lambda0.0.log
# wait

oumi train -c configs/tatqa_lambda1.0.yaml | 2>&1 tee logs/train_tatqa_lambda1.0.log
wait

# oumi train -c configs/sft_disable_completions_only.yaml | 2>&1 tee logs/train_sft_disable_completions_only.log
# wait

# oumi train -c configs/countdown_cross.yaml | 2>&1 tee logs/train_countdown_cross_v3.log
# wait


# oumi train -c configs/countdown_tatqa_lambda0.5.yaml \
#   --training.learning_rate 1e-5 \
#   --training.warmup_ratio 0.01 \
#   --training.output_dir "output/tatqa_qwen25_15b_qwen34b_lr1e-5_warmup0.01" \
#   --training.run_name "lr1e-5_warmup0.01_tatqa_qwen25_15b_qwen34b" \
#   | 2>&1 tee logs/lr1e-5_warmup0.01_tatqa_qwen25_15b_qwen34b.log
# wait

# oumi train -c configs/countdown_tatqa_lambda0.5.yaml \
#   --training.learning_rate 1e-6 \
#   --training.warmup_ratio 0.01 \
#   --training.output_dir "output/tatqa_qwen25_15b_qwen34b_lr1e-6_warmup0.01" \
#   --training.run_name "lr1e-6_warmup0.01_tatqa_qwen25_15b_qwen34b" \
#   | 2>&1 tee logs/lr1e-6_warmup0.01_tatqa_qwen25_15b_qwen34b.log
# wait

# oumi train -c configs/countdown_tatqa_lambda0.5.yaml \
#   --training.learning_rate 1e-7 \
#   --training.warmup_ratio 0.01 \
#   --training.output_dir "output/tatqa_qwen25_15b_qwen34b_lr1e-7_warmup0.01" \
#   --training.run_name "lr1e-7_warmup0.01_tatqa_qwen25_15b_qwen34b" \
#   | 2>&1 tee logs/lr1e-7_warmup0.01_tatqa_qwen25_15b_qwen34b.log
# wait

# oumi train -c configs/countdown_tatqa_lambda0.5.yaml \
#   --training.learning_rate 1e-5 \
#   --training.warmup_ratio 0.05 \
#   --training.output_dir "output/tatqa_qwen25_15b_qwen34b_lr1e-5_warmup0.05" \
#   --training.run_name "lr1e-5_warmup0.05_tatqa_qwen25_15b_qwen34b" \
#   | 2>&1 tee logs/lr1e-5_warmup0.05_tatqa_qwen25_15b_qwen34b.log
# wait

# oumi train -c configs/countdown_tatqa_lambda0.5.yaml \
#   --training.learning_rate 1e-6 \
#   --training.warmup_ratio 0.05 \
#   --training.output_dir "output/tatqa_qwen25_15b_qwen34b_lr1e-6_warmup0.05" \
#   --training.run_name "lr1e-6_warmup0.05_tatqa_qwen25_15b_qwen34b" \
#   | 2>&1 tee logs/lr1e-6_warmup0.05_tatqa_qwen25_15b_qwen34b.log
# wait

# oumi train -c configs/countdown_tatqa_lambda0.5.yaml \
#   --training.learning_rate 1e-7 \
#   --training.warmup_ratio 0.05 \
#   --training.output_dir "output/tatqa_qwen25_15b_qwen34b_lr1e-7_warmup0.05" \
#   --training.run_name "lr1e-7_warmup0.05_tatqa_qwen25_15b_qwen34b" \
#   | 2>&1 tee logs/lr1e-7_warmup0.05_tatqa_qwen25_15b_qwen34b.log
# wait

# oumi train -c configs/sft.yaml | 2>&1 tee logs/train_sft.log
# wait

# oumi train -c configs/countdown_v2.yaml | 2>&1 tee logs/train_countdown_v2.log
# wait

# oumi train -c countdown.yaml | 2>&1 tee train_countdown.log
# wait

# python countdown_trl.py | 2>&1 tee train_countdown_hf.log
# wait

# python countdown_trl.py | 2>&1 tee train_tatqa_hf.log
# wait

# oumi train -c debug_2.yaml | tee train_debug_2.log
# wait

# oumi train -c debug_3.yaml | tee train_debug_3.log
# wait

# oumi train -c debug_4.yaml | tee train_debug_4.log
# wait
