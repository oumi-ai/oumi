# oumi train -c configs/countdown_cross.yaml 2>&1 | tee logs/train_countdown_cross_v3.log
# wait

# oumi train -c configs/countdown_sft.yaml 2>&1 | tee logs/train_countdown_sft.log
# wait

# oumi train -c configs/countdown_v2.yaml 2>&1 | tee logs/train_countdown_v2_debug.log
# wait

# oumi train -c countdown.yaml 2>&1 | tee train_countdown.log
# wait

# python countdown_trl.py 2>&1 | tee train_countdown_hf.log
# wait

# oumi train -c debug_2.yaml | tee train_debug_2.log
# wait

# oumi train -c debug_3.yaml | tee train_debug_3.log
# wait

# oumi train -c debug_4.yaml | tee train_debug_4.log
# wait

# oumi train -c configs/tatqa_llama_lambda0.0.yaml 2>&1 | tee logs/train_tatqa_llama_lambda0.0.log
# # wait
# oumi train -c configs/tatqa_llama_lambda0.5.yaml 2>&1 | tee logs/train_tatqa_llama_lambda0.5.log
# wait
# oumi train -c configs/tatqa_llama_lambda1.0.yaml 2>&1 | tee logs/train_tatqa_llama_lambda1.0.log
# wait

# oumi train -c configs/tatqa_llama_lambda0.5_1epoch_lora.yaml 2>&1 | tee logs/train_tatqa_llama_lambda0.5_1epoch_lora.log
# wait
# oumi train -c configs/tatqa_llama_lambda0.5_1epoch.yaml 2>&1 | tee logs/train_tatqa_llama_lambda0.5_1epoch.log
# wait
# oumi train -c configs/tatqa_llama_lambda1.0.yaml 2>&1 | tee logs/train_tatqa_llama_lambda1.0.log
# wait

# oumi train -c configs/tatqa_cross_llama3.1_8b_qwen3_4b_lora.yaml 2>&1 | tee logs/train_tatqa_cross_llama3.1_8b_qwen3_4b_lora.log
# oumi train -c configs/tatqa_cross_llama3.1_8b_qwen3_4b_v2.yaml 2>&1 | tee logs/train_tatqa_cross_llama3.1_8b_qwen3_4b_v2.log
# wait

# oumi train -c configs/tatqa_llama_lambda0.0_1epoch_lora_inversekl.yaml 2>&1 | tee logs/train_tatqa_llama_lambda0.0_1epoch_lora_inversekl.log
oumi train -c configs/tatqa_llama_lambda1.0_1epoch_lora_inversekl.yaml 2>&1 | tee logs/train_tatqa_llama_lambda1.0_1epoch_lora_inversekl.log
wait

# oumi distributed torchrun \
#     -m oumi train \
#     -c configs/tatqa_llama_sft.yaml 2>&1 | tee logs/train_tatqa_llama_sft.log
