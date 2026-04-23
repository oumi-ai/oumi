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
# oumi train -c configs/tatqa_llama_lambda1.0_1epoch_lora_inversekl.yaml 2>&1 | tee logs/train_tatqa_llama_lambda1.0_1epoch_lora_inversekl.log
# wait

# oumi distributed torchrun \
#     -m oumi train \
#     -c configs/tatqa_llama_sft.yaml 2>&1 | tee logs/train_tatqa_llama_sft.log

# # Billing experiments
# oumi distributed torchrun \
#     -m oumi train \
#     -c configs/billing_expts/tatqa_llama_sft.yaml 2>&1 | tee logs/train_billingexpts_tatqa_llama_sft.log

# wait
# oumi train -c configs/billing_expts/tatqa_llama_lambda0.0_fft_fowardkl.yaml 2>&1 | tee logs/train_billingexpts_tatqa_llama_lambda0.0_fft_forwardkl.log
# wait
# oumi train -c configs/billing_expts/tatqa_llama_lambda0.0_fft_inversekl.yaml 2>&1 | tee logs/train_billingexpts_tatqa_llama_lambda0.0_fft_inversekl.log
# wait
# oumi train -c configs/billing_expts/tatqa_llama_lambda0.0_lora_forwardkl.yaml 2>&1 | tee logs/train_billingexpts_tatqa_llama_lambda0.0_lora_forwardkl.log
# wait
# oumi train -c configs/billing_expts/tatqa_llama_lambda0.0_lora_inversekl.yaml 2>&1 | tee logs/train_billingexpts_tatqa_llama_lambda0.0_lora_inversekl.log
# wait
# oumi train -c configs/billing_expts/tatqa_llama_lambda0.5_fft_forwardkl.yaml 2>&1 | tee logs/train_billingexpts_tatqa_llama_lambda0.5_fft_forwardkl.log
# wait
# oumi train -c configs/billing_expts/tatqa_llama_lambda0.5_fft_inversekl.yaml 2>&1 | tee logs/train_billingexpts_tatqa_llama_lambda0.5_fft_inversekl.log
# wait
# oumi train -c configs/billing_expts/tatqa_llama_lambda0.5_lora_forwardkl.yaml 2>&1 | tee logs/train_billingexpts_tatqa_llama_lambda0.5_lora_forwardkl.log
# wait
# oumi train -c configs/billing_expts/tatqa_llama_lambda0.5_lora_inversekl.yaml 2>&1 | tee logs/train_billingexpts_tatqa_llama_lambda0.5_lora_inversekl.log
# wait
# oumi train -c configs/billing_expts/tatqa_llama_lambda1.0_fft_forwardkl.yaml 2>&1 | tee logs/train_billingexpts_tatqa_llama_lambda1.0_fft_forwardkl.log
# wait
# oumi train -c configs/billing_expts/tatqa_llama_lambda1.0_fft_inversekl.yaml 2>&1 | tee logs/train_billingexpts_tatqa_llama_lambda1.0_fft_inversekl.log
# wait
# oumi train -c configs/billing_expts/tatqa_llama_lambda1.0_lora_forwardkl.yaml 2>&1 | tee logs/train_billingexpts_tatqa_llama_lambda1.0_lora_forwardkl.log
# wait
# oumi train -c configs/billing_expts/tatqa_llama_lambda1.0_lora_inversekl.yaml 2>&1 | tee logs/train_billingexpts_tatqa_llama_lambda1.0_lora_inversekl.log
# wait

# oumi train -c configs/stress_tests/llama_lambda0.5_1epoch_battery_troubleshoot.yaml 2>&1 | tee logs/train_llama_lambda0.5_1epoch_battery_troubleshoot.log
# wait

# oumi train -c configs/hermes_llama.yaml 2>&1 | tee logs/hermes_llama.log
# wait

# oumi train -c configs/alpaca_long_llama.yaml 2>&1 | tee logs/alpaca_long_llama.log
# wait

# oumi train -c configs/tatqa_llama_lambda0.5_1epoch_lora_bf16.yaml 2>&1 | tee logs/tatqa_llama_lambda0.5_1epoch_lora_bf16.log
# wait
# oumi train -c configs/tatqa_llama_lambda0.5_1epoch_lora_fp32.yaml 2>&1 | tee logs/tatqa_llama_lambda0.5_1epoch_lora_fp32.log
# wait
# oumi train -c configs/tatqa_llama_lambda0.5_1epoch_lora_bf16_nomixedprec.yaml 2>&1 | tee logs/tatqa_llama_lambda0.5_1epoch_lora_bf16_nomixedprec.log
# wait

# oumi distributed torchrun \
#     -m oumi train \
#     -c configs/train_qwen2.5_1.5b_sft.yaml 2>&1 | tee logs/train_qwen2.5_1.5b_sft.log
# wait

# CUDA_VISIBLE_DEVICES=0,1,2,3 oumi distributed torchrun \
#     --master-port=8009 \
#     -m oumi train \
#     -c configs/hermes_qwen2.5_1.5b_sft_new_collator.yaml 2>&1 | tee logs/hermes_qwen2.5_1.5b_sft_new_collator.log &
# CUDA_VISIBLE_DEVICES=4,5,6,7 oumi distributed torchrun \
#     --master-port=8008 \
#     -m oumi train \
#     -c configs/tatqa_qwen2.5_1.5b_sft_new_collator.yaml 2>&1 | tee logs/tatqa_qwen2.5_1.5b_sft_new_collator.log
# wait

CUDA_VISIBLE_DEVICES=0,1,2,3 oumi distributed torchrun \
    --master-port=8009 \
    -m oumi train \
    -c configs/hermes_llama3.1_8b_old_collator.yaml 2>&1 | tee logs/hermes_llama3.1_8b_old_collator.log &
CUDA_VISIBLE_DEVICES=4,5,6,7 oumi distributed torchrun \
    --master-port=8008 \
    -m oumi train \
    -c configs/tatqa_llama3.1_8b_old_collator.yaml 2>&1 | tee logs/tatqa_llama3.1_8b_old_collator.log
wait
