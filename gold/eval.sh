#   CUDA_VISIBLE_DEVICES=0 oumi evaluate -c /data/shanghong/oumi/gold/eval_configs/control_expts.yaml \
#     --output_dir "/data/shanghong/oumi/gold/output/control_tatqa/Qwen2.5-1.5B-Instruct" \
#     --model.model_name "Qwen/Qwen2.5-1.5B-Instruct" &

#   CUDA_VISIBLE_DEVICES=1 oumi evaluate -c /data/shanghong/oumi/gold/eval_configs/control_expts.yaml \
#     --output_dir "/data/shanghong/oumi/gold/output/control_tatqa/tatqa_qwen25_15b_qwen34b_lambda0.5" \
#     --model.model_name "/data/shanghong/oumi/gold/output/tatqa_qwen25_15b_qwen34b_lambda0.5/checkpoint-1100" &

#   CUDA_VISIBLE_DEVICES=2 oumi evaluate -c /data/shanghong/oumi/gold/eval_configs/control_expts.yaml \
#     --output_dir "/data/shanghong/oumi/gold/output/control_tatqa/qwen2.5_1.5b.sft_completions_only" \
#     --model.model_name "/data/shanghong/oumi/gold/output/qwen2.5_1.5b.sft_completions_only" &

#   CUDA_VISIBLE_DEVICES=3 oumi evaluate -c /data/shanghong/oumi/gold/eval_configs/control_expts.yaml \
#     --output_dir "/data/shanghong/oumi/gold/output/control_tatqa/Qwen3-4B-Instruct" \
#     --model.model_name "Qwen/Qwen3-4B-Instruct-2507" &

#   CUDA_VISIBLE_DEVICES=4 oumi evaluate -c /data/shanghong/oumi/gold/eval_configs/control_expts.yaml \
#     --output_dir "/data/shanghong/oumi/gold/output/control_tatqa/Llama-3.1-8B-Instruct" \
#     --model.model_name "meta-llama/Llama-3.1-8B-Instruct" &

#   CUDA_VISIBLE_DEVICES=5 oumi evaluate -c /data/shanghong/oumi/gold/eval_configs/control_expts.yaml \
#     --output_dir "/data/shanghong/oumi/gold/output/control_tatqa/tatqa_llama_lambda0.5" \
#     --model.model_name "/data/shanghong/oumi/gold/output/tatqa_llama_lambda0.5/checkpoint-1100" &

#   CUDA_VISIBLE_DEVICES=6 oumi evaluate -c /data/shanghong/oumi/gold/eval_configs/control_expts.yaml \
#     --output_dir "/data/shanghong/oumi/gold/output/control_tatqa/tinker_llama3.1_8b_ckpt700" \
#     --model.model_name "/data/shanghong/oumi/gold/output/tinker_llama3.1_8b_ckpt700" &

# wait

#   CUDA_VISIBLE_DEVICES=0,1,2,4 oumi evaluate -c /data/shanghong/oumi/gold/eval_configs/control_expts.yaml \
#     --output_dir "/data/shanghong/oumi/gold/output/control_tatqa/Llama-3.3-70B-Instruct" \
#     --model.model_name "meta-llama/Llama-3.3-70B-Instruct"

  # CUDA_VISIBLE_DEVICES=0 oumi evaluate -c /data/shanghong/oumi/gold/eval_configs/control_expts.yaml \
  #   --output_dir "/data/shanghong/oumi/gold/output/control_countdown/gold_qwen25_15b_qwen34b_v2" \
  #   --model.model_name "/data/shanghong/oumi/gold/output/gold_qwen25_15b_qwen34b_v2/checkpoint-3000"

  # CUDA_VISIBLE_DEVICES=1 oumi evaluate -c /data/shanghong/oumi/gold/eval_configs/control_expts.yaml \
  #   --output_dir "/data/shanghong/oumi/gold/output/control_countdown/grpo_countdown_qwen3-4b" \
  #   --model.model_name "/data/shanghong/oumi/grpo/output/grpo_countdown_qwen3-4b/checkpoint-12000"

  # CUDA_VISIBLE_DEVICES=1 oumi evaluate -c /data/shanghong/oumi/gold/eval_configs/control_expts.yaml \
  #   --output_dir "/data/shanghong/oumi/gold/output/control_countdown/tatqa_llama3.1_8b_sft" \
  #   --model.model_name "/data/shanghong/oumi/gold/output/tatqa_llama3.1_8b_sft"

  # CUDA_VISIBLE_DEVICES=1 oumi evaluate -c /data/shanghong/oumi/gold/eval_configs/control_expts.yaml \
  #   --output_dir "/data/shanghong/oumi/gold/output/control_countdown/tatqa_llama3.1_8b_sft" \
  #   --model.model_name "/data/shanghong/oumi/gold/output/tatqa_llama3.1_8b_sft"

  # CUDA_VISIBLE_DEVICES=1 oumi evaluate -c /data/shanghong/oumi/gold/eval_configs/control_expts.yaml \
  #   --output_dir "/data/shanghong/oumi/gold/output/control_countdown/tatqa_llama_lambda0.5_1epoch_inversekl_lora" \
  #   --model.model_name "meta-llama/Llama-3.1-8B-Instruct" \
  #   --model.adapter_model "/data/shanghong/oumi/gold/output/tatqa_llama_lambda0.5_1epoch_inversekl_lora/checkpoint-815"

  # CUDA_VISIBLE_DEVICES=2 oumi evaluate -c /data/shanghong/oumi/gold/eval_configs/control_expts.yaml \
  #   --output_dir "/data/shanghong/oumi/gold/output/control_countdown/tatqa_llama_lambda0.5_inversekl_uld" \
  #   --model.model_name "/data/shanghong/oumi/gold/output/tatqa_llama_lambda0.5_inversekl_uld/checkpoint-1600"

  CUDA_VISIBLE_DEVICES=3 oumi evaluate -c /data/shanghong/oumi/gold/eval_configs/control_expts.yaml \
    --output_dir "/data/shanghong/oumi/gold/output/control_countdown/tatqa_llama_lambda0.5_1epoch" \
    --model.model_name "/data/shanghong/oumi/gold/output/tatqa_llama_lambda0.5_1epoch/checkpoint-819"

  CUDA_VISIBLE_DEVICES=4 oumi evaluate -c /data/shanghong/oumi/gold/eval_configs/control_expts.yaml \
    --output_dir "/data/shanghong/oumi/gold/output/control_tatqa/tatqa_llama_lambda0.5_1epoch_lora" \
    --model.model_name "meta-llama/Llama-3.1-8B-Instruct" \
    --model.adapter_model "/data/shanghong/oumi/gold/output/tatqa_llama_lambda0.5_1epoch_lora/checkpoint-819"

  # CUDA_VISIBLE_DEVICES=5 oumi evaluate -c /data/shanghong/oumi/gold/eval_configs/control_expts.yaml \
  #   --output_dir "/data/shanghong/oumi/gold/output/control_tatqa/tatqa_llama_lambda0.5_1epoch_lora" \
    # --model.model_name "/data/shanghong/oumi/gold/output/tinker_llama3.1_8b_ckpt6200"

  CUDA_VISIBLE_DEVICES=5 oumi evaluate -c /data/shanghong/oumi/gold/eval_configs/control_expts.yaml \
    --output_dir "/data/shanghong/oumi/gold/output/control_tatqa/tinker_llama3.1_8b_ckpt6200" \
    --model.model_name "/data/shanghong/oumi/gold/output/tinker_llama3.1_8b_ckpt6200"

