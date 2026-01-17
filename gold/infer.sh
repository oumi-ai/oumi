CUDA_VISIBLE_DEVICES=0 python infer.py --input_file tatqa/train_modified.jsonl --output_file tatqa/train_final.jsonl --inference_config infer_configs/qwen3_4b.yaml | 2>&1 tee infer_logs/tatqa_train_modified.log &
wait
