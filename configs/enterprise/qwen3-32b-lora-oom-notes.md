### Debug worklog for Qwen3-32B LoRA training (pubmedqa use case)

Initial attempt OOM'ed:

```bash
# fails with OOM, pod dies and new one spawns
oumi distributed torchrun --nproc_per_node=8 -m oumi train \
  -c configs/enterprise/training/preset-validation/Qwen_Qwen3-32B_lora.yaml \
  --data.train.datasets.0.dataset_path=data/enterprise/pubmedqa/train.jsonl \
  --data.validation.datasets.0.dataset_path=data/enterprise/pubmedqa/val.jsonl \
  --training.output_dir=/data/tim/checkpoints/l1-validation/qwen3-32b-lora-pubmedqa
```

Try lowering all memory-intensive settings to get something to work, then see how far we can scale them back up:

```sh
# Base config (original that OOM'd)
CONFIG=configs/enterprise/training/preset-validation/Qwen_Qwen3-32B_lora-orig-oom.yaml
OUTPUT_DIR=/data/tim/checkpoints/qwen3-32b-oom-debug
# Memory-saving overrides (start minimal, scale up once stable)

# Keeping original lora settings.
# Once stable, try scaling up:
# - model_max_length: 128 -> 512 -> 1024 -> 2048 -> 4096 -> 8192
# - dataloader_prefetch_factor: 2 -> 8 -> 16 -> 32

### run 1 -----------------------------------------------------------------------------
RUN_ID=1

oumi train -c $CONFIG \
  --training.run_name=run-$RUN_ID \
  --training.output_dir=$OUTPUT_DIR-$RUN_ID \
  --data.train.datasets.0.dataset_path /data/tim/code/oumi/data/enterprise/pubmedqa/train.jsonl \
  --data.validation.datasets.0.dataset_path /data/tim/code/oumi/data/enterprise/pubmedqa/val.jsonl \
  --model.model_max_length 128 \
  --training.num_train_epochs 1 \
  --training.dataloader_prefetch_factor 2 \
  --training.dataloader_num_workers auto \
  --training.empty_device_cache_steps 10 \
  --peft.lora_r 16 \
  --peft.lora_alpha 32

# oof --model.attn_implementation flash_attention_2 gives:
# oumi.core.types.exceptions.HardwareException: Flash attention 2 was requested but it is not supported. Confirm that your hardware is compatible and then consider installing it: pip install -U flash-attn --no-build-isolation
# 
# Need to make sure this works on prod hardware setup so let's stick w sdpa...

# RUN_ID 1 produces:
# torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 250.00 MiB. GPU 0 has a total capacity of 79.19 GiB of which 2.07 GiB is free. Including 
# non-PyTorch memory, this process has 77.11 GiB memory in use. 75.23 GiB allowed; Of the allocated memory 74.23 GiB is allocated by PyTorch, and 943.54 MiB is
# reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid 
# fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)


### run 2 -----------------------------------------------------------------------------
RUN_ID=2

oumi train -c $CONFIG \
  --training.run_name=run-$RUN_ID \
  --training.output_dir=$OUTPUT_DIR-$RUN_ID \
  --data.train.datasets.0.dataset_path /data/tim/code/oumi/data/enterprise/pubmedqa/train.jsonl \
  --data.validation.datasets.0.dataset_path /data/tim/code/oumi/data/enterprise/pubmedqa/val.jsonl \
  --model.model_max_length 128 \
  --training.num_train_epochs 1 \
  --training.dataloader_prefetch_factor 2 \
  --training.dataloader_num_workers auto \
  --training.empty_device_cache_steps 10 \
  --peft.lora_r 8 \
  --peft.lora_alpha 16

# torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 250.00 MiB. GPU 0 has a total capacity of 79.19 GiB of which 2.09 GiB is free. Including 
# non-PyTorch memory, this process has 77.09 GiB memory in use. 75.23 GiB allowed; Of the allocated memory 74.99 GiB is allocated by PyTorch, and 148.31 MiB is
# reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid 
# fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)


### run 3 -----------------------------------------------------------------------------
RUN_ID=3

oumi train -c $CONFIG \
  --training.run_name=run-$RUN_ID \
  --training.output_dir=$OUTPUT_DIR-$RUN_ID \
  --data.train.datasets.0.dataset_path /data/tim/code/oumi/data/enterprise/pubmedqa/train.jsonl \
  --data.validation.datasets.0.dataset_path /data/tim/code/oumi/data/enterprise/pubmedqa/val.jsonl \
  --model.model_max_length 128 \
  --training.num_train_epochs 1 \
  --training.dataloader_prefetch_factor 2 \
  --training.dataloader_num_workers 0 \
  --training.empty_device_cache_steps 10 \
  --peft.lora_r 16 \
  --peft.lora_alpha 32

# torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 250.00 MiB. GPU 0 has a total capacity of 79.19 GiB of which 2.07 GiB is free. Including 
# non-PyTorch memory, this process has 77.11 GiB memory in use. 75.23 GiB allowed; Of the allocated memory 74.23 GiB is allocated by PyTorch, and 943.54 MiB is
# reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid 
# fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)



### run 4 -----------------------------------------------------------------------------
RUN_ID=4

oumi train -c $CONFIG \
  --training.run_name=run-$RUN_ID \
  --training.output_dir=$OUTPUT_DIR-$RUN_ID \
  --data.train.datasets.0.dataset_path /data/tim/code/oumi/data/enterprise/pubmedqa/train.jsonl \
  --data.validation.datasets.0.dataset_path /data/tim/code/oumi/data/enterprise/pubmedqa/val.jsonl \
  --model.model_max_length 128 \
  --training.num_train_epochs 1 \
  --training.dataloader_prefetch_factor 2 \
  --training.dataloader_num_workers 0 \
  --training.empty_device_cache_steps 10 \
  --peft.lora_target_modules '["q_proj", "v_proj"]' \
  --peft.lora_r 16 \
  --peft.lora_alpha 32

# torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 250.00 MiB. GPU 0 has a total capacity of 79.19 GiB of which 2.23 GiB is free. Including 
# non-PyTorch memory, this process has 76.95 GiB memory in use. 75.23 GiB allowed; Of the allocated memory 74.88 GiB is allocated by PyTorch, and 112.31 MiB is
# reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid 
# fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)


### run 4 -----------------------------------------------------------------------------
RUN_ID=4

oumi train -c $CONFIG \
  --training.run_name=run-$RUN_ID \
  --training.output_dir=$OUTPUT_DIR-$RUN_ID \
  --data.train.datasets.0.dataset_path /data/tim/code/oumi/data/enterprise/pubmedqa/train.jsonl \
  --data.validation.datasets.0.dataset_path /data/tim/code/oumi/data/enterprise/pubmedqa/val.jsonl \
  --model.model_max_length 128 \
  --training.num_train_epochs 1 \
  --training.dataloader_prefetch_factor 2 \
  --training.dataloader_num_workers 0 \
  --training.empty_device_cache_steps 10 \
  --peft.lora_target_modules '["q_proj", "v_proj"]' \
  --peft.lora_r 16 \
  --peft.lora_alpha 32


### run 5 -----------------------------------------------------------------------------
# oof haven't been using distributed, try this...
RUN_ID=5

oumi distributed torchrun --nproc_per_node=8 --master-port=9010 -m oumi train \
  -c $CONFIG \
  --training.run_name=run-$RUN_ID \
  --training.output_dir=$OUTPUT_DIR-$RUN_ID \
  --data.train.datasets.0.dataset_path /data/tim/code/oumi/data/enterprise/pubmedqa/train.jsonl \
  --data.validation.datasets.0.dataset_path /data/tim/code/oumi/data/enterprise/pubmedqa/val.jsonl \
  --model.model_max_length 128 \
  --training.num_train_epochs 1 \
  --training.dataloader_prefetch_factor 2 \
  --training.dataloader_num_workers 0 \
  --training.empty_device_cache_steps 10 \
  --peft.lora_target_modules '["q_proj", "v_proj"]' \
  --peft.lora_r 16 \
  --peft.lora_alpha 32

# command terminated with exit code 137


### run 6 -----------------------------------------------------------------------------
CONFIG=configs/enterprise/training/preset-validation/Qwen_Qwen3-32B_lora-orig-oom.yaml
OUTPUT_DIR=/data/tim/checkpoints/qwen3-32b-oom-debug

# after oom killer, need to re-run this on pod (installs c compiler):
./scripts/enterprise/gpu-pod-setup.sh \
  --hf-token <REDACTED> \
  --wandb-token <REDACTED>

RUN_ID=6

oumi distributed torchrun --nproc_per_node=8 --master-port=9010 -m oumi train \
  -c $CONFIG \
  --training.run_name run-$RUN_ID \
  --training.output_dir $OUTPUT_DIR-$RUN_ID \
  --data.train.datasets.0.dataset_path /data/tim/code/oumi/data/enterprise/pubmedqa/train.jsonl \
  --data.validation.datasets.0.dataset_path /data/tim/code/oumi/data/enterprise/pubmedqa/val.jsonl \
  --model.model_max_length 128 \
  --training.num_train_epochs 1 \
  --training.dataloader_prefetch_factor 2 \
  --training.dataloader_num_workers 0 \
  --training.empty_device_cache_steps 10 \
  --fsdp.cpu_offload false \
  --peft.lora_target_modules '["q_proj", "v_proj"]' \
  --peft.lora_r 16 \
  --peft.lora_alpha 32

# Training completed successfully but then we oom'ed out while saving the final model:
# INFO     [rank-0] Saving final model...                                                                                      train.py:586
# INFO     [rank-0] Saving FULL_STATE_DICT for final model checkpoint.                                                    hf_trainer.py:145
# ...
# command terminated with exit code 137

# so we got oom killer on model save. BUT we shouldn't be saving the full model here.
# we only need to save the adapter...

# TODO next try --training.compile false
# NB: havent tried this yet though...
# if run 7 fails, try that


### run 7 -----------------------------------------------------------------------------
CONFIG=configs/enterprise/training/preset-validation/Qwen_Qwen3-32B_lora-orig-oom.yaml
OUTPUT_DIR=/data/tim/checkpoints/qwen3-32b-oom-debug

RUN_ID=7

# try having each rank save only its shard...
oumi distributed torchrun --nproc_per_node=8 --master-port=9010 -m oumi train \
  -c $CONFIG \
  --training.run_name run-$RUN_ID \
  --training.output_dir $OUTPUT_DIR-$RUN_ID \
  --data.train.datasets.0.dataset_path /data/tim/code/oumi/data/enterprise/pubmedqa/train.jsonl \
  --data.validation.datasets.0.dataset_path /data/tim/code/oumi/data/enterprise/pubmedqa/val.jsonl \
  --training.compile false \
  --model.model_max_length 128 \
  --training.num_train_epochs 1 \
  --training.dataloader_prefetch_factor 2 \
  --training.dataloader_num_workers 0 \
  --training.empty_device_cache_steps 10 \
  --fsdp.cpu_offload false \
  --peft.lora_target_modules '["q_proj", "v_proj"]' \
  --peft.lora_r 16 \
  --peft.lora_alpha 32 \
  --fsdp.state_dict_type LOCAL_STATE_DICT


# again failed on save:
# command terminated with exit code 137


### run 8 -----------------------------------------------------------------------------
CONFIG=configs/enterprise/training/preset-validation/Qwen_Qwen3-32B_lora-orig-oom.yaml
OUTPUT_DIR=/data/tim/checkpoints/qwen3-32b-oom-debug

RUN_ID=8

# try SHARDED_STATE_DICT instead...
# but opus says this will not work since hf_trainer.py overrides this to FULL_STATE_DICT
oumi distributed torchrun --nproc_per_node=8 --master-port=9010 -m oumi train \
  -c $CONFIG \
  --training.run_name run-$RUN_ID \
  --training.output_dir $OUTPUT_DIR-$RUN_ID \
  --data.train.datasets.0.dataset_path /data/tim/code/oumi/data/enterprise/pubmedqa/train.jsonl \
  --data.validation.datasets.0.dataset_path /data/tim/code/oumi/data/enterprise/pubmedqa/val.jsonl \
  --training.compile false \
  --model.model_max_length 128 \
  --training.num_train_epochs 1 \
  --training.dataloader_prefetch_factor 2 \
  --training.dataloader_num_workers 0 \
  --training.empty_device_cache_steps 10 \
  --fsdp.cpu_offload false \
  --peft.lora_target_modules '["q_proj", "v_proj"]' \
  --peft.lora_r 16 \
  --peft.lora_alpha 32 \
  --fsdp.state_dict_type SHARDED_STATE_DICT




### run 9 -----------------------------------------------------------------------------
# WORKAROUND: Skip the final save (which forces FULL_STATE_DICT gather) and use
# epoch checkpoints instead. The final save code in hf_trainer.py OVERRIDES the
# state_dict_type to FULL_STATE_DICT, ignoring our setting. By using save_epoch=true
# and save_final_model=false, we avoid the problematic gather operation.
CONFIG=configs/enterprise/training/preset-validation/Qwen_Qwen3-32B_lora-orig-oom.yaml
OUTPUT_DIR=/data/tim/checkpoints/qwen3-32b-oom-debug

RUN_ID=9

oumi distributed torchrun --nproc_per_node=8 --master-port=9010 -m oumi train \
  -c $CONFIG \
  --training.run_name run-$RUN_ID \
  --training.output_dir $OUTPUT_DIR-$RUN_ID \
  --data.train.datasets.0.dataset_path /data/tim/code/oumi/data/enterprise/pubmedqa/train.jsonl \
  --data.validation.datasets.0.dataset_path /data/tim/code/oumi/data/enterprise/pubmedqa/val.jsonl \
  --training.compile false \
  --model.model_max_length 128 \
  --training.num_train_epochs 1 \
  --training.dataloader_prefetch_factor 2 \
  --training.dataloader_num_workers 0 \
  --training.empty_device_cache_steps 10 \
  --training.save_epoch true \
  --training.save_final_model false \
  --fsdp.cpu_offload false \
  --peft.lora_r 16 \
  --peft.lora_alpha 32

# if this works, then we will try modifying the code to only save the adapter... 

# nope still failed on epoch ckpt save </3


### run 10 -----------------------------------------------------------------------------
CONFIG=configs/enterprise/training/preset-validation/Qwen_Qwen3-32B_lora-orig-oom.yaml
OUTPUT_DIR=/data/tim/checkpoints/qwen3-32b-oom-debug

RUN_ID=10

oumi distributed torchrun --nproc_per_node=8 --master-port=9010 -m oumi train \
  -c $CONFIG \
  --training.run_name run-$RUN_ID \
  --training.output_dir $OUTPUT_DIR-$RUN_ID \
  --data.train.datasets.0.dataset_path /data/tim/code/oumi/data/enterprise/pubmedqa/train.jsonl \
  --data.validation.datasets.0.dataset_path /data/tim/code/oumi/data/enterprise/pubmedqa/val.jsonl \
  --training.compile false \
  --model.model_max_length 128 \
  --training.num_train_epochs 1 \
  --training.dataloader_prefetch_factor 2 \
  --training.dataloader_num_workers 0 \
  --training.empty_device_cache_steps 10 \
  --training.save_epoch true \
  --training.save_final_model false \
  --fsdp.state_dict_type SHARDED_STATE_DICT \
  --fsdp.cpu_offload false \
  --peft.lora_r 16 \
  --peft.lora_alpha 32

# okay -- this runs!
# wandb: 🚀 View run run-10 at: https://wandb.ai/lefft-oumi/huggingface/runs/gzpghbrn
# wandb: Find logs at: wandb/run-20260128_000822-gzpghbrn/logs
# [01/28/26 00:13:54] INFO     [rank-0] Successfully completed! (Rank: 0. Duration: 507.5 sec)

# but the loss curve is terrible of course lol, to be expected with such a small max_len, most of the data probably wasn't even seen...

# okay now see if we can actually eval the model...
# nope the bundle is fucked:
# ls -la /data/tim/checkpoints/qwen3-32b-oom-debug-10/checkpoint-25/pytorch_model_fsdp_0/
# total 536940
# drwxr-xr-x 2 root root     4096 Jan 28 00:13 .
# drwxr-xr-x 4 root root     4096 Jan 28 00:13 ..
# -rw-r--r-- 1 root root  1623812 Jan 28 00:13 .metadata
# -rw-r--r-- 1 root root 68521856 Jan 28 00:13 __0_0.distcp
# -rw-r--r-- 1 root root 68521856 Jan 28 00:13 __1_0.distcp
# -rw-r--r-- 1 root root 68521856 Jan 28 00:13 __2_0.distcp
# -rw-r--r-- 1 root root 68521856 Jan 28 00:13 __3_0.distcp
# -rw-r--r-- 1 root root 68521856 Jan 28 00:13 __4_0.distcp
# -rw-r--r-- 1 root root 68521856 Jan 28 00:13 __5_0.distcp
# -rw-r--r-- 1 root root 68521856 Jan 28 00:13 __6_0.distcp
# -rw-r--r-- 1 root root 68521856 Jan 28 00:13 __7_0.distcp


# but looks like we are actually saving the optimizer state, which we don't need to do.
# let's try and disable that...




### run 11 -----------------------------------------------------------------------------
# Based on run 6 (which completed training but OOM'd on save)
# Adding save_only_model=true to skip optimizer state saving
CONFIG=configs/enterprise/training/preset-validation/Qwen_Qwen3-32B_lora-orig-oom.yaml
OUTPUT_DIR=/data/tim/checkpoints/qwen3-32b-oom-debug

RUN_ID=11

oumi distributed torchrun --nproc_per_node=8 --master-port=9010 -m oumi train \
  -c $CONFIG \
  --training.run_name run-$RUN_ID \
  --training.output_dir $OUTPUT_DIR-$RUN_ID \
  --data.train.datasets.0.dataset_path /data/tim/code/oumi/data/enterprise/pubmedqa/train.jsonl \
  --data.validation.datasets.0.dataset_path /data/tim/code/oumi/data/enterprise/pubmedqa/val.jsonl \
  --model.model_max_length 128 \
  --training.num_train_epochs 1 \
  --training.dataloader_prefetch_factor 2 \
  --training.dataloader_num_workers 0 \
  --training.empty_device_cache_steps 10 \
  --training.trainer_kwargs.save_only_model true \
  --fsdp.cpu_offload false \
  --peft.lora_target_modules '["q_proj", "v_proj"]' \
  --peft.lora_r 16 \
  --peft.lora_alpha 32

# still OOMed on save, looks like optimizer state isn't the problem, it's the model weights gather


### run 12 -----------------------------------------------------------------------------
# CODE FIX: Added fsdp.force_full_state_dict_on_final_save flag
# This skips the FULL_STATE_DICT gather for final save, letting HF/PEFT
# save just the adapter weights without gathering the full 32B base model.

# first sync the code to cluster then install from local copy:
# (this will pip install -e .)
cd /data/tim && source .bashrc && cd code/oumi && ./scripts/enterprise/gpu-pod-setup.sh --hf-token <REDACTED> --wandb-token <REDACTED>
# pip install -e ".[gpu]"

CONFIG=configs/enterprise/training/preset-validation/Qwen_Qwen3-32B_lora-orig-oom.yaml
OUTPUT_DIR=/data/tim/checkpoints/qwen3-32b-oom-debug

RUN_ID=12

oumi distributed torchrun --nproc_per_node=8 --master-port=9010 -m oumi train \
  -c $CONFIG \
  --training.run_name run-$RUN_ID \
  --training.output_dir $OUTPUT_DIR-$RUN_ID \
  --data.train.datasets.0.dataset_path /data/tim/code/oumi/data/enterprise/pubmedqa/train.jsonl \
  --data.validation.datasets.0.dataset_path /data/tim/code/oumi/data/enterprise/pubmedqa/val.jsonl \
  --model.model_max_length 128 \
  --training.num_train_epochs 1 \
  --training.dataloader_prefetch_factor 2 \
  --training.dataloader_num_workers 0 \
  --training.empty_device_cache_steps 10 \
  --fsdp.cpu_offload false \
  --fsdp.force_full_state_dict_on_final_save false \
  --peft.lora_target_modules '["q_proj", "v_proj"]' \
  --peft.lora_r 16 \
  --peft.lora_alpha 32

# still OOMed - the flag alone doesn't help because default state_dict_type is FULL_STATE_DICT


### run 13 -----------------------------------------------------------------------------
# NEW CODE: Added _save_fsdp_peft_adapter() method that uses FSDP.summon_full_params()
# with rank0_only=True to gather only the trainable adapter weights, not the full 32B model.

# modified hf_trainer.py, try again...
CONFIG=configs/enterprise/training/preset-validation/Qwen_Qwen3-32B_lora-orig-oom.yaml
OUTPUT_DIR=/data/tim/checkpoints/qwen3-32b-oom-debug

RUN_ID=13

oumi distributed torchrun --nproc_per_node=8 --master-port=9010 -m oumi train \
  -c $CONFIG \
  --training.run_name run-$RUN_ID \
  --training.output_dir $OUTPUT_DIR-$RUN_ID \
  --data.train.datasets.0.dataset_path /data/tim/code/oumi/data/enterprise/pubmedqa/train.jsonl \
  --data.validation.datasets.0.dataset_path /data/tim/code/oumi/data/enterprise/pubmedqa/val.jsonl \
  --model.model_max_length 128 \
  --training.num_train_epochs 1 \
  --training.dataloader_prefetch_factor 2 \
  --training.dataloader_num_workers 0 \
  --training.empty_device_cache_steps 10 \
  --fsdp.cpu_offload false \
  --fsdp.force_full_state_dict_on_final_save false \
  --peft.lora_target_modules '["q_proj", "v_proj"]' \
  --peft.lora_r 16 \
  --peft.lora_alpha 32

# boom okay lookslike this actually completed and should be okay:
# wandb: 🚀 View run run-13 at: https://wandb.ai/lefft-oumi/huggingface/runs/u7hqk6to
# wandb: Find logs at: wandb/run-20260128_010321-u7hqk6to/logs
# [01/28/26 01:05:33] INFO     [rank-0] Successfully completed! (Rank: 0. Duration: 314.7 sec)            distributed_run.py:330
# root@tim-oumi-sleep-8gpu-96598f8cf-htlnz:/data/tim/code/oumi# ls -ltrh /data/tim/checkpoints/qwen3-32b-oom-debug-13/
# total 92M
# drwxr-xr-x 2 root root 4.0K Jan 28 01:00 logs
# drwxr-xr-x 3 root root 4.0K Jan 28 01:03 runs
# drwxr-xr-x 2 root root 4.0K Jan 28 01:05 telemetry
# -rw-r--r-- 1 root root 3.5K Jan 28 01:05 trainer_state.json
# -rw-r--r-- 1 root root  77M Jan 28 01:05 adapter_model.safetensors
# -rw-r--r-- 1 root root 5.3K Jan 28 01:05 tokenizer_config.json
# -rw-r--r-- 1 root root  613 Jan 28 01:05 special_tokens_map.json
# -rw-r--r-- 1 root root 4.1K Jan 28 01:05 chat_template.jinja
# -rw-r--r-- 1 root root  707 Jan 28 01:05 added_tokens.json
# -rw-r--r-- 1 root root  850 Jan 28 01:05 adapter_config.json
# -rw-r--r-- 1 root root 2.7M Jan 28 01:05 vocab.json
# -rw-r--r-- 1 root root 1.6M Jan 28 01:05 merges.txt
# -rw-r--r-- 1 root root  11M Jan 28 01:05 tokenizer.json


# can we evaluate it though?! 
# we only want task evals and control evals
run_evals() {
  local model=$1 out=$2
  oumi evaluate -c configs/enterprise/evaluation/task_pubmedqa.yaml \
    --model.model_name "$model" --model.trust_remote_code true \
    --tasks.0.eval_kwargs.test_data_path "/data/tim/code/oumi/data/enterprise/pubmedqa/test.jsonl" \
    --output_dir "$out/pubmedqa"
  oumi evaluate -c configs/enterprise/evaluation/control_evals.yaml \
    --model.model_name "$model" --model.trust_remote_code true \
    --output_dir "$out/control"
}

run_evals "$OUTPUT_DIR-$RUN_ID" "$OUTPUT_DIR/eval/run-$RUN_ID"
run_evals "Qwen/Qwen3-32B" "$OUTPUT_DIR/eval/Qwen3-32B"
```

Okay now try to run this with max len 1024:

```sh
### run 14 -----------------------------------------------------------------------------
CONFIG=configs/enterprise/training/preset-validation/Qwen_Qwen3-32B_lora-orig-oom.yaml
OUTPUT_DIR=/data/tim/checkpoints/qwen3-32b-oom-debug

RUN_ID=14

oumi distributed torchrun --nproc_per_node=8 --master-port=9010 -m oumi train \
  -c $CONFIG \
  --training.run_name run-$RUN_ID \
  --training.output_dir $OUTPUT_DIR-$RUN_ID \
  --data.train.datasets.0.dataset_path /data/tim/code/oumi/data/enterprise/pubmedqa/train.jsonl \
  --data.validation.datasets.0.dataset_path /data/tim/code/oumi/data/enterprise/pubmedqa/val.jsonl \
  --model.model_max_length 1024 \
  --training.num_train_epochs 1 \
  --training.dataloader_prefetch_factor 2 \
  --training.dataloader_num_workers 0 \
  --training.empty_device_cache_steps 10 \
  --fsdp.cpu_offload false \
  --fsdp.force_full_state_dict_on_final_save false \
  --peft.lora_target_modules '["q_proj", "v_proj"]' \
  --peft.lora_r 16 \
  --peft.lora_alpha 32

# this works, howbout 2048?

### run 15 -----------------------------------------------------------------------------
CONFIG=configs/enterprise/training/preset-validation/Qwen_Qwen3-32B_lora-orig-oom.yaml
OUTPUT_DIR=/data/tim/checkpoints/qwen3-32b-oom-debug

RUN_ID=15

oumi distributed torchrun --nproc_per_node=8 --master-port=9010 -m oumi train \
  -c $CONFIG \
  --training.run_name run-$RUN_ID \
  --training.output_dir $OUTPUT_DIR-$RUN_ID \
  --data.train.datasets.0.dataset_path /data/tim/code/oumi/data/enterprise/pubmedqa/train.jsonl \
  --data.validation.datasets.0.dataset_path /data/tim/code/oumi/data/enterprise/pubmedqa/val.jsonl \
  --model.model_max_length 2048 \
  --training.num_train_epochs 1 \
  --training.dataloader_prefetch_factor 2 \
  --training.dataloader_num_workers 0 \
  --training.empty_device_cache_steps 10 \
  --fsdp.cpu_offload false \
  --fsdp.force_full_state_dict_on_final_save false \
  --peft.lora_target_modules '["q_proj", "v_proj"]' \
  --peft.lora_r 16 \
  --peft.lora_alpha 32

# okay, now back to original max len of 8192...

### run 16 -----------------------------------------------------------------------------
# try original auto setting for dataloader workers
CONFIG=configs/enterprise/training/preset-validation/Qwen_Qwen3-32B_lora-orig-oom.yaml
OUTPUT_DIR=/data/tim/checkpoints/qwen3-32b-oom-debug

RUN_ID=16

oumi distributed torchrun --nproc_per_node=8 --master-port=9010 -m oumi train \
  -c $CONFIG \
  --training.run_name run-$RUN_ID \
  --training.output_dir $OUTPUT_DIR-$RUN_ID \
  --data.train.datasets.0.dataset_path /data/tim/code/oumi/data/enterprise/pubmedqa/train.jsonl \
  --data.validation.datasets.0.dataset_path /data/tim/code/oumi/data/enterprise/pubmedqa/val.jsonl \
  --model.model_max_length 8192 \
  --training.num_train_epochs 1 \
  --training.dataloader_prefetch_factor 2 \
  --training.dataloader_num_workers auto \
  --training.empty_device_cache_steps 10 \
  --fsdp.cpu_offload false \
  --fsdp.force_full_state_dict_on_final_save false \
  --peft.lora_target_modules '["q_proj", "v_proj"]' \
  --peft.lora_r 16 \
  --peft.lora_alpha 32


# can we restore all lora modules?

### run 17 -----------------------------------------------------------------------------
# if this works, it is the final config. if not, then we use run 15 as the final.
CONFIG=configs/enterprise/training/preset-validation/Qwen_Qwen3-32B_lora-orig-oom.yaml
OUTPUT_DIR=/data/tim/checkpoints/qwen3-32b-oom-debug

RUN_ID=17

oumi distributed torchrun --nproc_per_node=8 --master-port=9010 -m oumi train \
  -c $CONFIG \
  --training.run_name run-$RUN_ID \
  --training.output_dir $OUTPUT_DIR-$RUN_ID \
  --data.train.datasets.0.dataset_path /data/tim/code/oumi/data/enterprise/pubmedqa/train.jsonl \
  --data.validation.datasets.0.dataset_path /data/tim/code/oumi/data/enterprise/pubmedqa/val.jsonl \
  --model.model_max_length 8192 \
  --training.num_train_epochs 1 \
  --training.dataloader_prefetch_factor 2 \
  --training.dataloader_num_workers auto \
  --training.empty_device_cache_steps 10 \
  --fsdp.cpu_offload false \
  --fsdp.force_full_state_dict_on_final_save false

# works:
# wandb: 🚀 View run run-17 at: https://wandb.ai/lefft-oumi/huggingface/runs/b9k0aug4
# wandb: Find logs at: wandb/run-20260128_014224-b9k0aug4/logs
# [01/28/26 01:47:45] INFO     [rank-0] Successfully completed! (Rank: 0. Duration: 364.2 sec)

# evaluate it:
oumi evaluate -c configs/enterprise/evaluation/task_pubmedqa.yaml \
    --model.model_name "$OUTPUT_DIR-$RUN_ID" \
    --model.trust_remote_code true \
    --tasks.0.eval_kwargs.test_data_path "/data/tim/code/oumi/data/enterprise/pubmedqa/test.jsonl" \
    --output_dir "$OUTPUT_DIR/eval/run-$RUN_ID"

# ┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┓
# ┃ Benchmark           ┃ Metric              ┃ Score  ┃ Std Error ┃
# ┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━┩
# │ enterprise_pubmedqa │ Accuracy            │ 79.00% │ -         │
# ├─────────────────────┼─────────────────────┼────────┼───────────┤
# │ enterprise_pubmedqa │ Mean Response Chars │ 2.59   │ -         │
# ├─────────────────────┼─────────────────────┼────────┼───────────┤
# │ enterprise_pubmedqa │ Num Correct         │ 79     │ -         │
# ├─────────────────────┼─────────────────────┼────────┼───────────┤
# │ enterprise_pubmedqa │ Num Total           │ 100    │ -         │
# ├─────────────────────┼─────────────────────┼────────┼───────────┤
# │ enterprise_pubmedqa │ Macro F1            │ 55.59% │ -         │
# └─────────────────────┴─────────────────────┴────────┴───────────┘

# evaluate the baseline:
oumi evaluate -c configs/enterprise/evaluation/task_pubmedqa.yaml \
    --model.model_name "Qwen/Qwen3-32B" \
    --model.trust_remote_code true \
    --tasks.0.eval_kwargs.test_data_path "/data/tim/code/oumi/data/enterprise/pubmedqa/test.jsonl" \
    --output_dir "$OUTPUT_DIR/eval/Qwen3-32B"

# ┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┓
# ┃ Benchmark           ┃ Metric              ┃ Score  ┃ Std Error ┃
# ┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━┩
# │ enterprise_pubmedqa │ Accuracy            │ 12.00% │ -         │
# ├─────────────────────┼─────────────────────┼────────┼───────────┤
# │ enterprise_pubmedqa │ Mean Response Chars │ 209.70 │ -         │
# ├─────────────────────┼─────────────────────┼────────┼───────────┤
# │ enterprise_pubmedqa │ Num Correct         │ 12     │ -         │
# ├─────────────────────┼─────────────────────┼────────┼───────────┤
# │ enterprise_pubmedqa │ Num Total           │ 100    │ -         │
# ├─────────────────────┼─────────────────────┼────────┼───────────┤
# │ enterprise_pubmedqa │ Macro F1            │ 11.94% │ -         │
# └─────────────────────┴─────────────────────┴────────┴───────────┘
```

And boom, clear performance gain from LoRA, big accuracy gains and also response format is enforced (see delta in mean response chars)...

Also verified nothing changes with >1 epochs (run 18).

```sh
### run 18 -----------------------------------------------------------------------------
CONFIG=configs/enterprise/training/preset-validation/Qwen_Qwen3-32B_lora-orig-oom.yaml
OUTPUT_DIR=/data/tim/checkpoints/qwen3-32b-oom-debug

RUN_ID=18

oumi distributed torchrun --nproc_per_node=8 --master-port=9010 -m oumi train \
  -c $CONFIG \
  --training.run_name run-$RUN_ID \
  --training.output_dir $OUTPUT_DIR-$RUN_ID \
  --data.train.datasets.0.dataset_path /data/tim/code/oumi/data/enterprise/pubmedqa/train.jsonl \
  --data.validation.datasets.0.dataset_path /data/tim/code/oumi/data/enterprise/pubmedqa/val.jsonl \
  --model.model_max_length 8192 \
  --training.num_train_epochs 2 \
  --training.dataloader_prefetch_factor 2 \
  --training.dataloader_num_workers auto \
  --training.empty_device_cache_steps 10 \
  --fsdp.cpu_offload false \
  --fsdp.force_full_state_dict_on_final_save false

# yes, confirmed still runs
```