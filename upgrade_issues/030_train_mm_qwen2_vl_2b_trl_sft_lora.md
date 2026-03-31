# train_mm_qwen2_vl_2b_trl_sft_lora — E2E Test Failure

## Summary

The e2e test `test_train_multimodal_lora_1gpu_40gb[train_mm_qwen2_vl_2b_trl_sft_lora]` fails with an
`IndexError` in `transformers` v5.3.0's `Qwen2VLModel.get_rope_index()` method. The `attention_mask`
tensor has shape `[seq_len]` (1D) but `mm_token_type_ids` has shape `[batch_size, seq_len]` (2D),
causing an indexing mismatch when the mask is applied to filter tokens.

**Status**: Upstream bug in `transformers==5.3.0` Qwen2-VL implementation.

---

## Environment

| Component        | Version                           |
|------------------|-----------------------------------|
| oumi             | 0.8.dev69+g7baecbf68.d20260320   |
| transformers     | 5.3.0                             |
| trl              | 0.28.0                            |
| peft             | 0.18.1                            |
| torch            | 2.10.0+cu128                      |
| CUDA             | 12.8                              |
| cuDNN            | 9.10.2 (91002)                    |
| GPU              | NVIDIA L40S (46 GB)               |
| Python           | 3.11.15                           |

## Test Command

```bash
oumi train \
  -c /workspace/oumi/configs/recipes/vision/qwen2_vl_2b/sft/lora/train.yaml \
  --training.max_steps 3 \
  --training.output_dir <tmpdir>/train \
  --training.run_name train_mm_qwen2_vl_2b_trl_sft_lora \
  --training.trainer_type=TrainerType.TRL_SFT \
  --training.save_steps=3 \
  --training.enable_wandb=False
```

## Training Config

- **Model**: `Qwen/Qwen2-VL-2B-Instruct`
- **Trainer**: `TRL_SFT`
- **Dataset**: `merve/vqav2-small` (validation split)
- **Collator**: `vision_language_with_padding`
- **LoRA**: r=8, alpha=16, targets: q/k/v/o/gate/up/down proj
- **Frozen layers**: `visual`
- **dtype**: bfloat16
- **attn_implementation**: sdpa
- **per_device_train_batch_size**: 1
- **use_torchdata**: True

## Error

```
IndexError: The shape of the mask [378] at index 0 does not match the shape of
the indexed tensor [1, 378] at index 0
```

## Root Cause

In `transformers/models/qwen2_vl/modeling_qwen2_vl.py`, the `get_rope_index()` method (line 1118)
indexes `input_token_type` (derived from `mm_token_type_ids[batch_idx]`, shape `[seq_len]`)
with `attention_mask[batch_idx].bool()`. However, when `attention_mask` is passed in as 1D
(`[seq_len]`), `attention_mask[batch_idx]` selects a single scalar element (the value at
position `batch_idx`) rather than the intended row of the mask.

The call chain:
1. `Qwen2VLModel.forward()` (line 1338) calls `compute_3d_position_ids()` with the raw
   `attention_mask` from the trainer inputs.
2. `compute_3d_position_ids()` (line 1257) passes `attention_mask` to `get_rope_index()`.
3. `get_rope_index()` (line 1114-1118) iterates over batches and indexes:
   ```python
   for batch_idx, current_input_ids in enumerate(input_ids):
       input_token_type = mm_token_type_ids[batch_idx]       # shape: [seq_len]
       if attention_mask is not None:
           current_input_ids = current_input_ids[attention_mask[batch_idx].bool()]  # FAILS HERE
           input_token_type = input_token_type[attention_mask[batch_idx].bool()]    # line 1118
   ```

`get_rope_index()` expects `attention_mask` with shape `(batch_size, seq_len)` as documented
in its docstring (line 1089), but the TRL SFT trainer with oumi's `vision_language_with_padding`
collator appears to provide it as 1D `[seq_len]` when batch_size=1. This shape disagreement
causes the IndexError.

### Buggy Code Location

**File**: `transformers/models/qwen2_vl/modeling_qwen2_vl.py`
**Lines**: 1116-1118 (`get_rope_index`)
**Called from**: line 1257 (`compute_3d_position_ids`) → line 1338 (`forward`)

```python
# Line 1116-1118 in get_rope_index():
if attention_mask is not None:
    current_input_ids = current_input_ids[attention_mask[batch_idx].bool()]
    input_token_type = input_token_type[attention_mask[batch_idx].bool()]
```

When `attention_mask` shape is `[378]` (1D), `attention_mask[0]` returns a scalar (the first
element), not a 1D boolean mask. The subsequent `.bool()` on that scalar produces a 0-d tensor,
which cannot index a 1D tensor of shape `[378]`.

Also at line 1147:
```python
position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = llm_positions.to(position_ids.device)
```

## Full Stack Trace

```
Traceback (most recent call last):
  File "oumi/cli/train.py", line 226, in train
    oumi_train(parsed_config, verbose=verbose)
  File "oumi/__init__.py", line 270, in train
    return oumi.train.train(...)
  File "oumi/train.py", line 590, in train
    trainer.train(resume_from_checkpoint=checkpoint_location)
  File "oumi/core/trainers/hf_trainer.py", line 41, in train
    self._hf_trainer.train(resume_from_checkpoint=resume_from_checkpoint)
  File "transformers/trainer.py", line 1424, in train
    return inner_training_loop(...)
  File "transformers/trainer.py", line 1506, in _inner_training_loop
    self._run_epoch(...)
  File "transformers/trainer.py", line 1734, in _run_epoch
    tr_loss_step = self.training_step(model, inputs, num_items_in_batch)
  File "trl/trainer/sft_trainer.py", line 1290, in training_step
    return super().training_step(*args, **kwargs)
  File "transformers/trainer.py", line 1906, in training_step
    loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
  File "trl/trainer/sft_trainer.py", line 1187, in compute_loss
    (loss, outputs) = super().compute_loss(...)
  File "transformers/trainer.py", line 1978, in compute_loss
    outputs = model(**inputs)
  File "peft/peft_model.py", line 1923, in forward
    return self.base_model(...)
  File "peft/tuners/tuners_utils.py", line 311, in forward
    return self.model.forward(*args, **kwargs)
  File "transformers/models/qwen2_vl/modeling_qwen2_vl.py", line 1501, in forward  [Qwen2VLForConditionalGeneration]
    outputs: Qwen2VLModelOutputWithPast = self.model(...)
  File "transformers/models/qwen2_vl/modeling_qwen2_vl.py", line 1338, in forward  [Qwen2VLModel]
    position_ids = self.compute_3d_position_ids(...)
  File "transformers/models/qwen2_vl/modeling_qwen2_vl.py", line 1257, in compute_3d_position_ids
    position_ids, rope_deltas = self.get_rope_index(...)
  File "transformers/models/qwen2_vl/modeling_qwen2_vl.py", line 1118, in get_rope_index
    input_token_type = input_token_type[attention_mask[batch_idx].bool()]
                       ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
IndexError: The shape of the mask [378] at index 0 does not match the shape of
the indexed tensor [1, 378] at index 0
```

## Possible Fixes

1. **In transformers (upstream)**: `get_rope_index` should handle 1D attention_mask by
   unsqueezing it to 2D at the top of the method:
   ```python
   if attention_mask is not None and attention_mask.dim() == 1:
       attention_mask = attention_mask.unsqueeze(0)
   ```

2. **In oumi (workaround)**: Ensure the `vision_language_with_padding` collator always returns
   a 2D `attention_mask` with shape `(batch_size, seq_len)` even when `batch_size=1`. Or add a
   squeeze/unsqueeze fixup in the trainer before forwarding to the model.

## Reproduction

```bash
cd /workspace/oumi/tests
pytest -s -vv -m "e2e and single_gpu" -k "train_mm_qwen2_vl_2b_trl_sft_lora" --timeout=1200 e2e/
```

## Related Context

- This test previously also failed due to a cuDNN issue (`nvidia-cudnn-cu13` conflicting with
  `nvidia-cudnn-cu12`). After fixing cuDNN, this shape mismatch error was revealed as the actual
  remaining failure.
- The eval test `eval_mm_llama32v_11b_single_gpu` was fixed by the cuDNN fix alone and now passes.
- This may be a regression introduced in transformers v5.x; the `get_rope_index` and
  `compute_3d_position_ids` methods contain new 3D RoPE logic for multimodal inputs.
