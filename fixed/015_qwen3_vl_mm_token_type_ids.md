# Qwen3-VL mm_token_type_ids Not Handled in Internal Model Config

## Breaking Change

Training Qwen3-VL models fails with an `IndexError` during the forward pass because `mm_token_type_ids` has a mismatched batch dimension compared to `attention_mask`.

```
IndexError: The shape of the mask [290] at index 0 does not match the shape of
the indexed tensor [1, 290] at index 0
```

## Root Cause

The Qwen3-VL processor produces `mm_token_type_ids` with a leading batch dimension `[1, seq_len]`. This field was not registered in oumi's internal model config for Qwen3-VL, so the feature generator did not know to strip the batch dimension via `first_dim_action=DROP_IF_DUMMY`.

Without this, each example's `mm_token_type_ids` retains its `[1, seq_len]` shape. The collator's `pad_to_max_dim_and_stack` then stacks these into `[batch_size, 1, seq_len]` instead of the expected `[batch_size, seq_len]`. During the model's forward pass, `attention_mask[batch_idx]` produces a 1D tensor while `mm_token_type_ids[batch_idx]` remains 2D, causing the shape mismatch.

## How to Reproduce

Train any Qwen3-VL model with a vision-language dataset:

```bash
oumi train -c configs/recipes/vision/qwen3_vl/sft/4b_instruct_lora_train.yaml
```

## Files Changed

- `src/oumi/core/configs/internal/supported_models.py`

## Fix Applied

Added `mm_token_type_ids` to the Qwen3-VL internal model config with `DROP_IF_DUMMY` so the processor's leading batch dimension is stripped when processing individual examples.

```python
config.model_input_features["mm_token_type_ids"] = InternalFeatureSpec(
    name="mm_token_type_ids",
    required=False,
    variable_shape=False,
    first_dim_action=InternalFeatureFirstDimAction.DROP_IF_DUMMY,
)
```
