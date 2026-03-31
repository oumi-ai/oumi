# Validation: Qwen3-VL-2B-Instruct LoRA Fine-Tuning (vqav2-small)

## Config

`configs/recipes/vision/qwen3_vl_2b/sft/2b_instruct_lora_train.yaml` (initial version with vqav2)

## Model

- **Model**: `Qwen/Qwen3-VL-2B-Instruct`
- **Parameters**: 2.14B total, 8.7M trainable (0.41%)
- **Type**: Vision-language model
- **LoRA**: r=8, alpha=16, targets: q/k/v/o/gate/up/down_proj

## Dataset

- **Dataset**: `merve/vqav2-small` (validation[:512], 512 examples)
- **Steps**: 10
- **Batch size**: 1 × 32 gradient accumulation = effective 32

## Environment

- **GPU**: 1× NVIDIA L40S (44.4 GB)
- **Transformers**: 5.3.0
- **Peak GPU memory**: 5.31 GB

## Training Results

| Step | Loss  | Grad Norm | Token Accuracy | LR       |
|------|-------|-----------|----------------|----------|
| 1    | 4.431 | 0.163     | 0.6094         | 0        |
| 2    | 4.524 | 0.150     | 0.6165         | 2.00e-05 |
| 3    | 4.414 | 0.155     | 0.6177         | 1.94e-05 |
| 5    | 4.373 | 0.142     | 0.6259         | 1.50e-05 |
| 8    | 4.273 | 0.154     | 0.6534         | 5.00e-06 |
| 10   | 4.266 | 0.144     | 0.6517         | 6.03e-07 |

- **Throughput**: ~529 tokens/sec

## Issues Encountered

1. **`torchdata.datapipes` missing** — `use_torchdata: True` failed because `torchdata` 0.11.0 removed `datapipes`. Fixed by setting `use_torchdata: False`.
2. **Full dataset too slow without torchdata** — 21k image examples processed eagerly at init. Fixed by using `split: "validation[:512]"` to slice at the HF dataset level.
3. **`include_tokens_per_second` removed in transformers v5** — Fixed in `training_params.py`. See `upgrade_issues/003_include_tokens_per_second_removed.md`.
4. **`mm_token_type_ids` shape mismatch** — Processor returns `[1, seq_len]` but collator stacked into `[1, 1, seq_len]`. Fixed by adding `mm_token_type_ids` to Qwen3-VL internal model config with `DROP_IF_DUMMY`. See `upgrade_issues/015_qwen3_vl_mm_token_type_ids.md`.
5. **`freeze_layers: ["visual"]` not found** — Visual encoder layer name not recognized. 0 layers frozen. See `upgrade_issues/017_qwen3_vl_freeze_layers_visual_not_found.md`.

## Result

**PASS** — Training completes, loss decreases, model saved. Multiple code fixes required.
