# Validation: Qwen3-VL-2B-Instruct LoRA Fine-Tuning (llava-instruct-mix-vsft, 100 steps)

## Config

`configs/recipes/vision/qwen3_vl_2b/sft/2b_instruct_lora_train.yaml`

## Model

- **Model**: `Qwen/Qwen3-VL-2B-Instruct`
- **Parameters**: 2.14B total, 8.7M trainable (0.41%)
- **Type**: Vision-language model
- **LoRA**: r=8, alpha=16, targets: q/k/v/o/gate/up/down_proj
- **Save mode**: MERGED (LoRA merged into base weights for vLLM compatibility)

## Dataset

- **Dataset**: `HuggingFaceH4/llava-instruct-mix-vsft` (train[:512], 512 examples)
- **Type**: Multi-turn instruction-following conversations with images
- **Steps**: 100
- **Batch size**: 1 × 32 gradient accumulation = effective 32
- **Effective epochs**: ~6.25 over the 512-example subset

## Environment

- **GPU**: 1× NVIDIA L40S (44.4 GB)
- **Transformers**: 5.3.0
- **Peak GPU memory**: 6.40 GB

## Training Results

| Step | Loss  | Grad Norm | Token Accuracy | LR       | Tokens Seen |
|------|-------|-----------|----------------|----------|-------------|
| 10   | 1.916 | 0.1447    | 0.7105         | 1.98e-05 | 125K        |
| 20   | 1.545 | 0.0675    | 0.7596         | 1.87e-05 | 251K        |
| 30   | 1.358 | 0.0498    | 0.7909         | 1.67e-05 | 373K        |
| 40   | 1.183 | 0.0512    | 0.8065         | 1.39e-05 | 498K        |
| 50   | 1.081 | 0.0485    | 0.8185         | 1.08e-05 | 621K        |
| 60   | 1.060 | 0.0481    | 0.8195         | 7.60e-06 | 746K        |
| 70   | 1.013 | 0.0517    | 0.8278         | 4.63e-06 | 871K        |
| 80   | 0.969 | 0.0566    | 0.8359         | 2.23e-06 | 993K        |
| 90   | 0.993 | 0.0689    | 0.8254         | 6.28e-07 | 1.12M       |
| 100  | 0.957 | 0.0579    | 0.8365         | 5.24e-09 | 1.24M       |

- **Throughput**: ~743 tokens/sec
- **Total runtime**: 28 minutes
- **Final train loss**: 1.207 (average over all steps)

## Inference Validation (vLLM)

Tested merged checkpoint with `configs/recipes/vision/qwen3_vl_2b/inference/vllm_infer.yaml`:

**Prompt**: "Describe this image in detail." (The Great Wave off Kanagawa)

**Fine-tuned model response**: "This is a detailed description of the famous Japanese woodblock print, *The Great Wave off Kanagawa*, created by the artist Katsushika Hokusai in 1831. The image is a dynamic and powerful depiction of a massive, tumultuous wave, rendered in a style that is both realistic..."

**Base model response** (for comparison): "This image is a famous Japanese woodblock print titled 'The Great Wave off Kanagawa' by the artist Katsushika Hokusai. It is a masterpiece of ukiyo-e art..."

Both produce detailed descriptions. The fine-tuned model shows instruction-following style consistent with llava-instruct training data.

## Issues Encountered

1. **Image tokens exceed model_max_length** — With `model_max_length: 4096`, some examples had more visual tokens than text tokens could fit. Error: `Image features and image tokens do not match, tokens: 4081, features: 8580`. Fixed by:
   - Increasing `model_max_length` to 8192
   - Adding `processor_kwargs.max_pixels: 502400` to cap image resolution

2. **`peft_save_mode` enum case** — `"merged"` rejected, must be `"MERGED"` (uppercase).

3. **Tokenizer cross-version incompatibility** — Merged save with transformers v5 produced `extra_special_tokens` as a list, which v4 can't load. Fixed by re-saving tokenizer from base model. See `upgrade_issues/016_extra_special_tokens_v5_v4_incompatibility.md`.

4. **`set_tokenizer` removed in vLLM** — vLLM 0.14 no longer has `LLM.set_tokenizer()`. Fixed with `hasattr` guard. See `upgrade_issues/013_vllm_set_tokenizer_removed.md`.

## Result

**PASS** — Training completes with good convergence (loss 1.92 → 0.96), merged checkpoint loads and runs inference in vLLM successfully.
