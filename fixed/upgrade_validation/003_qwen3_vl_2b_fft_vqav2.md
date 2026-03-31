# Validation: Qwen3-VL-2B-Instruct Full Fine-Tuning (vqav2-small)

## Config

`configs/recipes/vision/qwen3_vl_2b/sft/2b_instruct_fft_train.yaml`

## Model

- **Model**: `Qwen/Qwen3-VL-2B-Instruct`
- **Parameters**: 2.13B total, 2.13B trainable (100%)
- **Type**: Vision-language model, full fine-tune (visual encoder intended to be frozen but `freeze_layers` ineffective)

## Dataset

- **Dataset**: `merve/vqav2-small` (validation[:512], 512 examples)
- **Steps**: 10
- **Batch size**: 1 × 32 gradient accumulation = effective 32

## Environment

- **GPU**: 1× NVIDIA L40S (44.4 GB)
- **Transformers**: 5.3.0
- **Peak GPU memory**: 18.05 GB

## Training Results

| Step | Loss  | Grad Norm | Token Accuracy | LR       |
|------|-------|-----------|----------------|----------|
| 1    | 4.431 | 8.875     | 0.6094         | 0        |
| 2    | 4.524 | 9.000     | 0.6165         | 2.00e-05 |
| 3    | 2.759 | 7.156     | 0.6914         | 1.94e-05 |
| 4    | 2.023 | 2.875     | 0.7594         | 1.77e-05 |
| 5    | 1.832 | 2.594     | 0.7707         | 1.50e-05 |
| 6    | 1.542 | 2.844     | 0.8086         | 1.17e-05 |
| 7    | 1.513 | 3.141     | 0.8257         | 8.26e-06 |
| 8    | 1.442 | 3.203     | 0.8146         | 5.00e-06 |
| 9    | 1.401 | 3.219     | 0.8062         | 2.34e-06 |
| 10   | 1.358 | 3.312     | 0.8188         | 6.03e-07 |

- **Throughput**: ~580 tokens/sec
- **Total runtime**: 166s

## Comparison with LoRA

| Metric             | FFT       | LoRA      |
|--------------------|-----------|-----------|
| Loss (step 10)     | 1.358     | 4.266     |
| Token accuracy     | 0.819     | 0.652     |
| Peak GPU memory    | 18.05 GB  | 5.31 GB   |
| Trainable params   | 2.13B     | 8.7M      |

FFT converges much faster on small datasets since all parameters are trainable.

## Inference Validation (vLLM)

Tested with `configs/recipes/vision/qwen3_vl_2b/inference/vllm_infer.yaml`:

- **Base model response**: "This image is a famous Japanese woodblock print titled 'The Great Wave off Kanagawa' by the artist Katsushika Hokusai..."
- **Fine-tuned model response**: "wave"

The fine-tuned model produces VQA-style short answers as expected from vqav2 training.

## Issues Encountered

Same as LoRA run (002) — all fixes from that run carried over. No additional issues.

## Result

**PASS** — Training completes, strong convergence, inference works with vLLM.
