# Qwen3-VL freeze_layers "visual" Not Found

## Issue

The existing Qwen3-VL training configs specify `freeze_layers: ["visual"]` to freeze the visual encoder during fine-tuning, but this layer name is not found in the model:

```
WARNING: Layer 'visual' not found in model.
WARNING: 0 layer(s) frozen based on the config: ['visual'].
```

This means the visual encoder is **not being frozen** during training, contrary to the config's intent.

## Root Cause

The layer name `visual` was valid in earlier versions of the Qwen-VL model architecture but the layer naming changed in the Qwen3-VL models in newer transformers versions. The freeze logic in `src/oumi/builders/models.py` does a prefix match against model parameter names, but no parameters start with `visual` in the current Qwen3-VL architecture.

## Impact

- **FFT configs**: All parameters (including the visual encoder) are trainable, increasing memory usage and potentially degrading visual features when training on small datasets.
- **LoRA configs**: The visual encoder is not explicitly frozen, but LoRA only targets specific modules (`q_proj`, `v_proj`, etc.), so the visual encoder parameters are still not updated via LoRA. However, gradient computation still flows through the visual encoder, wasting compute.

## Files Affected

- `configs/recipes/vision/qwen3_vl/sft/4b_instruct_lora_train.yaml`
- `configs/recipes/vision/qwen3_vl/sft/4b_instruct_fft_train.yaml`
- `configs/recipes/vision/qwen3_vl_2b/sft/2b_instruct_lora_train.yaml`
- `configs/recipes/vision/qwen3_vl_2b/sft/2b_instruct_fft_train.yaml`

## Fix Required

Investigate the correct layer name prefix for the visual encoder in the current Qwen3-VL architecture and update the configs. Run the following to find the correct prefix:

```python
from transformers import AutoModelForImageTextToText
model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
# Find visual encoder parameter name prefixes
prefixes = set()
for name, _ in model.named_parameters():
    if "visual" in name.lower() or "vision" in name.lower() or "image" in name.lower():
        prefixes.add(name.split(".")[0] + "." + name.split(".")[1])
print(sorted(prefixes))
```
