# `use_cache=False` causes Idefics2/Idefics3 models to ignore image content during generation

## Description

Setting `use_cache=False` in `GenerationConfig` causes Idefics2 and Idefics3 VLM architectures to produce text completely unrelated to the input image. The model behaves as if no image was provided, despite `pixel_values` and `pixel_attention_mask` being correctly passed to `model.generate()`.

With `use_cache=True` (the default), the same models, inputs, and configuration produce correct image-grounded output.

Other VLM architectures (LLaVA, LLaVA-Next, Qwen2-VL) are **not** affected.

## Minimal reproduction

```python
import torch
import transformers
from PIL import Image

model_name = "HuggingFaceTB/SmolVLM-256M-Instruct"
processor = transformers.AutoProcessor.from_pretrained(model_name)
model = transformers.AutoModelForImageTextToText.from_pretrained(
    model_name, dtype=torch.bfloat16
).to("cuda")

image = Image.new("RGB", (100, 100), color="red")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What color is this image?"},
        ],
    }
]
text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=text, images=[image], return_tensors="pt").to("cuda")

for use_cache in [True, False]:
    config = transformers.GenerationConfig(
        max_new_tokens=15, do_sample=False, use_cache=use_cache
    )
    output = model.generate(**inputs, generation_config=config)
    response = processor.tokenizer.decode(
        output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    print(f"use_cache={use_cache!s:<5}  {response!r}")
```

### Observed output

```
use_cache=True   ' The image is red in color.'
use_cache=False  ' The image is a screenshot of a table.'
```

### Expected output

Both should produce image-grounded text describing a red image. `use_cache` is a performance optimization for autoregressive decoding — it should not change the semantic content of the output.

## Affected models

Tested on `transformers==5.3.0`:

| Model | Architecture | `use_cache=True` | `use_cache=False` | Affected? |
|-------|-------------|---|---|---|
| `HuggingFaceTB/SmolVLM-256M-Instruct` | Idefics3 | `' The image is red in color.'` | `' The image is a screenshot of a table.'` | **YES** |
| `HuggingFaceTB/SmolVLM-500M-Instruct` | Idefics3 | `' The image is red in color.'` | `' The image is a photograph of a white and black colored text...'` | **YES** |
| `HuggingFaceM4/idefics2-8b` | Idefics2 | `'Red'` | `'Redmond'` | **YES** |
| `Qwen/Qwen2-VL-2B-Instruct` | Qwen2VL | `"I'm sorry, but as an AI..."` | `"I'm sorry, but as an AI..."` | No |
| `llava-hf/llava-1.5-7b-hf` | LLaVA | `'This image is red.'` | `'This image is red.'` | No |
| `llava-hf/llava-v1.6-mistral-7b-hf` | LLaVA-Next | `'The image is a solid, bright red color.'` | `'The image is a solid, bright red color.'` | No |

Additional observations:
- With a real photograph (The Great Wave off Kanagawa, 800x552 JPEG) on SmolVLM-256M:
  - `use_cache=True`: `' A Japanese woodblock print shows a large wave crashing'`
  - `use_cache=False`: `' A user has asked a question about a product of'`
- The processor output tensors (`input_ids`, `pixel_values`, `pixel_attention_mask`, `attention_mask`) are identical in both cases. Only `GenerationConfig.use_cache` differs.
- Reproducible with both `torch.float16` and `torch.bfloat16`.
- Reproducible with both `device_map="auto"` and explicit `.to("cuda")`.
- Batch size does not matter (tested batch_size=1 and batch_size=2).

## Root cause hypothesis

When `use_cache=False`, the model recomputes attention from scratch at every decoding step. In Idefics2 and Idefics3, image embeddings are merged into the hidden states by replacing `<image>` token positions with vision encoder outputs. This merging likely only occurs on the first forward pass (when `inputs_embeds` is constructed from `input_ids` + `pixel_values`). On subsequent decoding steps without KV cache, the model receives the full `input_ids` again but may not re-merge the image embeddings, causing the image tokens to be treated as empty/uninitialized embeddings.

LLaVA and Qwen2-VL handle this correctly — likely because their architectures re-merge image embeddings on every forward pass, or their `prepare_inputs_for_generation` methods correctly preserve `pixel_values` across steps when `use_cache=False`.

The fix would be to ensure `Idefics2ForConditionalGeneration.prepare_inputs_for_generation` and `Idefics3ForConditionalGeneration.prepare_inputs_for_generation` pass through `pixel_values` and `pixel_attention_mask` on every step (not just the first) when `use_cache=False`.

## Environment

- `transformers` version: 5.3.0
- `torch` version: 2.10.0+cu128
- Python: 3.11
- GPU: CUDA-capable GPU
