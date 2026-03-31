# `use_cache=False` causes Idefics2, Idefics3, and SmolVLM to silently ignore image content during generation

## Description

`prepare_inputs_for_generation` in `Idefics2ForConditionalGeneration`, `Idefics3ForConditionalGeneration`, and `SmolVLMForConditionalGeneration` unconditionally drops `pixel_values` after the first generation step, even when `use_cache=False`. Without a KV cache to retain the image embeddings from step 1, subsequent decoding steps see raw `<image>` token embeddings instead of actual vision encoder features. The model produces text completely unrelated to the input image.

All 43 other VLMs in transformers handle this correctly. Only the Idefics family has this bug.

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

Both should produce image-grounded text. `use_cache` is a performance optimization for autoregressive decoding — it should not change the semantic content of the output.

## Affected models

Runtime-verified on `transformers==5.3.0`:

| Model | Architecture | `use_cache=True` | `use_cache=False` | Affected? |
|-------|-------------|---|---|---|
| `HuggingFaceTB/SmolVLM-256M-Instruct` | Idefics3 | `' The image is red in color.'` | `' The image is a screenshot of a table.'` | **YES** |
| `HuggingFaceTB/SmolVLM-500M-Instruct` | Idefics3 | `' The image is red in color.'` | `' The image is a photograph of a white and black colored text...'` | **YES** |
| `HuggingFaceM4/idefics2-8b` | Idefics2 | `'Red'` | `'Redmond'` | **YES** |
| `Qwen/Qwen2-VL-2B-Instruct` | Qwen2VL | `"I'm sorry, but as an AI..."` | `"I'm sorry, but as an AI..."` | No |
| `llava-hf/llava-1.5-7b-hf` | LLaVA | `'This image is red.'` | `'This image is red.'` | No |
| `llava-hf/llava-v1.6-mistral-7b-hf` | LLaVA-Next | `'The image is a solid, bright red color.'` | `'The image is a solid, bright red color.'` | No |

Additional runtime observations:
- With a real photograph (The Great Wave off Kanagawa, 800x552 JPEG) on SmolVLM-256M:
  - `use_cache=True`: `' A Japanese woodblock print shows a large wave crashing'`
  - `use_cache=False`: `' A user has asked a question about a product of'`
- The processor output tensors (`input_ids`, `pixel_values`, `pixel_attention_mask`, `attention_mask`) are identical in both cases. Only `GenerationConfig.use_cache` differs.
- Reproducible with both `torch.float16` and `torch.bfloat16`.
- Reproducible with both `device_map="auto"` and explicit `.to("cuda")`.
- Batch size does not matter (tested batch_size=1 and batch_size=2).

## Root cause

### The buggy code

All three affected models share the same code in `prepare_inputs_for_generation`:

**`modeling_idefics3.py`, lines 935–937** (identical in `modeling_idefics2.py:1161` and `modeling_smolvlm.py:908`):

```python
def prepare_inputs_for_generation(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    cache_position=None,
    pixel_values=None,
    pixel_attention_mask=None,
    image_hidden_states=None,
    logits_to_keep=None,
    is_first_iteration=False,
    **kwargs,
):
    model_inputs = super().prepare_inputs_for_generation(
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        pixel_values=pixel_values,
        pixel_attention_mask=pixel_attention_mask,
        image_hidden_states=image_hidden_states,
        logits_to_keep=logits_to_keep,
        is_first_iteration=is_first_iteration,
        **kwargs,
    )

    if image_hidden_states is not None or not is_first_iteration:  # <-- BUG
        model_inputs["pixel_values"] = None
        model_inputs["pixel_attention_mask"] = None

    return model_inputs
```

The condition `not is_first_iteration` unconditionally drops `pixel_values` after step 1, regardless of whether a KV cache exists to preserve the image information.

### Why this breaks generation with `use_cache=False`

Autoregressive generation calls `prepare_inputs_for_generation` before every forward pass. Here's what happens at each step:

**Step 1** (`is_first_iteration=True`):
- `pixel_values` is passed through to `forward()`
- `Idefics3Model.forward()` runs the vision encoder, producing `image_hidden_states`
- `inputs_merger` replaces `<image>` token embeddings with actual image features
- Model sees the image — first token is correct

**Step 2+** (`is_first_iteration=False`):
- `prepare_inputs_for_generation` sets `pixel_values = None` (the bug)
- With `use_cache=True`: doesn't matter — image KV entries from step 1 are in the cache
- With `use_cache=False`: **no cache exists**, and `pixel_values` is `None`, so in `Idefics3Model.forward()`:
  ```python
  # line 686-706 of modeling_idefics3.py
  if inputs_embeds is None:
      inputs_embeds = self.text_model.get_input_embeddings()(input_ids)

  if pixel_values is not None:           # <-- False! pixel_values was set to None
      image_hidden_states = self.get_image_features(...)  # SKIPPED

  if image_hidden_states is not None:    # <-- Also False!
      inputs_embeds = self.inputs_merger(...)  # SKIPPED
  ```
- The `<image>` token positions retain their raw token embeddings (a single learned placeholder vector with no image content)
- The model effectively sees no image — output is garbage

### How correct models handle this

**43 out of 46** VLMs in transformers handle this correctly, using one of two equivalent patterns:

**Pattern A** — keep pixel_values when no cache (used by LLaVA, LLaVA-Next, Aria, Aya Vision, Mistral3, InternVL, etc. — 28 models):

```python
# modeling_llava.py, line 429
if is_first_iteration or not kwargs.get("use_cache", True):
    model_inputs["pixel_values"] = pixel_values
```

**Pattern B** — drop pixel_values only when cache is active (used by Qwen2-VL, Qwen3-VL, Chameleon, GLM4V, etc. — 15 models):

```python
# modeling_qwen2_vl.py
if not is_first_iteration and use_cache:
    model_inputs["pixel_values"] = None
```

Both patterns are logically equivalent: pixel_values are preserved whenever `use_cache=False`, ensuring the vision encoder re-runs at every step (which is the inherent cost of not using a cache — same as re-computing text KV states).

### Full audit results

| # | Model | Status | Pattern |
|---|-------|--------|---------|
| 1 | aria | CORRECT | `is_first_iteration or not use_cache` |
| 2 | aya_vision | CORRECT | `is_first_iteration or not use_cache` |
| 3 | chameleon | CORRECT | `not is_first_iteration and use_cache` |
| 4 | cohere2_vision | CORRECT | `is_first_iteration or not use_cache` |
| 5 | deepseek_vl | CORRECT | `is_first_iteration or not use_cache` |
| 6 | deepseek_vl_hybrid | CORRECT | `is_first_iteration or not use_cache` |
| 7 | ernie4_5_vl_moe | CORRECT | `not is_first_iteration and use_cache` |
| 8 | fast_vlm | CORRECT | `is_first_iteration or not use_cache` |
| 9 | fuyu | CORRECT | `not is_first_iteration and use_cache` |
| 10 | gemma3 | CORRECT | `is_first_iteration or not use_cache` |
| 11 | gemma3n | CORRECT | `is_first_iteration or not use_cache` |
| 12 | git | CORRECT | `is_first_iteration or not use_cache` |
| 13 | glm46v | CORRECT | `not is_first_iteration and use_cache` |
| 14 | glm4v | CORRECT | `not is_first_iteration and use_cache` |
| 15 | glm4v_moe | CORRECT | `not is_first_iteration and use_cache` |
| 16 | glm_ocr | CORRECT | `not is_first_iteration and use_cache` |
| 17 | got_ocr2 | CORRECT | `is_first_iteration or not use_cache` |
| 18 | **idefics2** | **BUGGY** | `not is_first_iteration` (no use_cache check) |
| 19 | **idefics3** | **BUGGY** | `not is_first_iteration` (no use_cache check) |
| 20 | internvl | CORRECT | `is_first_iteration or not use_cache` |
| 21 | janus | CORRECT | `is_first_iteration or not use_cache` |
| 22 | kosmos2 | CORRECT | `not is_first_iteration and use_cache` |
| 23 | lfm2_vl | CORRECT | `is_first_iteration or not use_cache` |
| 24 | lighton_ocr | CORRECT | `is_first_iteration or not use_cache` |
| 25 | llama4 | CORRECT | `is_first_iteration or not use_cache` |
| 26 | llava | CORRECT | `is_first_iteration or not use_cache` |
| 27 | llava_next | CORRECT | `is_first_iteration or not use_cache` |
| 28 | llava_next_video | CORRECT | `is_first_iteration or not use_cache` |
| 29 | llava_onevision | CORRECT | `is_first_iteration or not use_cache` |
| 30 | mistral3 | CORRECT | `is_first_iteration or not use_cache` |
| 31 | mllama | CORRECT | `not is_first_iteration and use_cache` |
| 32 | ovis2 | CORRECT | `is_first_iteration or not use_cache` |
| 33 | paddleocr_vl | CORRECT | `not is_first_iteration and use_cache` |
| 34 | paligemma | CORRECT | `is_first_iteration or not use_cache` |
| 35 | perception_lm | CORRECT | `is_first_iteration or not use_cache` |
| 36 | qwen2_5_vl | CORRECT | `not is_first_iteration and use_cache` |
| 37 | qwen2_vl | CORRECT | `not is_first_iteration and use_cache` |
| 38 | qwen3_5 | CORRECT | `not is_first_iteration and use_cache` |
| 39 | qwen3_5_moe | CORRECT | `not is_first_iteration and use_cache` |
| 40 | qwen3_vl | CORRECT | `not is_first_iteration and use_cache` |
| 41 | qwen3_vl_moe | CORRECT | `not is_first_iteration and use_cache` |
| 42 | **smolvlm** | **BUGGY** | `not is_first_iteration` (no use_cache check) |
| 43 | video_llama_3 | CORRECT | `not is_first_iteration and use_cache` |
| 44 | video_llava | CORRECT | `is_first_iteration or not use_cache` |
| 45 | vipllava | CORRECT | `is_first_iteration or not use_cache` |

## Proposed fix

The fix is a one-line change in each of the three affected files. The `not is_first_iteration` condition needs an additional `use_cache` guard, matching the pattern used by every other VLM in the codebase:

### `modeling_idefics2.py` (line ~1161)

```diff
-        if image_hidden_states is not None or not is_first_iteration:
+        if image_hidden_states is not None or (not is_first_iteration and kwargs.get("use_cache", True)):
             model_inputs["pixel_values"] = None
             model_inputs["pixel_attention_mask"] = None
```

### `modeling_idefics3.py` (line 935)

```diff
-        if image_hidden_states is not None or not is_first_iteration:
+        if image_hidden_states is not None or (not is_first_iteration and kwargs.get("use_cache", True)):
             model_inputs["pixel_values"] = None
             model_inputs["pixel_attention_mask"] = None
```

### `modeling_smolvlm.py` (line ~908)

```diff
-        if image_hidden_states is not None or not is_first_iteration:
+        if image_hidden_states is not None or (not is_first_iteration and kwargs.get("use_cache", True)):
             model_inputs["pixel_values"] = None
             model_inputs["pixel_attention_mask"] = None
```

This preserves the optimization of skipping the vision encoder on subsequent steps when the KV cache is active, while correctly re-processing `pixel_values` when there is no cache to fall back on.

## Environment

- `transformers` version: 5.3.0
- `torch` version: 2.10.0+cu128
- Python: 3.11
- GPU: CUDA-capable GPU
