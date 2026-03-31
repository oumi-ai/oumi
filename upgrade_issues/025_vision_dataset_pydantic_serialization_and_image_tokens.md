# Vision Dataset Serialization and Image Token Mismatch

## Status: FIXED

## Issue

All 12 SmolVLM vision dataset integration tests in `test_sft_vision_datasets_load_datasets.py` failed with two overlapping errors:

1. **Pydantic serialization warnings**: `PydanticSerializationUnexpectedValue(Expected 3 fields but got 2)` and `Expected str` when serializing `ContentItem` and `Message` objects.
2. **Image token mismatch**: `ValueError: The total number of <image> tokens in the prompts should be the same as the number of images passed. Found 0 <image> tokens and 1 images.`
3. **Jinja template error**: `jinja2.exceptions.UndefinedError: str object has no element 0` for datasets with mixed string/list content in messages.

### Affected Tests

All `test_build_dataset_mixture[* HuggingFaceTB/SmolVLM-256M-Instruct]` parameterized tests:
- `merve/vqav2-small`, `HuggingFaceM4/Docmatix`, `HuggingFaceM4/the_cauldron`
- `allenai/pixmo-ask-model-anything`, `allenai/pixmo-cap`, `allenai/pixmo-cap-qa`
- `mnist_sft`, `hiyouga/geometry3k`, `lmms-lab/multimodal-open-r1-8k-verified`
- `huggingfaceh4/llava-instruct-mix-vsft`, `hf_vision`

## Root Cause

Three interconnected issues in the serialization and template pipeline:

### 1. `_convert_messages_to_dicts` output format (primary cause)

**File**: `src/oumi/core/processors/default_processor.py`, line 229

The method used `msg.model_dump(mode="json", exclude_none=True, exclude_unset=True)` which produced oumi's internal format:
```json
{"type": "image_binary", "binary": "base64...", "content": null}
{"type": "text", "content": "Hello"}
```

But HuggingFace processor chat templates expect:
```json
{"type": "image"}
{"type": "text", "text": "Hello"}
```

Key differences:
- oumi uses `type: "image_binary"` / `"image_url"` / `"image_path"` → HF expects `type: "image"`
- oumi uses `content: "text"` → HF expects `text: "text"`
- oumi serializes binary data and includes nullable fields → HF expects minimal image markers

### 2. Chat template mismatch for SmolVLM

**File**: `tests/integration/datasets/test_sft_vision_datasets_load_datasets.py`, line 23

The test used `_DEFAULT_CHAT_TEMPLATE = "qwen2-vl-instruct"` for all models including SmolVLM. The oumi qwen2-vl-instruct template was hand-crafted to handle oumi's internal format (uses `startswith('image')` and fallback `item['content']`), but it generates qwen2-vl-specific tokens (`<|vision_start|><|image_pad|><|vision_end|>`). The SmolVLM processor's `__call__` method expects `<image>` tokens, causing the mismatch.

### 3. String content in mixed conversations

**File**: `src/oumi/datasets/vision_language/pixmo_cap_qa.py`, lines 46-63

Some datasets (pixmo-cap-qa, pixmo-cap, etc.) create conversations with mixed content types:
- First message: `content=[ContentItem(type=IMAGE_URL, ...)]` (list)
- Subsequent messages: `content="question text"` (string)

SmolVLM's built-in chat template does `message['content'][0]['type']` on ALL messages. When content is a string, `"text"[0]` gives `"t"`, and `"t"['type']` raises `jinja2.exceptions.UndefinedError`.

### 4. Pydantic binary field serializer (secondary)

**File**: `src/oumi/core/types/conversation.py`, line 156

The `_encode_binary` field serializer converted `None` to `""` even when `exclude_none=True` was used, causing Pydantic to warn about field count mismatch (expected 3 fields, got 2 after exclusion).

## Fix

### 1. Rewrote `_convert_messages_to_dicts` to output HF-compatible format

**File**: `src/oumi/core/processors/default_processor.py`

```python
def _convert_messages_to_dicts(self, messages: list[Message]) -> list[dict]:
    result = []
    for msg in messages:
        msg_dict: dict = {"role": str(msg.role)}
        if isinstance(msg.content, str):
            msg_dict["content"] = [{"type": "text", "text": msg.content}]
        else:
            content_list = []
            for item in msg.content:
                if item.is_image():
                    content_list.append({"type": "image"})
                elif item.is_text():
                    content_list.append({"type": "text", "text": item.content or ""})
            msg_dict["content"] = content_list
        result.append(msg_dict)
    return result
```

Key changes:
- String content always converted to `[{"type": "text", "text": "..."}]` for HF template compatibility
- Image types normalized to `{"type": "image"}` regardless of oumi's internal type (image_binary, image_url, image_path)
- Text content uses `text` key instead of `content` key

### 2. Changed default chat template to `"auto"` for SmolVLM

**File**: `tests/integration/datasets/test_sft_vision_datasets_load_datasets.py`, line 23

```python
_DEFAULT_CHAT_TEMPLATE: Final[str] = "auto"
```

This lets each model use its own built-in chat template, generating the correct image tokens.

### 3. Updated 6 text-only oumi chat templates

Since content is now always a list, text-only templates that used `message['content']` directly needed updating to handle both string and list formats:

**Files modified**:
- `src/oumi/datasets/chat_templates/chat_ml.jinja`
- `src/oumi/datasets/chat_templates/default.jinja`
- `src/oumi/datasets/chat_templates/default_gen.jinja`
- `src/oumi/datasets/chat_templates/gemma2-it.jinja`
- `src/oumi/datasets/chat_templates/gpt2.jinja`
- `src/oumi/datasets/chat_templates/zephyr.jinja`

Each template now checks `message['content'] is string` before accessing content directly, and iterates through list items to extract text when content is a list.

Templates that already handled both formats (qwen2-vl-instruct, qwen3-vl-instruct, phi3-instruct, llava, molmo, internvl3, llama3-instruct) were not modified.

### 4. Added `when_used="unless-none"` to binary serializer

**File**: `src/oumi/core/types/conversation.py`, line 156

```python
@pydantic.field_serializer("binary", when_used="unless-none")
def _encode_binary(self, value: bytes | None) -> str:
```

This tells Pydantic to skip the serializer when value is None, allowing proper field exclusion with `exclude_none=True`.

## Verification

- **10 of 12 SmolVLM tests now pass** (2 remaining failures are pre-existing dataset config issues — see issues #020 and #021)
- **Unit tests**: 4110 passed (up from 4087 — 23 template-related tests now pass that were previously failing)
- **All processor unit tests pass** including `test_processor_convert_messages_to_dicts` and `test_processor_apply_chat_template_with_message_objects`

## Notes

- The `Phi-3-vision` test also had this Pydantic error but now fails with a different `trust_remote_code` issue (see issue #022)
- The `when_used="unless-none"` fix suppresses the Pydantic warning but the real fix was the `_convert_messages_to_dicts` rewrite which avoids `model_dump` entirely
