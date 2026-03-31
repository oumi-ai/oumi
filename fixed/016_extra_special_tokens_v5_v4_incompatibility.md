# extra_special_tokens Cross-Version Incompatibility (transformers v5 vs v4)

## Breaking Change

A tokenizer saved by transformers v5 cannot be loaded by transformers v4. Loading fails with:

```
AttributeError: 'list' object has no attribute 'keys'
```

The reverse direction (v4 → v5) works fine. This affects any checkpoint saved with merged LoRA weights (or any saved model) when the transformers version is downgraded between training and inference.

## Compatibility Matrix

| Saved by | Loaded by | Result |
|----------|-----------|--------|
| v4       | v4        | OK |
| v4       | v5        | OK — v5 converts `additional_special_tokens` → `extra_special_tokens` |
| v5       | v5        | OK |
| v5       | v4        | **BROKEN** — two issues (see below) |

## Root Cause

Transformers v5 changed how special tokens are serialized in `tokenizer_config.json`:

### What v4 saves

```json
{
  "additional_special_tokens": ["<|im_start|>", "<|im_end|>", ...],
  "extra_special_tokens": {}
}
```

- `additional_special_tokens`: a **list** of token strings (via `special_tokens_map` property which includes `SPECIAL_TOKENS_ATTRIBUTES`)
- `extra_special_tokens`: an empty **dict** (for model-specific named tokens)

### What v5 saves

```json
{
  "extra_special_tokens": ["<|im_start|>", "<|im_end|>", ...]
}
```

- `extra_special_tokens`: a **list** of token strings (the old `additional_special_tokens` merged in)
- `additional_special_tokens`: **not saved** (v5's `special_tokens_map` property excludes it)

### Why v5 → v4 loading breaks

Two failures occur when v4 loads a v5-saved tokenizer:

**Failure 1: Type mismatch (crashes)**

In v4's `__init__`:
```python
self.extra_special_tokens = kwargs.pop("extra_special_tokens", {})  # gets a list
self._set_model_specific_special_tokens(special_tokens=self.extra_special_tokens)
```

`_set_model_specific_special_tokens` calls `special_tokens.keys()`, but a list has no `.keys()` method → `AttributeError`.

**Failure 2: Missing tokens (silent data loss)**

Even if Failure 1 were patched, `additional_special_tokens` is absent from the v5-saved config. v4 uses `additional_special_tokens` to track tokens like `<|im_start|>`, `<|vision_start|>`, etc. Without it, these tokens would not be recognized as special tokens, affecting decoding behavior (e.g., `skip_special_tokens=True` would not skip them).

### Why v4 → v5 loading works

In v5's `from_pretrained`:
```python
# V5: Convert deprecated additional_special_tokens to extra_special_tokens
if "additional_special_tokens" in init_kwargs:
    init_kwargs.setdefault("extra_special_tokens", init_kwargs.pop("additional_special_tokens"))
```

v5 explicitly handles the v4 format by converting `additional_special_tokens` to `extra_special_tokens`.

## How to Reproduce

```python
# Save with transformers v5
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
tok.save_pretrained("/tmp/saved_tokenizer")

# Load with transformers v4 (after downgrade)
tok = AutoTokenizer.from_pretrained("/tmp/saved_tokenizer")
# AttributeError: 'list' object has no attribute 'keys'
```

## Files Changed

This is an **upstream transformers issue**, not an oumi bug. No oumi code changes are strictly required.

## Workaround

Re-save the tokenizer from the base model (loaded with the current transformers version) after merging:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
tok.save_pretrained("output/merged_model")
```

## Potential Defensive Fix

The merged save path in `src/oumi/core/trainers/hf_trainer.py` could explicitly re-save the tokenizer after the merge step. This would ensure the tokenizer is always serialized in the format expected by the currently installed transformers version:

```python
# After merged_model.save_pretrained(output_dir)
tokenizer = self._hf_trainer.tokenizer
if tokenizer is not None:
    tokenizer.save_pretrained(output_dir)
```
