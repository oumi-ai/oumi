# vLLM LLM.set_tokenizer() Removed

## Breaking Change

`LLM.set_tokenizer()` was removed from vLLM. Calling it raises `AttributeError: 'LLM' object has no attribute 'set_tokenizer'`. Only `LLM.get_tokenizer()` remains.

The method was removed earlier than v0.17 — it is already absent in v0.14.0. The original fix guarded with `is_vllm_v017_or_later()` which was insufficient.

## Root Cause

vLLM removed the ability to externally set the tokenizer on an `LLM` instance. The tokenizer is now managed internally by the engine and configured at construction time.

## How to Reproduce

```python
import vllm
llm = vllm.LLM(model="some-model")
# Fails on vLLM>=0.14
llm.set_tokenizer(my_tokenizer)
# AttributeError: 'LLM' object has no attribute 'set_tokenizer'
```

## Files Changed

- `src/oumi/inference/vllm_inference_engine.py`

## Fix Applied

Use `hasattr` check instead of version-based gating, since the exact removal version is unclear and the method may have been removed across different vLLM release lines:

```python
self._llm = vllm.LLM(**final_vllm_kwargs)
# set_tokenizer was removed in vLLM v0.14+.
if hasattr(self._llm, "set_tokenizer"):
    self._llm.set_tokenizer(self._tokenizer)
```
