# vLLM GuidedDecodingParams Renamed to StructuredOutputsParams

## Breaking Change

In vLLM v0.17, the guided decoding API was renamed:
- `vllm.sampling_params.GuidedDecodingParams` -> `vllm.sampling_params.StructuredOutputsParams`
- `SamplingParams(guided_decoding=...)` -> `SamplingParams(structured_outputs=...)`
- `GuidedDecodingParams.from_optional(...)` was removed; use direct construction instead

Importing `GuidedDecodingParams` from `vllm.sampling_params` raises `ImportError`.

## Root Cause

vLLM v0.17 renamed "guided decoding" to "structured outputs" to better reflect the feature's purpose. The `from_optional` class method was also removed — the class is now a pydantic dataclass that accepts keyword arguments directly.

## How to Reproduce

```python
# Fails on vLLM>=0.17
from vllm.sampling_params import GuidedDecodingParams
# ImportError: cannot import name 'GuidedDecodingParams'

# Works on vLLM>=0.17
from vllm.sampling_params import StructuredOutputsParams

# Fails on vLLM>=0.17
params = GuidedDecodingParams.from_optional(json=schema)
# Works on vLLM>=0.17
params = StructuredOutputsParams(json=schema)

# Fails on vLLM>=0.17
SamplingParams(guided_decoding=params)
# Works on vLLM>=0.17
SamplingParams(structured_outputs=params)
```

## Files Changed

- `src/oumi/utils/packaging.py` — added `is_vllm_v017_or_later()` helper
- `src/oumi/inference/vllm_inference_engine.py` — conditional import and usage
- `tests/unit/inference/test_vllm_inference_engine.py` — version-aware assertions

## Fix Applied

Added `is_vllm_v017_or_later()` version helper and used it for backward-compatible imports and kwarg selection.

```python
# Import:
try:
    from vllm.sampling_params import GuidedDecodingParams as VLLMGuidedDecodingParams
except ImportError:
    from vllm.sampling_params import StructuredOutputsParams as VLLMGuidedDecodingParams

# Construction (direct constructor works in both vLLM 0.10 and 0.17):
guided_decoding = VLLMGuidedDecodingParams(json=..., regex=..., choice=...)

# SamplingParams kwarg:
if is_vllm_v017_or_later():
    SamplingParams(..., structured_outputs=guided_decoding)
else:
    SamplingParams(..., guided_decoding=guided_decoding)
```
