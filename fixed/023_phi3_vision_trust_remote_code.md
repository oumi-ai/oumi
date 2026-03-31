# Phi-3-vision trust_remote_code Not Forwarded to Processor

## Status: FIXED

## Issue

The Phi-3-vision test fails with:
```
ValueError: The repository microsoft/Phi-3-vision-128k-instruct contains custom code which must be executed to correctly load the model. You can inspect the repository content at https://hf.co/microsoft/Phi-3-vision-128k-instruct.
```

### Affected Test

- `test_sft_vision_datasets_load_datasets.py::test_build_dataset_mixture[merve/vqav2-small microsoft/Phi-3-vision-128k-instruct]`

This was previously masked by the Pydantic serialization error (issue #018). In transformers v5, the `trust_remote_code` requirement is enforced more strictly.

## Root Cause

The `trust_remote_code=True` parameter is set in the test's `LoadDatasetInfo` (line 88) and correctly passed through the dataset builder. However, it does not reach the `build_processor()` call inside the feature generator.

### Parameter Flow (expected)

```
Test config (trust_remote_code=True)
  → DatasetParams.trust_remote_code
    → _load_dataset() in src/oumi/builders/data.py (line 286)
      → VisionLanguageSftDataset.__init__() (line 113)
        → VisionLanguageConversationFeatureGenerator.__init__() (line 142)
          → build_processor(trust_remote_code=trust_remote_code)
```

### Where the Chain Breaks

**File**: `src/oumi/core/feature_generators/vision_language_conversation_feature_generator.py`, line 139

The feature generator calls `build_processor()`:
```python
processor = build_processor(
    processor_name,
    tokenizer,
    trust_remote_code=trust_remote_code,
    processor_kwargs=processor_kwargs,
)
```

The `trust_remote_code` parameter here comes from the feature generator's `__init__` parameter (line 85):
```python
def __init__(
    self,
    ...
    trust_remote_code: bool = False,
    ...
):
```

The default is `False`. The value must be explicitly passed from `VisionLanguageSftDataset.__init__()`.

**File**: `src/oumi/core/datasets/vision_language_dataset.py`, line 113

```python
self._feature_generator = VisionLanguageConversationFeatureGenerator(
    ...
    trust_remote_code=trust_remote_code,
    ...
)
```

This correctly passes the parameter. But the `trust_remote_code` value at this point depends on what was passed to `VisionLanguageSftDataset.__init__()`.

**File**: `src/oumi/core/datasets/base_sft_dataset.py`, lines 55-60

```python
def __init__(self, *, dataset_name=None, ..., **kwargs) -> None:
    super().__init__(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        split=split,
        **kwargs,
    )
```

`BaseSftDataset.__init__` does NOT explicitly list `trust_remote_code` as a parameter — it relies on `**kwargs` to forward it. This should work but makes the flow opaque.

**File**: `src/oumi/core/datasets/base_map_dataset.py`, line 70-80

```python
def __init__(
    self,
    *,
    dataset_name: str | None = None,
    ...
    trust_remote_code: bool = False,
    **kwargs,
):
```

`BaseMapDataset.__init__` has `trust_remote_code` with default `False`. If `trust_remote_code` is absorbed by `BaseMapDataset` and NOT forwarded to child class attributes, the `VisionLanguageSftDataset` may not see the correct value.

### Actual Root Cause Found

The break was NOT in the inheritance chain — it was in `DatasetParams.__post_init__`.

**File**: `src/oumi/core/configs/params/data_params.py`, lines 184-190

```python
if self.trust_remote_code:
    warnings.warn(
        "`trust_remote_code` is deprecated and will be removed in the future.",
        DeprecationWarning,
        stacklevel=2,
    )
    self.trust_remote_code = False  # ← SILENTLY RESETS TO FALSE
```

The `DatasetParams` dataclass has a `__post_init__` that emits a deprecation warning and then **forcibly resets** `trust_remote_code` to `False`. This means regardless of what the test or config sets, the value is always `False` by the time `_load_dataset()` reads it.

The deprecation was intended for HF dataset loading (datasets v4+ removed loading scripts), but `DatasetParams.trust_remote_code` also flows to the processor build pipeline, where it's still needed for models with custom processor code (e.g., Phi-3-vision).

## Fix

**File**: `src/oumi/core/configs/params/data_params.py`

1. Removed the deprecation warning and forced reset entirely — the parameter is still needed for processors
2. Updated the docstring to clarify that while HF datasets no longer need it, processors still do:

```python
trust_remote_code: bool = False
"""Whether to trust remote code when loading the dataset and processor.

Note: While HuggingFace datasets no longer require this parameter (dataset
loading scripts were removed in datasets v4+), it is still needed for
processors that contain custom code (e.g., Phi-3-vision).
"""
```

## Verification

Test passes:
```
1 passed, 2 warnings in 67.49s
```

## Impact

Medium — affects any vision model that requires `trust_remote_code=True` for its processor (Phi-3-vision, some Llava variants). Models that don't require custom code (SmolVLM, standard HF models) are unaffected.

## Notes

- This is a transformers v5 regression — earlier versions did not strictly enforce `trust_remote_code` for processor loading
- The deprecation of `trust_remote_code` should be revisited — it's still required for models with custom processor code
