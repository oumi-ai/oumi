# AutoModelForVision2Seq Renamed to AutoModelForImageTextToText

## Breaking Change

`transformers.AutoModelForVision2Seq` was renamed to `transformers.AutoModelForImageTextToText` in transformers v5.0.0. All imports and references to `AutoModelForVision2Seq` will fail with `ImportError` or `AttributeError`.

## Root Cause

HuggingFace renamed this auto class to better reflect its purpose (models that take image+text input and produce text output). The old name suggested "vision to sequence" which was less descriptive.

## How to Reproduce

```python
# Fails on transformers>=5.0.0
from transformers import AutoModelForVision2Seq
# ImportError: cannot import name 'AutoModelForVision2Seq' from 'transformers'
```

## Files Changed

- `src/oumi/core/configs/internal/supported_models.py` — VLM default model class
- `src/oumi/utils/verl_model_merger.py` — import and usage in model type detection
- `tests/unit/utils/test_torch_naming_heuristics.py` — test parametrization

## Fix Applied

Used `is_transformers_v5()` to conditionally import/select the correct class, so both v4 and v5 work.

Before:
```python
from transformers import AutoModelForVision2Seq
default_vlm_class = transformers.AutoModelForVision2Seq
```

After:
```python
from oumi.utils.packaging import is_transformers_v5

# At import time:
if is_transformers_v5():
    from transformers import AutoModelForImageTextToText
else:
    from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText

# At usage sites:
if is_transformers_v5():
    default_vlm_class = transformers.AutoModelForImageTextToText
else:
    default_vlm_class = transformers.AutoModelForVision2Seq
```
