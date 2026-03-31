# KTOConfig max_prompt_length Removed in TRL v0.29

## Breaking Change

The `max_prompt_length` parameter was removed from `trl.KTOConfig` in TRL v0.29. Passing it raises `TypeError: KTOConfig.__init__() got an unexpected keyword argument 'max_prompt_length'`.

## Root Cause

TRL v0.29 simplified the KTO trainer configuration. The `max_prompt_length` parameter was removed, leaving only `max_length` for controlling sequence lengths. KTO was also moved to `trl.experimental.kto` (still accessible from `trl` top-level with a deprecation warning).

## How to Reproduce

```python
from trl import KTOConfig
# Fails with TRL>=0.29
config = KTOConfig(output_dir="/tmp", max_prompt_length=128)
# TypeError: KTOConfig.__init__() got an unexpected keyword argument 'max_prompt_length'
```

## Files Changed

- `tests/integration/train/test_train.py` — `test_train_kto` test configuration

## Fix Applied

Conditionally include `max_prompt_length` only when running on TRL < 0.29 using `is_trl_v0_29_or_later()`.

Before:
```python
trainer_kwargs={
    "max_length": 512,
    "max_prompt_length": 128,
    "remove_unused_columns": False,
    "desirable_weight": 0.8,
},
```

After:
```python
trainer_kwargs={
    "max_length": 512,
    # max_prompt_length was removed in TRL v0.29.
    **(
        {}
        if is_trl_v0_29_or_later()
        else {"max_prompt_length": 128}
    ),
    "remove_unused_columns": False,
    "desirable_weight": 0.8,
},
```
