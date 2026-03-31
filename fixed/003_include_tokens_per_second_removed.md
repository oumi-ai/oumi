# include_tokens_per_second Removed from TrainingArguments

## Breaking Change

The `include_tokens_per_second` parameter was removed from `transformers.TrainingArguments` in transformers v5.0.0. Passing it raises `TypeError: __init__() got an unexpected keyword argument 'include_tokens_per_second'`.

## Root Cause

In transformers v5, the performance metrics reporting was refactored. The `include_tokens_per_second` flag was removed because tokens-per-second tracking is now handled differently (or always enabled by default).

## How to Reproduce

```python
from transformers import TrainingArguments
# Fails on transformers>=5.0.0
args = TrainingArguments(output_dir="/tmp", include_tokens_per_second=True)
# TypeError: __init__() got an unexpected keyword argument 'include_tokens_per_second'
```

## Files Changed

- `src/oumi/core/configs/params/training_params.py`

## Fix Applied

Conditionally pass `include_tokens_per_second` only when running on transformers v4 using `is_transformers_v5()`.

Before:
```python
result = config_class(
    ...
    include_tokens_per_second=self.include_performance_metrics,
    include_num_input_tokens_seen=self.include_performance_metrics,
    ...
)
```

After:
```python
result = config_class(
    ...
    # include_tokens_per_second was removed in transformers v5.
    **(
        {"include_tokens_per_second": self.include_performance_metrics}
        if not is_transformers_v5()
        else {}
    ),
    include_num_input_tokens_seen=self.include_performance_metrics,
    ...
)
```
