# TRL v0.29 DPO Dataset Column Names Changed

## Breaking Change

TRL v0.29 renamed the expected pre-tokenized DPO dataset columns:
- `prompt_input_ids` -> `prompt_ids`
- `chosen_input_ids` -> `chosen_ids`
- `rejected_input_ids` -> `rejected_ids`

If the old column names are present, `DPOTrainer` raises:
```
ValueError: No columns in the dataset match the model's forward method signature:
(prompt_ids, chosen_ids, rejected_ids, ...).
The following columns have been ignored: [chosen_input_ids, prompt_input_ids, rejected_input_ids].
```

## Root Cause

TRL v0.29 simplified the column naming convention for preference datasets. The `_input_ids` suffix was dropped to make column names shorter and more consistent across trainer types.

## How to Reproduce

```python
from trl import DPOTrainer
import datasets

# Create dataset with old column names — fails with TRL>=0.29
ds = datasets.Dataset.from_dict({
    "prompt_input_ids": [[1, 2, 3]],
    "chosen_input_ids": [[4, 5]],
    "rejected_input_ids": [[6, 7]],
})
# DPOTrainer will reject this dataset
```

## Files Changed

- `src/oumi/core/datasets/base_dpo_dataset.py` — `_process_sample()` return keys
- `src/oumi/core/datasets/vision_language_dpo_dataset.py` — VLM DPO output keys

## Fix Applied

Conditionally return the correct column names based on TRL version using `is_trl_v0_29_or_later()` (new helper added to `oumi.utils.packaging`).

Before:
```python
return {
    "prompt_input_ids": prompt_input_ids,
    "chosen_input_ids": chosen_input_ids,
    "rejected_input_ids": rejected_input_ids,
}
```

After:
```python
from oumi.utils.packaging import is_trl_v0_29_or_later

if is_trl_v0_29_or_later():
    return {
        "prompt_ids": prompt_input_ids,
        "chosen_ids": chosen_input_ids,
        "rejected_ids": rejected_input_ids,
    }
return {
    "prompt_input_ids": prompt_input_ids,
    "chosen_input_ids": chosen_input_ids,
    "rejected_input_ids": rejected_input_ids,
}
```
