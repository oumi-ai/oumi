# multimodal-open-r1-8k-verified Test Failure — Wrong Split

## Status: UNFIXED (pre-existing, previously masked)

## Issue

The auto-discovered test for `lmms-lab/multimodal-open-r1-8k-verified` fails with:
```
ValueError: Unknown split "test". Should be one of ['train'].
```

### Affected Test

- `test_sft_vision_datasets_load_datasets.py::test_build_dataset_mixture[lmms-lab/multimodal-open-r1-8k-verified HuggingFaceTB/SmolVLM-256M-Instruct]`

This was previously masked by the Pydantic serialization error (issue #018).

## Root Cause

### The Dataset Class

**File**: `src/oumi/datasets/vision_language/lmms_lab_multimodal_open_r1.py`

```python
@register_dataset("lmms-lab/multimodal-open-r1-8k-verified")
class LmmsLabMultimodalOpenR1Dataset(HuggingFaceVisionDataset):
```

The HuggingFace dataset `lmms-lab/multimodal-open-r1-8k-verified` only has a `train` split. There is no `test` or `validation` split.

### How the Test Applies the Wrong Default

**File**: `tests/integration/datasets/test_sft_vision_datasets_load_datasets.py`

Line 21 (note the typo in the variable name):
```python
_DEFALT_DATASET_SPLIT: Final[str] = "test"
```

Line 156-164 — auto-discovered datasets use this default:
```python
result.append(
    LoadDatasetInfo(
        dataset_name=dataset_name,
        ...
        dataset_split=_DEFALT_DATASET_SPLIT,  # Always "test"
        ...
    )
)
```

The `_load_hf_hub_dataset()` method in `src/oumi/core/datasets/base_map_dataset.py` (line 469) passes this split to `datasets.load_dataset()`, which raises the ValueError.

## Recommended Fix

**Option A (recommended)**: Add an explicit `LoadDatasetInfo` entry with the correct split, and exclude from auto-discovery:

**File**: `tests/integration/datasets/test_sft_vision_datasets_load_datasets.py`

Add to the manual list in `_get_all_sft_vision_dataset_infos()` (around line 138):
```python
LoadDatasetInfo(
    dataset_name="lmms-lab/multimodal-open-r1-8k-verified",
    model_name=_DEFAULT_MODEL_NAME,
    dataset_split="train",
    chat_template="auto",
    trust_remote_code=True,
    max_rows=32,
    expected_rows=32,
),
```

The auto-discovery exclusion happens automatically because the manual list entries are already filtered out at lines 141-148.

**Option B**: Add split auto-detection in the auto-discovery loop:

```python
import datasets as hf_datasets

def _get_preferred_split(dataset_name: str) -> str:
    try:
        info = hf_datasets.get_dataset_config_info(dataset_name)
        for preferred in ["test", "validation", "train"]:
            if preferred in info.splits:
                return preferred
        return list(info.splits.keys())[0]
    except Exception:
        return "test"  # fallback to current default
```

This generalizes to future datasets but adds network calls during test collection.

**Option C**: Change the default split from `"test"` to `"train"`:

```python
_DEFALT_DATASET_SPLIT: Final[str] = "train"
```

Simple but may break datasets that only have a `test` split.

## Impact

Low — this is one specific dataset. The dataset is still properly tested when invoked with the correct split via config files.
