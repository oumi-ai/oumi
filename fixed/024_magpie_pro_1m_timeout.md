# Magpie-Pro-1M Full Epoch Test Timeout

## Status: UNFIXED (pre-existing)

## Issue

The full-epoch test for the Magpie-Pro-1M dataset times out:
```
FAILED test_sft_datasets_full_epoch.py::test_dataset_conversation[Magpie-Align/Llama-3-Magpie-Pro-1M-v0.1]
Timeout (>120.0s)
```

### Affected Test

- `tests/integration/datasets/test_sft_datasets_full_epoch.py::test_dataset_conversation[Magpie-Align/Llama-3-Magpie-Pro-1M-v0.1]`

## Root Cause

### The Dataset

**File**: `src/oumi/datasets/sft/magpie.py`, lines 42-66

```python
@register_dataset("Magpie-Align/Llama-3-Magpie-Pro-1M-v0.1")
class MagpieProDataset(BaseSftDataset):
    default_dataset = "Magpie-Align/Llama-3-Magpie-Pro-1M-v0.1"
```

The dataset contains approximately **1 million rows**.

### The Test

**File**: `tests/integration/datasets/test_sft_datasets_full_epoch.py`, lines 44-92

The test iterates through EVERY row in the dataset:
```python
@pytest.mark.e2e
def test_dataset_conversation(dataset_fixture):
    dataset = dataset_fixture
    for idx in range(len(dataset)):
        conversation = dataset.conversation(idx)
        # ... validates each conversation
```

### The Loading Process

**File**: `src/oumi/core/datasets/base_map_dataset.py`, lines 466-505

`_load_hf_hub_dataset()` calls `datasets.load_dataset()` and converts ALL data to pandas via `.to_pandas()`. For 1M rows:
1. Download/cache: ~3.4GB dataset
2. Convert to pandas: loads all rows into memory
3. Iterate + transform: `dataset.conversation(idx)` for each of 1M rows

This exceeds the 120-second timeout (and likely even a 300-second timeout).

### Timeout Configuration

The test file has no explicit `@pytest.mark.timeout()` marker. The timeout comes from the test runner configuration (likely `--timeout=120` or a pytest-timeout default).

## Recommended Fix

**Option A: Exclude from full-epoch test**

**File**: `tests/integration/datasets/test_sft_datasets_full_epoch.py`

Add to the exclusion list (if one exists) or add a skip marker:
```python
@pytest.mark.parametrize("dataset_fixture", [...], indirect=True)
@pytest.mark.e2e
def test_dataset_conversation(dataset_fixture):
    ...
```

Or within the fixture/test:
```python
if dataset_name == "Magpie-Align/Llama-3-Magpie-Pro-1M-v0.1":
    pytest.skip("Dataset too large for full epoch test (1M rows)")
```

**Option B: Increase timeout**

Add a generous timeout marker:
```python
@pytest.mark.timeout(1200)  # 20 minutes
@pytest.mark.e2e
def test_dataset_conversation(dataset_fixture):
    ...
```

This only works if the test environment has sufficient resources and CI allows long-running tests.

**Option C: Sample rows instead of full epoch**

```python
def test_dataset_conversation(dataset_fixture):
    dataset = dataset_fixture
    total = len(dataset)
    max_rows = min(total, 5000)
    indices = range(max_rows) if total <= 5000 else random.sample(range(total), 5000)
    for idx in indices:
        conversation = dataset.conversation(idx)
        ...
```

This validates the dataset's transformation logic without requiring full iteration.

**Option D: Mark as slow test with separate CI stage**

```python
@pytest.mark.slow
@pytest.mark.e2e
@pytest.mark.timeout(1200)
def test_dataset_conversation(dataset_fixture):
    ...
```

Then run slow tests separately: `pytest -m slow --timeout=1200`

## Impact

Low — the purpose of the full-epoch test is to validate that `transform_conversation()` works for every row without errors. Sampling (Option C) provides similar coverage with much better performance. The dataset's transformation logic is also covered by unit tests.

## Notes

- The test file is already listed in the "Excluded Tests" section of tests_status.md as a slow test to run manually
- Other large datasets in the same test may also approach timeout limits as new datasets are added
- Consider whether full-epoch iteration is necessary or if sampling provides sufficient test coverage
