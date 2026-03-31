# hf_vision Test Failure — Missing Required kwargs

## Status: UNFIXED (pre-existing, previously masked)

## Issue

The auto-discovered `hf_vision` test fails with:
```
TypeError: HuggingFaceVisionDataset.__init__() missing 3 required keyword-only arguments: 'hf_dataset_path', 'image_column', and 'question_column'
```

### Affected Test

- `test_sft_vision_datasets_load_datasets.py::test_build_dataset_mixture[hf_vision HuggingFaceTB/SmolVLM-256M-Instruct]`

This was previously masked by the Pydantic serialization error (issue #018) which failed earlier in the pipeline.

## Root Cause

### The Dataset Class

**File**: `src/oumi/datasets/vision_language/huggingface.py`, lines 38-106

`HuggingFaceVisionDataset` is a generic wrapper registered as `"hf_vision"`. It requires three mandatory kwargs:

```python
@register_dataset("hf_vision")
class HuggingFaceVisionDataset(VisionLanguageSftDataset):
    def __init__(
        self,
        *,
        hf_dataset_path: str,        # REQUIRED — HuggingFace dataset path
        image_column: str,           # REQUIRED — column name containing images
        question_column: str,        # REQUIRED — column name containing questions
        answer_column: str | None = None,
        system_prompt_column: str | None = None,
        system_prompt: str | None = None,
        **kwargs,
    ):
```

### How the Test Fails

**File**: `tests/integration/datasets/test_sft_vision_datasets_load_datasets.py`, lines 150-164

The test auto-discovers all registered `VisionLanguageSftDataset` subclasses:
```python
def _get_all_sft_vision_dataset_names() -> list[str]:
    for key, value in REGISTRY.get_all(RegistryType.DATASET).items():
        if issubclass(value, VisionLanguageSftDataset):
            datasets.append(key)
```

It then creates a generic `LoadDatasetInfo` for each discovered dataset (lines 156-164) with only standard kwargs (`processor_name`, `limit`, `return_tensors`). The three required `HuggingFaceVisionDataset` kwargs are not provided.

The dataset is instantiated via `build_dataset_mixture()` → `_load_dataset()` in `src/oumi/builders/data.py` (line 280-288), which passes `**dataset_kwargs` from the test — but those kwargs don't include the required arguments.

### Why It Can't Work Generically

`hf_vision` is a generic wrapper that maps arbitrary HuggingFace datasets to the vision-language format. Each usage requires specifying which columns contain images, questions, and answers. There is no sensible default. Example usage from config files:

```yaml
# From configs/projects/dcvlr/starter_kit/qwenvl-openr1.yaml
datasets:
  - dataset_name: hf_vision
    dataset_kwargs:
      hf_dataset_path: "lmms-lab/multimodal-open-r1-8k-verified"
      image_column: "images"
      question_column: "problem"
      answer_column: "solution"
```

## Recommended Fix

**Option A (recommended)**: Add `"hf_vision"` to the `_EXCLUDED_DATASETS` set:

**File**: `tests/integration/datasets/test_sft_vision_datasets_load_datasets.py`, line 62

```python
_EXCLUDED_DATASETS = set({
    "coco_captions",
    "nlphuji/flickr30k",
    "vision_language_jsonl",
    "vl_sft",
    "hf_vision",  # ← ADD: requires dataset-specific config (hf_dataset_path, columns)
})
```

**Option B**: Add a specific test entry with valid kwargs for a known dataset, plus exclude from auto-discovery:

```python
LoadDatasetInfo(
    dataset_name="hf_vision",
    model_name=_DEFAULT_MODEL_NAME,
    dataset_split="train",
    chat_template="auto",
    trust_remote_code=True,
    max_rows=32,
    expected_rows=32,
    # Would need extra_kwargs or similar mechanism to pass:
    # hf_dataset_path="lmms-lab/multimodal-open-r1-8k-verified"
    # image_column="images"
    # question_column="problem"
),
```

This requires extending `LoadDatasetInfo` to support extra dataset kwargs, which is more invasive.

## Impact

Low — `hf_vision` is not a standalone dataset but a configurable wrapper. Other datasets that inherit from it (like `LmmsLabMultimodalOpenR1Dataset`) are tested separately.
