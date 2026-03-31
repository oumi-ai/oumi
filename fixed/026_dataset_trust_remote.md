# Plan: Thread `ModelParams.trust_remote_code` Through `build_dataset_mixture`

## Goal

Remove the dual use of `DatasetParams.trust_remote_code`. Currently it controls both HF dataset loading (deprecated, no-op in datasets v4+) and processor building (still needed). After this change, processor `trust_remote_code` comes exclusively from `ModelParams`.

## Production Code Changes

### 1. `src/oumi/builders/data.py`

Add `model_params: ModelParams | None = None` to `build_dataset_mixture()`:

```python
def build_dataset_mixture(
    data_params: DataParams,
    tokenizer: BaseTokenizer | None,
    dataset_split: DatasetSplit,
    seq_length: int | None = None,
    seed: int | None = None,
    model_params: ModelParams | None = None,  # NEW
) -> DatasetType | PretrainingAsyncTextDataset:
```

Pass it through to `_load_dataset()`. In `_load_dataset()`, derive the processor trust flag from `model_params.trust_remote_code` instead of `dataset_params.trust_remote_code`:

```python
def _load_dataset(
    dataset_params: DatasetParams,
    stream: bool,
    tokenizer: BaseTokenizer | None = None,
    trust_remote_code: bool = False,  # NEW — from ModelParams
) -> ...:
    ...
    dataset = dataset_class(
        ...
        trust_remote_code=trust_remote_code,  # was dataset_params.trust_remote_code
        **dataset_kwargs,
    )
```

### 2. `src/oumi/builders/oumi_data.py`

Same signature change to the torchdata `build_dataset_mixture()` and `_load_dataset()`.

### 3. `src/oumi/train.py`

Pass `config.model` to both `build_dataset_mixture()` calls (lines 360, 369):

```python
train_dataset = build_dataset_mixture(
    config.data,
    tokenizer,
    DatasetSplit.TRAIN,
    seq_length=config.model.model_max_length,
    model_params=config.model,  # NEW
)
```

### 4. `src/oumi/core/configs/params/data_params.py`

Re-deprecate `DatasetParams.trust_remote_code` — restore the forced reset to `False` in `__post_init__`, since the processor path will now use `ModelParams`:

```python
if self.trust_remote_code:
    warnings.warn(
        "`trust_remote_code` on DatasetParams is deprecated. "
        "Use `ModelParams.trust_remote_code` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    self.trust_remote_code = False
```

### 5. `src/oumi/core/datasets/vision_language_dataset.py`

No structural change needed. It already accepts `trust_remote_code` in its constructor. The value just comes from `ModelParams` via the builder instead of from `DatasetParams`.

## Test Code Changes

### 6. `tests/integration/datasets/test_sft_vision_datasets_load_datasets.py`

Pass `model_params` to `build_dataset_mixture()`. Stop setting `trust_remote_code` on `DatasetParams`:

```python
model_params = ModelParams(
    model_name=info.model_name,
    trust_remote_code=info.trust_remote_code,
    chat_template=info.chat_template,
)
...
dataset = build_dataset_mixture(
    DataParams(train=train_split), tokenizer, DatasetSplit.TRAIN,
    model_params=model_params,
)
```

### 7. `tests/unit/builders/test_build_data.py`

Update `build_dataset_mixture()` calls to pass `model_params` if testing trust_remote_code behavior.

### 8. `tests/unit/builders/test_oumi_data.py`

Same — update calls if needed (11 call sites). Most don't involve trust_remote_code so they just get the new optional parameter defaulting to `None`.

### 9. `tests/unit/builders/test_data_mixtures.py`

Same — 15+ call sites. Optional parameter, no change needed unless testing trust_remote_code.

## Result

- `DatasetParams.trust_remote_code` → truly deprecated, forced to `False`, only ever applied to HF dataset loading (already a no-op)
- `ModelParams.trust_remote_code` → single source of truth for model, tokenizer, and processor trust
- Collator path (already uses `ModelParams`) and dataset path converge on the same source
- New `model_params` parameter is optional with `None` default, so existing callers that don't pass it continue working
