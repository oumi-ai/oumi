# PreTrainedTokenizer Subclasses Must Call super().__init__()

## Breaking Change

In transformers v5, `PreTrainedTokenizer.__setattr__` was changed to require internal state (e.g., `_special_tokens_map`) to be initialized before any attribute assignment. Subclasses that skip `super().__init__()` and directly set attributes like `self.eos_token_id = None` will fail with `AttributeError: 'MockTokenizer' has no attribute '_special_tokens_map'`.

## Root Cause

Transformers v5 added a `__setattr__` hook to `PreTrainedTokenizerBase` that intercepts assignments to special token attributes (like `eos_token_id`, `pad_token_id`, etc.) and stores them in `_special_tokens_map`. This internal dict must be initialized by `super().__init__()` before any such assignment. In v4, there was no such hook and direct assignment worked.

Additionally, calling `super().__init__()` itself now requires abstract methods to be implemented (e.g., `get_vocab()`), so simple mock tokenizers that inherit from `PreTrainedTokenizer` need a different approach.

## How to Reproduce

```python
import transformers

class MockTokenizer(transformers.PreTrainedTokenizer):
    def __init__(self):
        # Fails on transformers>=5.0.0 — no super().__init__()
        self.eos_token_id = None

tok = MockTokenizer()
# AttributeError: MockTokenizer has no attribute '_special_tokens_map'
```

## Files Changed

- `tests/unit/datasets/test_pretraining_async_text_dataset.py`

## Fix Applied

Replaced the `PreTrainedTokenizer` subclass with a `unittest.mock.MagicMock` based mock that doesn't inherit from the transformers base class. The mock only needs to be callable and have `eos_token_id` set, so a MagicMock is sufficient.

Before:
```python
class MockTokenizer(transformers.PreTrainedTokenizer):
    def __init__(self):
        self.eos_token_id = None
    def __call__(self, x, **kwargs):
        ...
```

After:
```python
from unittest.mock import MagicMock

def _create_mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.eos_token_id = None
    def tokenize_call(x, **kwargs):
        ...
    tokenizer.side_effect = tokenize_call
    return tokenizer
```
