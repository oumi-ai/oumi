# SpecialTokensMixin Removed in Transformers v5

## Breaking Change

`transformers.SpecialTokensMixin` was removed entirely in transformers v5.0.0. Any code that imports or instantiates `SpecialTokensMixin` will fail with `ImportError`.

## Root Cause

In transformers v4, `SpecialTokensMixin` was a utility class that held special token attributes (`pad_token`, `eos_token`, `bos_token`, etc.) and was used as a base class for tokenizers. In v5, HuggingFace refactored the tokenizer internals and removed this class, folding its functionality directly into `PreTrainedTokenizerBase`.

## How to Reproduce

```python
# Fails on transformers>=5.0.0
from transformers import SpecialTokensMixin
# ImportError: cannot import name 'SpecialTokensMixin' from 'transformers'
```

## Files Changed

- `src/oumi/core/tokenizers/special_tokens.py`

## Fix Applied

Replaced `SpecialTokensMixin` with a simple `@dataclass` called `SpecialTokensConfig` that provides the same interface (`.pad_token`, `.eos_token`, `.bos_token`, `.unk_token`, `.additional_special_tokens`). The only attribute accessed by callers was `.pad_token`, so this is a drop-in replacement.

Before:
```python
from transformers import SpecialTokensMixin
LLAMA_SPECIAL_TOKENS_MIXIN = SpecialTokensMixin(pad_token="<|finetune_right_pad_id|>")
```

After:
```python
from dataclasses import dataclass

@dataclass
class SpecialTokensConfig:
    pad_token: str | None = None
    eos_token: str | None = None
    bos_token: str | None = None
    unk_token: str | None = None
    additional_special_tokens: list[str] = field(default_factory=list)

LLAMA_SPECIAL_TOKENS = SpecialTokensConfig(pad_token="<|finetune_right_pad_id|>")
```
