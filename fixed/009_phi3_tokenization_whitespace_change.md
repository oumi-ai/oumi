# Phi-3 Tokenizer Whitespace Handling Changed in Transformers v5

## Breaking Change

The Phi-3 tokenizer (`microsoft/Phi-3-vision-128k-instruct`) produces different token IDs in transformers v5. Specifically:
- Extra whitespace tokens (token `29871`) before newlines are no longer emitted
- The word "This" now tokenizes as `4013` instead of `910`
- Overall token sequence is shorter (16 tokens vs 19 tokens for the same input)

This breaks any tests or code that hardcodes expected token ID sequences.

## Root Cause

Transformers v5 updated the fast tokenizer backend and changed whitespace normalization behavior. The `29871` token was a whitespace artifact from the SentencePiece/Llama tokenizer family that was inserted before certain characters. This normalization was cleaned up in v5, resulting in more compact tokenizations.

## How to Reproduce

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("microsoft/Phi-3-vision-128k-instruct",
                                     trust_remote_code=True)
prompt = "<|user|>\nWhat is this?<|end|>\n<|assistant|>\nThis is a test.<|end|>\n"

# transformers v4.57: [32010, 29871, 13, 5618, 338, 445, 29973, 32007, 29871, 13, 32001, 910, 338, 263, 1243, 29889, 32007, 29871, 13]
# transformers v5.3:  [32010, 13, 5618, 338, 445, 29973, 32007, 13, 32001, 4013, 338, 263, 1243, 29889, 32007, 13]
tokens = tok.encode(prompt, add_special_tokens=False)
```

## Files Changed

- `tests/integration/datasets/test_vision_language_completions_only.py`

## Fix Applied

Test now uses `is_transformers_v5()` to select the correct expected token sequence for each version.

```python
if is_transformers_v5():
    expected_tokens = [
        32010, 13, 5618, 338, 445, 29973, 32007, 13,
        32001, 4013, 338, 263, 1243, 29889, 32007, 13,
    ]
else:
    expected_tokens = [
        32010, 29871, 13, 5618, 338, 445, 29973, 32007, 29871, 13,
        32001, 910, 338, 263, 1243, 29889, 32007, 29871, 13,
    ]
```
