# Processor apply_chat_template Now Requires Dict Messages

## Breaking Change

In transformers v5, the `apply_chat_template` method on processor objects (e.g., `LlavaProcessor`) now strictly expects `list[dict[str, str]]` messages, not arbitrary objects. Previously, any object with `.role` and `.content` attributes would work. Now, the method calls `.get("role")` on each message, which fails on non-dict types with `AttributeError: 'Message' object has no attribute 'get'`.

## Root Cause

Transformers v5 tightened the type expectations for `apply_chat_template` on processors. The internal implementation now uses dict-style access (`.get()`) rather than attribute access on messages. This is part of a broader move to standardize the chat template API around plain dicts.

## How to Reproduce

```python
from transformers import AutoProcessor
from oumi.core.types.conversation import Message, Role

processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
msg = Message(role=Role.USER, content="Hello")

# Fails on transformers>=5.0.0
processor.apply_chat_template([[msg]])
# AttributeError: 'Message' object has no attribute 'get'
```

## Files Changed

- `src/oumi/core/processors/default_processor.py`

## Fix Applied

Added explicit conversion of `Message` objects to dicts before passing them to the processor's `apply_chat_template`. The tokenizer path (the `if isinstance(self._worker_processor, BaseTokenizer)` branch) was not affected, only the processor path.

Before:
```python
result = self._worker_processor.apply_chat_template(
    [conversation], add_generation_prompt=add_generation_prompt
)
```

After:
```python
dict_conversation = [
    {"role": msg.role.value, "content": msg.content}
    for msg in conversation
]
result = self._worker_processor.apply_chat_template(
    [dict_conversation], add_generation_prompt=add_generation_prompt
)
```
