# Masking Analysis: Legacy vs Span-Based Collator

## Background

PRs #2368 and #2369 introduce span-based masking via a `MaskingMethod` enum as a replacement for the legacy instruction+response template collator. This analysis compares the two approaches to understand their behavior on tool-calling conversations.

## How the Pipeline Works

1. Training data contains `Conversation` objects with `Role.TOOL` messages
2. `Conversation.to_dict()` produces `{"role": "tool", "content": "..."}`
3. HuggingFace `apply_chat_template` renders messages using the **model's Jinja template**
4. The collator receives the final token sequence and applies label masking

The collator has no knowledge of roles — it only sees tokens and matches patterns.

## Legacy Collator

Uses `instruction_template` + `response_template` pattern matching:
- Finds all instruction starts and response starts
- Masks tokens between each instruction→response pair (user questions)
- Unmasks tokens between each response→instruction pair (assistant answers)

**Limitation**: Tool results must match `instruction_template` to be treated as a masking boundary. Whether this works depends entirely on the model's Jinja template.

## Span-Based Collator

Uses `response_template` + `end_of_turn_template`:
- Masks everything
- Finds each `response_template` occurrence
- Unmasks from response content start to the next `end_of_turn_template` (inclusive)

**Advantage**: Only assistant response spans are unmasked. Tool results, user messages, and system prompts are masked by construction — no dependency on how the Jinja template renders non-assistant roles.

## Model-Specific Behavior

### Qwen (ChatML format)

Qwen's Jinja template converts `role="tool"` into a user message:

```
<|im_start|>user
<tool_response>
{"temp": "65F"}
</tool_response><|im_end|>
```

Since the legacy collator uses `instruction_template: "<|im_start|>user\n"`, it matches tool results as instruction boundaries and masks them correctly — **by accident**.

The only difference between legacy and span-based masking on Qwen is a trailing `\n` after each `<|im_end|>`:
- Legacy: trains on it (inter-turn whitespace included in loss)
- Span-based: masks it (more correct)

Aggregate stats over 50 samples:

| Dataset | Differing tokens | % of total tokens |
|---------|-----------------|-------------------|
| TatQA (no tool calls) | 50 | 0.13% |
| Hermes (with tool calls) | 160 | 0.21% |

### Llama 3.1

Llama's Jinja template renders `role="tool"` as a distinct `ipython` role:

```
<|start_header_id|>ipython<|end_header_id|>

"{"temp": "65F"}"<|eot_id|>
```

The legacy collator uses `instruction_template: "<|start_header_id|>user<|end_header_id|>\n\n"`, which does **not** match `<|start_header_id|>ipython<|end_header_id|>`. As a result:

- Tool results are **not recognized** as instruction boundaries
- Everything between the first `response_template` match and the next `instruction_template` match is unmasked
- **The model trains on tool result tokens** (ipython header, tool content, etc.)

Example diff on a single tool-calling sample:

| Metric | Legacy | Span-based |
|--------|--------|------------|
| Unmasked tokens | 49 | 25 |
| Diff positions | — | 26 |

The 26 differing tokens include the full tool result content (`ipython` header + JSON response) and the next assistant response header — all incorrectly unmasked by legacy.

## TatQA Eval Results

Both collators were evaluated on TatQA using Qwen2.5-1.5B-Instruct (where masking is nearly identical):

| | Base Model | Tuned (main/legacy) | Tuned (pr2/span) | Teacher (70B) |
|--|-----------|--------------------|--------------------|---------------|
| Accuracy | 42.0% | 59.3% | 58.7% | 67.1% |

The 0.6% difference is consistent with training non-determinism (FSDP, data shuffling), not a masking correctness issue. Both collators produce functionally equivalent masks for Qwen on this dataset.

## Conclusion

| | Qwen | Llama |
|--|------|-------|
| Legacy masks tool results correctly? | Yes (accidentally — Jinja converts tool→user) | **No** (tool→ipython, not matched) |
| Span-based masks tool results correctly? | Yes | Yes |

The span-based collator is the correct general solution. It does not depend on model-specific Jinja template behavior to correctly mask non-assistant content. The legacy collator only works for models whose chat templates happen to render tool messages with the same header as user messages.
