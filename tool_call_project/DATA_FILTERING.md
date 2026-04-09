# Data Filtering Documentation

## Source Dataset

**Hermes Reasoning Tool Use** (`gold/data/hermes_reasoning_tool_use_{train,val,test}_split.jsonl`)

| Split | Records |
|-------|---------|
| train | 45,904  |
| val   | 2,550   |
| test  | 2,550   |

Each record is a multi-turn conversation with roles: `system`, `user`, `assistant`, `tool`.
Four scenario categories: `single`, `multiturn`, `multistep`, `relevance`.

---

## Filtering Script

**Script:** `filter_dataset.py`

A single script that produces both training and eval data, controlled by `--mode`:

```bash
python filter_dataset.py --mode train   # Training data only
python filter_dataset.py --mode eval    # Eval data only
python filter_dataset.py --mode both    # Both (default)
```

### Shared Filters (applied in both modes)

| Filter | Reason | Train removed | Val removed | Test removed |
|--------|--------|---------------|-------------|--------------|
| Remove `relevance` category | These examples have no tool calls — the model should decline. Not relevant for training or evaluating tool call generation. | 13,655 | 779 | 784 |
| Remove mislabeled assistant turns | Some assistant turns contain `<tool_response>` instead of `<tool_call>`. This teaches the model to hallucinate tool responses. Concentrated in ToolAce (197), Glaive (64), Nous-Hermes (49) sources, and multistep (168) / multiturn (142) categories. | 310 | 9 | 21 |

### Mode: train (`*_clean.jsonl`)

**Output:** `data/hermes_reasoning_tool_use_{split}_clean.jsonl`

Full conversations with quality filters applied. No truncation — suitable for SFT training.

| Split | Before | After |
|-------|--------|-------|
| train | 45,904 | 31,939 |
| val   | 2,550  | 1,762  |
| test  | 2,550  | 1,745  |

### Mode: eval (`*_tool_calls_only.jsonl`)

**Output:** `data/hermes_reasoning_tool_use_{split}_tool_calls_only.jsonl`

Same quality filters, plus additional eval processing:

| Step | Description |
|------|-------------|
| Skip unparseable | Skip records with no parseable `<tool_call>` JSON in any assistant turn. |
| Truncate messages | Remove the last assistant turn containing a tool call and everything after it. The remaining messages are the input context for inference. |
| Store gold tool call | Extracted `{"name": ..., "arguments": ...}` stored in `metadata.gold_tool_call`. |
| Store gold assistant content | Full text of the removed assistant turn stored in `metadata.gold_assistant_content` for inspection. |

| Split | Before | After |
|-------|--------|-------|
| train | 45,904 | 31,939 |
| val   | 2,550  | 1,762  |
| test  | 2,550  | 1,745  |

---

## Known Data Quality Issues

### 1. Mislabeled assistant turns

~310 training records have assistant turns containing `<tool_response>` without `<tool_call>`.
These are tool execution results that should have been labeled as `tool` role, not `assistant`.
This causes the model to learn to output `<tool_response>` instead of `<tool_call>`.

**Impact observed:** SmolLM2-135M baseline predicted `<tool_response>` in 52% of outputs.
After SFT on unfiltered data, this dropped to 16% but was not eliminated.

By source: ToolAce (197), Glaive (64), Nous-Hermes (49).
By category: multistep (168), multiturn (142).

### 2. Nested `<tool_call>` in `<think>` blocks

Some assistant messages mention `<tool_call>` inside `<think>` reasoning blocks.
A naive regex parser matched the wrong `<tool_call>` tag. Fixed by using `rfind` to
locate the last `</tool_call>` and working backwards to the nearest `<tool_call>`.

**Impact:** 1 test record was previously dropped (conv_id `19825`), and all records
after index 1466 were shifted by 1, misaligning predictions with gold labels.

---

## File Summary

```
tool_call_project/
├── filter_dataset.py             # Single script: --mode train|eval|both
├── data/
│   ├── *_clean.jsonl             # Full conversations, noisy samples removed
│   └── *_tool_calls_only.jsonl   # Truncated for eval, gold in metadata
```
