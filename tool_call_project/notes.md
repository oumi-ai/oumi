# Investigation: Tool Use in Oumi

## Context

Understanding how tool/function calling is implemented in oumi's conversation format and how tool-related tokens are handled during training in the collator pipeline. This is foundational knowledge before any changes to extend or fix tool use behavior.

---

## Background: Tool Calling Primer

Tool calling is a two-step exchange between the model and your code:

| Term | Producer | What it is |
|---|---|---|
| **Tool call** | The **model (ASSISTANT)** | "I want to call function X with these arguments" |
| **Tool result** | **Your code** | The return value after you ran the function |

### Conversation flow example

```
User:  "What's the weather in Paris?"

Assistant (tool call — role=ASSISTANT, content=None):
  tool_calls: [{ id: "call_abc123", function: { name: "get_weather", arguments: '{"location":"Paris"}' } }]

  → your code runs get_weather("Paris") → "Sunny, 18°C"

Tool result (role=TOOL):
  tool_call_id: "call_abc123"
  content: "Sunny, 18°C"

Assistant (final answer):
  content: "The weather in Paris is sunny and 18°C."
```

The `tool_call_id` on the tool result links it back to the specific call that triggered it — important when the model makes multiple tool calls in parallel.

### Is this how HuggingFace does it?

The **data format** (roles, `tool_calls`, `tool_call_id`) follows the **OpenAI schema**, which is the de-facto standard. HuggingFace models that support tool calling use the same schema. The difference is in serialization:

- **OpenAI/Anthropic API**: structured JSON objects; the API handles formatting
- **HuggingFace local models**: `apply_chat_template()` converts those same objects into a raw string with model-specific special tokens (e.g., `<tool_call>`, `[TOOL_RESULTS]`) before tokenization

Oumi's design — use OpenAI-style `ToolCall`/`ToolDefinition` objects internally, serialize per-API — is consistent with how HuggingFace works.

### Feature branch age

`oelachqar/infer-tool-calling` has one commit on top of main, dated **2025-11-28** (~4 months old, not yet merged).

---

## Part 1: Tool Use in the Conversation Format

### Current State (main branch)

**Key file:** `src/oumi/core/types/conversation.py`

The `Message` class uses:
- `role` (Role enum): SYSTEM, USER, ASSISTANT, **TOOL** (already present)
- `content` (str | list[ContentItem])
- No tool call fields yet (no `tool_calls`, no `tool_call_id`)

The `FinishReason` enum includes `TOOL_CALLS` — indicating the model triggered tool calls.

### In-flight feature branch: `oelachqar/infer-tool-calling`

A complete tool calling implementation exists but is **not yet merged to main**. It adds:

#### New file: `src/oumi/core/types/tool_call.py`
- `ToolDefinition` / `FunctionDefinition` — describes a callable tool
- `ToolCall` / `FunctionCall` — a specific tool invocation (name + JSON arguments string)
- `ToolChoiceType` enum: AUTO, NONE, REQUIRED

#### Extended `Message` fields:
```python
tool_calls: Optional[list[ToolCall]] = None   # ASSISTANT role only
tool_call_id: Optional[str] = None            # TOOL role only
content: ... | None                           # Can be None when tool_calls is set
```

#### Extended `GenerationParams`:
```python
tools: Optional[list[ToolDefinition]] = None
tool_choice: Optional[Union[str, dict]] = None
parallel_tool_calls: bool = True
```

#### API serialization:
- **OpenAI:** `tool_calls` array on assistant messages; TOOL role → `{"role": "tool", "tool_call_id": ...}`
- **Anthropic:** assistant `tool_use` content blocks; TOOL role → user message with `tool_result` block

---

## Part 2: Tool Masking in Collators

### Key files:
- `src/oumi/core/collators/trl_data_collator_for_completion_only_lm.py` — core masking logic
- `src/oumi/core/tokenizers/utils.py` — `mask_labels_for_completions_only()`, `mask_labels_without_user_template()`
- `src/oumi/core/feature_generators/vision_language_conversation_feature_generator.py` — VLM pipeline

### Current behavior: NO tool-specific masking

The collator doesn't know about roles at all. It only sees a flat token sequence and looks for special delimiter strings to decide what to mask. Here's how each mode works:

---

#### Case A: Without `instruction_template` (lines 87–122)

Scans for the **last** occurrence of `response_template` and masks everything up to and including it.

```
Token sequence (flat string after chat_template):
  <sys>...</sys> <user>Hello</user> <assistant> Hi! </assistant> <user>Weather?</user> <assistant> [RESP] Sunny.

                                                               ^^^^^^^^^^^^^^^^^^
                                                               response_template (last occurrence)

Labels after masking (line 122):
  [-100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100]  ← all masked up to here
                                                                   [Sunny .]  ← only this trains
```

Key line:
```python
# line 122
batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index
```

---

#### Case B: With `instruction_template` (lines 124–195)

Finds **all** occurrences of both templates, then masks the regions between each `[INST]` and the following `[RESP]`. What remains unmasked is only the content between `[RESP]` and the next `[INST]`.

```
Token sequence (example with tool turn):
  [INST] Hello [/INST] [RESP] Sure, let me check. <tool_call>...</tool_call> [INST] [tool result: Sunny] [/INST] [RESP] It's sunny!

  human_token_ids_idxs   = [pos_INST_1, pos_INST_2]
  response_token_ids_idxs = [pos_after_RESP_1, pos_after_RESP_2]

  Loop pairs (human_start → response_end):
    Pair 0: 0 → pos_after_RESP_1    mask labels[:pos_after_RESP_1]   (line 192)
    Pair 1: pos_INST_2 → pos_after_RESP_2    mask labels[pos_INST_2:pos_after_RESP_2]   (line 190)

Labels result:
  [-100 -100 -100 -100] [Sure, let me check. <tool_call>...</tool_call>] [-100 -100 -100 -100 -100] [It's sunny!]
   ^^ masked ^^                  ^^ UNMASKED (trains) ^^                      ^^ masked ^^              ^^ UNMASKED ^^
```

Key lines:
```python
# line 190 — mask between instruction and response (middle turns)
batch["labels"][i, start:end] = self.ignore_index

# line 192 — mask from beginning up to first response (first turn)
batch["labels"][i, :end] = self.ignore_index

# line 195 — mask trailing instruction with no following response
batch["labels"][i, human_token_ids_idxs[-1]:] = self.ignore_index
```

---

**Result for tool turns:** The tool result (`[INST] [tool result: ...] [/INST]`) sits between two instruction markers, so it is always in a masked region. Only ASSISTANT turns contribute to training loss.

This means:
- **Training on tool call generation** (assistant deciding to call a tool) works correctly — `<tool_call>...</tool_call>` is in the ASSISTANT turn, so it is **unmasked**.
- **Tool results** (role=TOOL) are always masked — the model never trains to produce them (correct, since your code produces them).
- There is no mechanism today to selectively unmask/mask parts of tool call arguments.

---

## Summary of Files to Read for Deeper Work

| File | Relevance |
|------|-----------|
| `src/oumi/core/types/conversation.py` | Message, Conversation, Role, FinishReason |
| `src/oumi/core/types/tool_call.py` | Tool types (feature branch only) |
| `src/oumi/core/collators/trl_data_collator_for_completion_only_lm.py` | Core label masking |
| `src/oumi/core/tokenizers/utils.py` | `mask_labels_for_completions_only()` |
| `src/oumi/core/feature_generators/vision_language_conversation_feature_generator.py` | VLM completion masking |
| `src/oumi/core/configs/params/generation_params.py` | GenerationParams (tools added in feature branch) |
| `src/oumi/inference/openai_inference_engine.py` | OpenAI tool call serialization |
| `src/oumi/inference/anthropic_inference_engine.py` | Anthropic tool call serialization |
