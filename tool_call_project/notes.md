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

#### Case B: With `instruction_template` (lines 124–195) — BUG WITH TOOL TURNS

The *intended* behaviour is to find all `instruction_template` and `response_template` positions, pair them up via `zip()`, and mask the regions between each instruction and its following response.

However, `zip()` stops at the shorter list. In a tool-calling conversation, the tool result uses a **different role marker** (e.g., `<|im_start|>tool\n` instead of `<|im_start|>user\n`), so the collator only sees **1 instruction** but **2 responses**:

```
Actual token sequence (SmolLM2 example):
  <|im_start|>user\n  What's the weather?  <|im_end|>
  <|im_start|>assistant\n  <tool_call>...</tool_call>  <|im_end|>
  <|im_start|>tool\n  {"result": "Sunny"}  <|im_end|>       ← NOT matched as instruction!
  <|im_start|>assistant\n  It's sunny!  <|im_end|>

  human_token_ids_idxs    = [pos_user]                    ← only 1 match
  response_token_ids_idxs = [pos_asst_1, pos_asst_2]      ← 2 matches

  zip([pos_user], [pos_asst_1, pos_asst_2])
    → only 1 pair: (pos_user, pos_asst_1)
    → masks labels[:pos_asst_1]

  Everything after pos_asst_1 is UNMASKED — including the tool result.
```

Key lines:
```python
# line 185–186 — zip stops at the shorter list
for idx, (start, end) in enumerate(zip(human_token_ids_idxs, response_token_ids_idxs)):

# line 190 — mask between instruction and response (middle turns)
batch["labels"][i, start:end] = self.ignore_index

# line 192 — mask from beginning up to first response (first turn)
batch["labels"][i, :end] = self.ignore_index

# line 194–195 — only fires when responses < instructions, not our case
if len(response_token_ids_idxs) < len(human_token_ids_idxs):
    batch["labels"][i, human_token_ids_idxs[-1]:] = self.ignore_index
```

---

**Actual result for tool turns (BUG):**

| Turn | Role | Expected | Actual (Case B) |
|------|------|----------|-----------------|
| 1 | user | masked | masked |
| 2 | assistant (tool_call) | unmasked | unmasked |
| 3 | tool (tool result) | **masked** | **unmasked ← BUG** |
| 4 | assistant (final answer) | unmasked | unmasked |

The root cause: the collator only recognises `<|im_start|>user\n` as an instruction boundary. The tool result uses `<|im_start|>tool\n` — a different marker the collator ignores. After the first assistant turn ends, the tool result leaks into the unmasked region.

### Fix: `ToolAwareCompletionsCollator`

A new collator (`src/oumi/core/collators/tool_aware_completions_collator.py`) fixes this by taking a fundamentally different approach:

1. Start with **all labels masked** (-100).
2. Find every `response_template` → next `end_of_turn_template` span.
3. **Unmask** only those spans (assistant content).
4. Optionally re-mask spans containing `tool_call_start_template` (`mask_tool_calls=True`).

This never relies on instruction markers, so tool result turns are always masked correctly.

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
