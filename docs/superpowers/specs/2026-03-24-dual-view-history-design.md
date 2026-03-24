# Dual-View History for Synthesis Tool-Call Robustness

## Problem

The synthesis pipeline generates multi-turn tool-calling conversations by alternating LLM calls for user and assistant turns. Both roles share a single `histories` list as context. This causes two failures:

1. **Assistant hallucinates tool usage in prose.** The assistant LLM sometimes narrates "Let me run a query..." followed by fabricated results, without emitting `<tool_call>` tags. `parse_tool_call` returns `None`, and the hallucinated narrative becomes plain `content` in the output.

2. **User absorbs assistant tool mechanics.** The user LLM sees `<tool_call>` tags and `[Tool result from RunQuery]: {...}` messages in the shared history. It absorbs this pattern and generates responses that reference fabricated data, describe tool executions, or otherwise break character as a user.

Bug 2 is a downstream consequence of sharing a single history. A real user never sees raw API calls — they see the assistant's natural language summary. The user LLM should have the same experience.

## Solution: Dual-View History

Replace the single `histories` list with two parallel histories per sample:

- **`full_histories`** — Complete record: user messages, assistant messages (including `<tool_call>` tags), tool results. Used when prompting the **assistant** LLM.
- **`conversational_histories`** — User-visible record: user messages and assistant prose responses only. No tool call mechanics. Used when prompting the **user** LLM.

## Design

### Data Structures

In `_synthesize_all_samples`, replace:

```python
histories: list[list[Message]] = [[] for _ in samples]
```

With:

```python
full_histories: list[list[Message]] = [[] for _ in samples]
conversational_histories: list[list[Message]] = [[] for _ in samples]
```

### Prompt Building (History Selection)

When building the prompt for each turn, select the appropriate history based on role:

```python
history = full_histories[i] if is_tool_turn else conversational_histories[i]

msgs = (
    [persona_msg]
    + history
    + [Message(role=Role.USER, content=turn_info)]
    + turn_tool_msgs[i]
)
```

- Assistant tool turns use `full_histories` — sees prior `<tool_call>` tags and tool results as in-context examples of correct behavior.
- User turns use `conversational_histories` — sees only natural language exchanges, never tool mechanics.

`turn_tool_msgs` is already scoped to assistant tool turns only, so it does not leak to the user.

### History Updates

**Non-tool-call path** (when `tool_call is None` — covers both user messages and assistant prose responses):

```python
if tool_call is None:
    full_histories[idx].extend(turn_tool_msgs[idx])
    full_histories[idx].append(Message(role=role, content=text))
    conversational_histories[idx].append(Message(role=role, content=text))
    if tool_executor:
        content = ToolExecutor.strip_tool_tags(text)
        content = ToolExecutor.strip_bare_tool_json(content)
        output_messages[idx].append(
            {"role": role.value, "content": content}
        )
    continue
```

Both histories get the conversational message. Only `full_histories` gets `turn_tool_msgs` (the tool call/result exchange from the current turn).

**`_record_tool_result`** (during assistant tool turns):

No changes to this method. Tool call mechanics are appended to `turn_tool_msgs[idx]`, which feeds into `full_histories` only (via the merge in the non-tool-call path above).

However, `_record_tool_result` also extracts prose content around tool calls via `extract_content_around_tool_call` (e.g., "Let me check that for you."). This prose goes into `turn_tool_msgs[idx]` as an assistant message. When `turn_tool_msgs` is merged into `full_histories`, these prose fragments must also be selectively appended to `conversational_histories` — otherwise the user LLM will see gaps where the assistant said something conversational before calling a tool.

In the non-tool-call merge block, iterate `turn_tool_msgs[idx]` and append any `Role.ASSISTANT` messages to `conversational_histories` (skip `Role.USER` messages which are tool results):

```python
if tool_call is None:
    # Merge tool turn messages into full history
    full_histories[idx].extend(turn_tool_msgs[idx])
    # Selectively add assistant prose fragments to conversational history
    for msg in turn_tool_msgs[idx]:
        if msg.role == Role.ASSISTANT:
            conversational_histories[idx].append(msg)
    full_histories[idx].append(Message(role=role, content=text))
    conversational_histories[idx].append(Message(role=role, content=text))
    ...
```

**Generated simulator retry context:**

The retry lambda for generated tool results (line ~976) reads `histories` directly to build context for the tool-result-generation LLM. This must use `full_histories` since it is part of the tool execution pipeline:

```python
retry_prompt_fn=lambda j: (
    tool_executor.build_generated_simulator_prompt(
        gen_items[j][2],
        full_histories[gen_items[j][0]]
        + turn_tool_msgs[gen_items[j][0]],
    )
),
```

**End-of-method conversations build:**

```python
for sample, history in zip(samples, full_histories):
    ...
```

Uses `full_histories` for the complete conversation record.

### What Does NOT Change

- `_record_tool_result` — unchanged, still writes to `turn_tool_msgs` and `output_messages`
- `output_messages` — unchanged, still the source for the final JSONL output
- `tool_executor.py` — unchanged
- Output format — unchanged
- `_format_persona` prompt strengthening (rules 5-7) — kept as belt-and-suspenders

## Why This Works

The assistant LLM sees its own prior `<tool_call>` usage in `full_histories`, creating strong in-context reinforcement of the correct format. Every prior tool interaction in its history demonstrates the pattern it should follow.

The user LLM never sees tool mechanics, so it cannot parrot `<tool_call>` tags, reference raw query results, or generate assistant-like content. It only sees the natural conversation flow — exactly what a real user would experience.

## Files Changed

- `src/oumi/core/synthesis/conversation_synthesizer.py` — `_synthesize_all_samples` method only

## Risks

- **History divergence bugs.** Two histories must stay synchronized on conversational messages. A missed append to one history would cause the user and assistant to have different views of the conversation content (not just tool mechanics). Mitigated by the update logic being in a single code block.
- **Existing tests.** Tests that mock or inspect `histories` will need updating to reference `full_histories` and `conversational_histories`.
