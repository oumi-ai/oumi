# Agentic Tool Synthesis Design

## Overview

Extend the multi-turn conversation synthesis system to support agentic tool use. Users define tools, the conversation guider provides generic turn-level instructions, and the agent autonomously decides when and which tools to call. Tool outputs are resolved via deterministic or LLM-generated strategies. An optional environment layer enables stateful tool interactions when needed.

**Core principles:**
- **Tools-first, agent-driven** — the agent chooses tools, not the planner
- **Stateless by default** — environment is optional
- **Universal output format** — HuggingFace/OpenAI standard for training compatibility
- **Auto-injection** — system handles tool framing, the user just defines tools and personas

---

## Goals

- Enable users to define tools with JSON Schema parameters and output strategies
- Agent autonomously decides when/which tools to call (no tool names in plan)
- Support multi-tool calls within a single assistant turn (chained reasoning)
- Output in universal tool call format (HuggingFace `apply_chat_template` compatible)
- Tool calls are sub-steps within an ASSISTANT turn (no impact on `min_turns`/`max_turns`)
- Optionally support mutable environment state for stateful tool interactions
- System auto-injects tool catalog and usage framing when tools are available

---

## How Tools Work in Training Data

Training a model to use tools requires two things in every dataset row: (1) a **tool registry** that describes available tools, and (2) **messages** that demonstrate when/how to call them. This matches the format used by Nemotron-Agentic-v1, TOUCAN-1.5M, and the HuggingFace/OpenAI standard.

### Anatomy of a training row

A single row in the output dataset has this structure:

```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "SearchOrders",
        "description": "Look up order details by customer ID",
        "parameters": {
          "type": "object",
          "properties": {
            "customer_id": {"type": "string", "description": "The customer's ID"}
          },
          "required": ["customer_id"]
        }
      }
    }
  ],
  "messages": [
    {"role": "system", "content": "You are a customer support agent..."},
    {"role": "user", "content": "Can you check my order?"},
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_001",
        "type": "function",
        "function": {
          "name": "SearchOrders",
          "arguments": "{\"customer_id\": \"CUST002\"}"
        }
      }]
    },
    {
      "role": "tool",
      "tool_call_id": "call_001",
      "content": "{\"order_id\": \"ORD-001\", \"status\": \"delivered\"}"
    },
    {
      "role": "assistant",
      "content": "Your order ORD-001 has been delivered."
    }
  ]
}
```

### The two top-level fields

**`tools`** — Describes available functions. Never appears inside messages. When training, the tokenizer uses `apply_chat_template(messages, tools=tools)` to inject these into the model's native format (Llama uses `<|python_tag|>`, Qwen uses special tokens, Mistral uses `[AVAILABLE_TOOLS]`, etc.). This is auto-generated from `ToolAttribute` definitions.

**`messages`** — The conversation, containing three tool-related message types:
 
| Message type | `role` | Key fields | Description |
|---|---|---|---|
| Tool call | `assistant` | `tool_calls: [{id, type, function: {name, arguments}}]` | Agent decides to call a tool. `content` is `null` or empty. |
| Tool result | `tool` | `tool_call_id`, `content` | System returns the tool's output. Linked by `tool_call_id`. |
| Text response | `assistant` | `content` | Agent responds to user after reasoning over tool results. |

### arguments: string vs dict

The `arguments` field varies by convention:
- **Nemotron-Agentic-v1**: JSON string (`"{\"customer_id\": \"CUST002\"}"`)
- **HuggingFace apply_chat_template**: Accepts both, converts internally
- **OpenAI API responses**: JSON string

**Our output uses JSON string** for maximum compatibility. `apply_chat_template` handles both.

### Synthesis-time vs output format

During synthesis, the generating LLM produces `<tool_call>` tagged text (a generation-time convention). The system parses this and converts to the standard message format for the output dataset. The model being *trained on* this data never sees `<tool_call>` tags — it sees the native tool-calling format for its architecture.

| Concern | Format |
|---------|--------|
| LLM generation (internal, synthesis-time) | `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` |
| Output (training data) | Standard `assistant.tool_calls` / `role: "tool"` messages |

---

## Turn Counting

A **turn** is a change of conversation participant. Tool calls within an ASSISTANT turn do not count as separate turns.

```
Turn 1 [USER]:      "Can you check my order and process a return?"
Turn 2 [ASSISTANT]:  ← ONE turn containing:
    tool_call: SearchOrders(customer_id="CUST002")
    tool_result: {"order_id": "ORD-001", ...}
    tool_call: ProcessReturn(order_id="ORD-001", reason="...")
    tool_result: {"return_id": "RET-001", ...}
    response: "I've processed your return..."
Turn 3 [USER]:      "Thanks!"
Turn 4 [ASSISTANT]:  "You're welcome."
```

`min_turns`/`max_turns` count participant alternations only. The output messages array will have more messages than turns — this is expected and matches the standard format.

---

## Tool Output Strategies

| Strategy | Source | When to use |
|----------|--------|-------------|
| **DETERMINISTIC** | User-defined outputs, one selected per conversation | Generic tools whose output doesn't depend on input args (e.g., escalation tickets, status confirmations) |
| **GENERATED** | LLM simulates output given schema + args + context | Tools whose output must be coherent with the input arguments (e.g., order lookups, searches, eligibility checks) |

### DETERMINISTIC

- Single output in the list: always returns that output
- Multiple outputs: one randomly selected per conversation (weighted by `sample_rate`, uniform if unset)
- Selection happens once at conversation initialization, not per-call
- **Important:** Deterministic outputs ignore input arguments entirely. Use this only for tools where the response is generic regardless of what was queried (e.g., a tool that always returns an escalation ticket). Do NOT use for lookup/search tools where the response should match the query.

### GENERATED

- LLM receives: tool config + args + conversation history + optional environment snapshot
- `output_instruction` provides hints for realistic simulation
- `output_schema` gives the LLM structure guidance for the response format
- **Use this for any tool where the output should be coherent with the input arguments** (e.g., searching orders by ID, checking eligibility for a specific item)

---

## Config Schema

### Current state (already implemented)

**`tool_params.py`** — Contains `ToolOutputStrategy`, `ToolOutputValue`, `ToolAttribute`. All validation is in place (sample rate normalization, strategy-specific field validation).

**`synthesis_params.py`** — `MultiTurnAttribute` has the `available_tools: list[str]` field with validation.

**`conversation.py`** — `Role.TOOL` exists in the enum.

### Still needed

#### 1. `EnvironmentConfig` (in `tool_params.py`)

```python
@dataclass
class EnvironmentConfig:
    """Optional environment state for stateful tool interactions.

    Entities define structured state with {placeholders} that get resolved
    from sample attributes. Context provides domain rules/constraints.
    """
    entities: dict[str, Any] = field(default_factory=dict)
    context: str = ""
```

#### 2. `tools` field on `GeneralSynthesisParams` (in `synthesis_params.py`)

```python
from oumi.core.configs.params.tool_params import ToolAttribute

class GeneralSynthesisParams(BaseParams):
    # ... existing fields ...

    tools: list[ToolAttribute] | None = None
    """Tool definitions for agentic synthesis.

    Tools are defined here and referenced by id from
    MultiTurnAttribute.available_tools."""
```

#### 3. `environment` field on `MultiTurnAttribute` (in `synthesis_params.py`)

```python
from oumi.core.configs.params.tool_params import EnvironmentConfig

class MultiTurnAttribute:
    # ... existing fields ...

    environment: EnvironmentConfig | None = None
    """Optional environment state for stateful tool interactions."""
```

#### 4. Cross-validation (in `GeneralSynthesisParams.__post_init__`)

```python
# Validate that all available_tools ids reference defined tools
if self.tools and self.multiturn_attributes:
    tool_ids = {t.id for t in self.tools}
    for mt in self.multiturn_attributes:
        for tool_id in mt.available_tools:
            if tool_id not in tool_ids:
                raise ValueError(
                    f"MultiTurnAttribute '{mt.id}' references unknown "
                    f"tool '{tool_id}'. Available: {tool_ids}"
                )
```

---

## Guider (replaces Planner)

The conversation planner becomes a **guider**. When `available_tools` is non-empty, it provides generic task-oriented instructions without naming specific tools.

### What the system auto-injects into the guider prompt

```
The assistant has access to tools and should need to use them to complete
their tasks. Write instructions that require the assistant to look up
information, verify details, or take actions — describe WHAT to accomplish,
not which specific tool to use.
```

### Guider output format

```json
[
  {"turn": 1, "role": "USER", "instruction": "ask about returning their order"},
  {"turn": 2, "role": "ASSISTANT", "instruction": "help the customer by looking up their order details and checking if a return is possible"},
  {"turn": 3, "role": "USER", "instruction": "confirm they want to proceed with the return"},
  {"turn": 4, "role": "ASSISTANT", "instruction": "process the return request and confirm next steps"}
]
```

No `tool_calls` field. No tool names. The guider describes what the agent should accomplish — the agent decides how.

---

## Auto-Injection (when `available_tools` is non-empty)

The user writes domain-specific personas. The system automatically handles all tool-related framing.

### What gets auto-injected

| What | Where | Source |
|------|-------|--------|
| Tool catalog (names, descriptions, parameters) | ASSISTANT prompt | Auto-generated from `ToolAttribute` definitions |
| Tool call format instructions | ASSISTANT prompt | Hardcoded template |
| "Use tools, don't fabricate" framing | ASSISTANT prompt | Hardcoded template |
| "Write action-oriented instructions" | Guider system prompt | Hardcoded template |
| Zero-tool-use filtering | Post-synthesis | Automatic when tools available |

### Tool catalog generation

The system builds the catalog from `ToolAttribute` fields:

```python
def build_tool_catalog(tools: list[ToolAttribute]) -> str:
    """Auto-generate tool catalog from ToolAttribute definitions.

    Produces formatted text injected into the ASSISTANT's system prompt
    during synthesis. This is NOT the output format — it's what the
    generating LLM sees to know what tools are available.
    """
```

### Tool definitions for output (separate from catalog)

```python
def build_tool_definitions(tools: list[ToolAttribute]) -> list[dict]:
    """Convert ToolAttributes to OpenAI/HuggingFace tool format.

    This is the `tools` field in the output dataset row.
    Used at training time with apply_chat_template(messages, tools=tools).
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
        }
        for tool in tools
    ]
```

### Combined ASSISTANT prompt (user persona + auto-injected)

```
{user-written persona}

You have access to the following tools. Use them to look up information
and perform actions — do not guess or fabricate data.

Tools:
{auto-generated tool catalog}

To use a tool, output:
<tool_call>{"name": "ToolName", "arguments": {"param": "value"}}</tool_call>

After receiving a tool result, you may call another tool or respond to the user.
```

---

## Agentic Turn Loop

During an ASSISTANT turn, the agent enters an agentic loop where it autonomously decides to call tools or respond.

### Single-sample logic

```python
# Inside an ASSISTANT turn
tool_call_counter = 0
while True:
    response = generate(
        persona + tool_catalog + conversation_history + turn_instruction
    )
    tool_call = parse_tool_call(response)  # look for <tool_call> tags

    if tool_call is None:
        # Natural language response — turn is done
        history.append(assistant_message(content=response))
        break

    # Tool call detected — resolve output and continue loop
    tool_call_counter += 1
    call_id = f"call_{tool_call_counter:03d}"
    tool_config = tools_by_name[tool_call["name"]]
    tool_result = resolve_output(tool_call, tool_config, deterministic_outputs, env)

    # Append as standard format messages
    history.append(assistant_message(tool_calls=[{
        "id": call_id,
        "type": "function",
        "function": {
            "name": tool_call["name"],
            "arguments": json.dumps(tool_call["arguments"]),
        }
    }]))
    history.append(tool_message(tool_call_id=call_id, content=tool_result))
    # Loop: agent sees tool result and decides next action
```

### How it integrates with the existing batched turn loop

The current `_synthesize_all_samples` iterates `turn_idx` from 0 to `max_turns`, batching all samples that need that turn. The agentic loop nests inside this:

```python
for turn_idx in range(max_turns):
    # ... existing: collect samples needing this turn ...

    role = turn_order[turn_idx % len(turn_order)]

    if role == Role.ASSISTANT and has_tools:
        # AGENTIC LOOP: replaces single inference for this turn
        active_indices = list(sample_indices)  # all samples at this turn
        while active_indices:
            # 1. Batch inference for all active samples
            prompts = [build_prompt(i) for i in active_indices]
            results = inference_engine.infer(prompts)

            # 2. Parse results — split into tool-callers and responders
            still_active = []
            for idx, result in zip(active_indices, results):
                tool_call = parse_tool_call(result)
                if tool_call is None:
                    histories[idx].append(Message(role=Role.ASSISTANT, content=result))
                else:
                    # 3. Resolve tool outputs
                    tool_result = resolve_output(tool_call, ...)
                    histories[idx].append(...)  # assistant tool_call message
                    histories[idx].append(...)  # tool result message
                    still_active.append(idx)

            active_indices = still_active
            # 4. Next iteration: only samples still calling tools
    else:
        # Existing behavior for USER turns / non-tool ASSISTANT turns
        ...
```

### Batching across samples

At each step of the agentic loop:
1. All "active" samples (still calling tools) are batched for generation
2. Parse results — samples with tool calls continue, samples without are done
3. Resolve tool outputs for continuing samples (DETERMINISTIC: instant, GENERATED: batched LLM)
4. Next step: batch only still-active samples

This mirrors the existing pattern where samples finish at different turn counts.

### Safety: max tool calls per turn

To prevent infinite loops, enforce a `max_tool_calls_per_turn` limit (default: 5). If reached, the agent's next response is used as-is (no further tool calls parsed).

### Filtering

If a conversation with `available_tools` produces zero tool calls across all turns, it is filtered out post-synthesis (same pattern as existing empty message filtering). No retry — the combination of auto-injected framing and persona should be sufficient.

---

## Universal Output Format

Output follows the HuggingFace/OpenAI standard for maximum training compatibility. Works directly with `apply_chat_template(messages, tools=tools)`.

### Complete output row structure

```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "SearchOrders",
        "description": "Look up order details by customer ID",
        "parameters": {
          "type": "object",
          "properties": {
            "customer_id": {"type": "string", "description": "The customer's ID"}
          },
          "required": ["customer_id"]
        }
      }
    }
  ],
  "messages": [
    {"role": "system", "content": "You are a customer support agent..."},
    {"role": "user", "content": "Can you check my order?"},
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_001",
        "type": "function",
        "function": {
          "name": "SearchOrders",
          "arguments": "{\"customer_id\": \"CUST002\"}"
        }
      }]
    },
    {
      "role": "tool",
      "tool_call_id": "call_001",
      "content": "{\"order_id\": \"ORD-001\", \"status\": \"delivered\"}"
    },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_002",
        "type": "function",
        "function": {
          "name": "ProcessReturn",
          "arguments": "{\"order_id\": \"ORD-001\", \"reason\": \"customer request\"}"
        }
      }]
    },
    {
      "role": "tool",
      "tool_call_id": "call_002",
      "content": "{\"return_id\": \"RET-001\", \"status\": \"approved\"}"
    },
    {"role": "assistant", "content": "Your return has been processed. Return ID: RET-001."}
  ]
}
```

Key details:
- `arguments` is a **JSON string** — matches Nemotron-Agentic-v1 and OpenAI convention
- `content` is `null` on tool-call assistant messages (no text alongside the call)
- `id`/`tool_call_id` are auto-generated incrementing identifiers (e.g., `call_001`)
- `role: "tool"` for results, linked by `tool_call_id`
- `tools` is a top-level field, separate from `messages`

### How `apply_chat_template` uses this

At training time, the tokenizer formats tools into the model's native representation:

```python
tokenizer.apply_chat_template(
    row["messages"],
    tools=row["tools"],
    tokenize=True,
)
```

Each model family handles this differently:
- **Llama 3**: Injects tool definitions into system prompt, uses `<|python_tag|>` for calls
- **Qwen 2.5**: Uses `<tool_call>` / `</tool_call>` special tokens
- **Mistral**: Uses `[AVAILABLE_TOOLS]` / `[TOOL_CALLS]` markers
- **Gemma**: Injects as structured text in system message

The output format is model-agnostic — `apply_chat_template` handles the conversion.

---

## Pipeline Flow

```
1. DatasetPlanner.plan()            -> sampled attributes per sample
2. DataSynthesizer.synthesize()     -> generated attributes per sample
3. ConversationSynthesizer:
   a. Resolve tools for each multiturn_attribute:
      - Look up ToolAttribute objects from GeneralSynthesisParams.tools
        using MultiTurnAttribute.available_tools ids
      - For DETERMINISTIC tools: sample one output per tool per conversation
      - For environment: resolve {placeholders} in template (if present)
   b. Build tool catalog text + tool definitions dict
   c. Guide conversation (guider sees tool availability, not specifics)
   d. Execute participant turns:
      - USER turn: existing behavior
      - ASSISTANT turn (no tools): existing behavior
      - ASSISTANT turn (tools available): agentic loop
        i.   Generate response (LLM sees persona + tool catalog + history + guide)
        ii.  Parse for <tool_call> tags
        iii. If tool call: resolve output, append messages, continue loop
        iv.  If natural language: append response, turn done
        v.   Safety: stop after max_tool_calls_per_turn
   e. Post-process:
      - Filter conversations with zero tool calls (when tools were available)
      - Filter conversations with empty messages (existing behavior)
4. AttributeTransformer.transform() -> transformed attributes
5. Output: dict with "tools" (definitions) + "messages" (standard format)
```

---

## ToolExecutor

New class in `src/oumi/core/synthesis/tool_executor.py`. Handles output resolution and tool call parsing. Does NOT generate arguments — the agent does that naturally.

### Interface

```python
class ToolExecutor:
    """Resolves tool outputs and parses tool calls from LLM responses.

    Used by ConversationSynthesizer during the agentic turn loop.
    """

    def __init__(
        self,
        tools: list[ToolAttribute],
        inference_engine: BaseInferenceEngine | None = None,
        inference_config: InferenceConfig | None = None,
    ):
        """Initialize with tool definitions.

        Args:
            tools: Tool definitions for this conversation.
            inference_engine: Required for GENERATED strategy tools.
            inference_config: Config for GENERATED tool output inference.
        """
        self._tools_by_name = {t.name: t for t in tools}

    def resolve_output(
        self,
        tool_call: dict,                    # {"name": "...", "arguments": {...}}
        tool_config: ToolAttribute,
        deterministic_output: str | None,   # pre-selected for this conversation
        environment: "EnvironmentManager | None" = None,
    ) -> str:
        """Resolve tool output.

        DETERMINISTIC: returns pre-selected output.
        GENERATED: runs LLM simulation conditioned on tool config + args
                   + optional environment snapshot.
        """

    def resolve_outputs_batch(
        self,
        tool_calls: list[dict],
        tool_configs: list[ToolAttribute],
        deterministic_outputs: list[str | None],
        environments: list["EnvironmentManager | None"],
    ) -> list[str]:
        """Batch resolve tool outputs.

        DETERMINISTIC outputs are resolved instantly.
        GENERATED outputs are batched into a single LLM inference call.
        """

    def parse_tool_call(self, response: str) -> dict | None:
        """Parse <tool_call> tags from agent response.

        Returns {"name": "...", "arguments": {...}} or None.
        Handles malformed JSON gracefully (returns None).
        """

    def sample_deterministic_outputs(
        self,
        tools: list[ToolAttribute],
    ) -> dict[str, str]:
        """Sample one deterministic output per tool for a conversation.

        Called once at conversation initialization. Returns
        {tool_id: selected_output_json_string}.
        Only includes tools with DETERMINISTIC strategy.
        """
```

### GENERATED simulator prompt

```
You are simulating the tool "{name}".

Description: {description}
Parameters: {parameters}
Output schema: {output_schema}
{output_instruction}

{optional: Current environment state:}
{environment_snapshot}

The agent called this tool with arguments:
{tool_call_args}

Produce a realistic JSON response based on the arguments.
Output ONLY valid JSON, no explanation.
```

---

## EnvironmentManager (Optional, Phase 2)

New class in `src/oumi/core/synthesis/environment_manager.py`. Only instantiated when `MultiTurnAttribute.environment` is defined.

```python
class EnvironmentManager:
    """Manages mutable environment state for stateful tool interactions.

    When present, the environment snapshot is injected into GENERATED
    tool output prompts, allowing tool simulators to produce contextually
    consistent outputs.
    """
    def __init__(self, entities: dict[str, Any], context: str): ...
    def get_snapshot(self) -> dict: ...
    def apply_updates(self, updates: dict[str, Any]) -> None: ...
    def get_history(self) -> list[dict]: ...
```

When present: environment snapshot injected into GENERATED prompts, tools with `mutates_environment: true` can return updates.
When absent: tools operate statelessly.

---

## Example Config

```yaml
strategy: GENERAL
num_samples: 10
output_path: agentic_tool_dataset.jsonl

strategy_params:
  sampled_attributes:
    - id: customer_id
      name: Customer ID
      description: Unique customer identifier
      possible_values:
        - {id: cust1, name: "CUST001", description: "Customer 001"}
        - {id: cust2, name: "CUST002", description: "Customer 002"}

  generated_attributes:
    - id: customer_name
      instruction_messages:
        - role: SYSTEM
          content: "Generate a realistic customer name. Output only the name."
        - role: USER
          content: "Generate a name."

  tools:
    - id: search_orders
      name: SearchOrders
      description: "Look up order details by customer ID"
      parameters:
        type: object
        properties:
          customer_id: {type: string, description: "The customer's ID"}
        required: [customer_id]
      output_strategy: GENERATED
      generated_output:
        instruction: >
          Return realistic order details matching the queried customer ID.
          Include order ID, status (processing/shipped/delivered), item name,
          and total price. Make the result consistent with conversation context.
      output_schema:
        type: object
        properties:
          order_id: {type: string}
          status: {type: string}
          item: {type: string}
          total: {type: string}

    - id: check_return_eligibility
      name: CheckReturnEligibility
      description: "Check if an order is eligible for return"
      parameters:
        type: object
        properties:
          order_id: {type: string, description: "The order ID to check"}
        required: [order_id]
      output_strategy: GENERATED
      output_instruction: "Check eligibility based on order status and return window"
      output_schema:
        eligible: {type: boolean}
        window_remaining: {type: integer}
        reason: {type: string}

    - id: process_return
      name: ProcessReturn
      description: "Initiate a return for an order"
      parameters:
        type: object
        properties:
          order_id: {type: string, description: "The order to return"}
          reason: {type: string, description: "Reason for return"}
        required: [order_id, reason]
      output_strategy: GENERATED
      output_instruction: "Generate a return confirmation with return ID"

  multiturn_attributes:
    - id: support_conversation
      min_turns: 4
      max_turns: 8
      available_tools: [search_orders, check_return_eligibility, process_return]

      role_instruction_messages:
        USER: |
          You are {customer_name} (ID: {customer_id}), contacting support about your order.
          Be cooperative and provide information when asked.
        ASSISTANT: |
          You are a customer support agent for BrightTech Electronics.

      output_system_prompt: |
        You are a customer support agent with access to order management tools.

  passthrough_attributes:
    - support_conversation

inference_config:
  model:
    model_name: claude-sonnet-4-20250514
  engine: ANTHROPIC
  generation:
    max_new_tokens: 8192
    temperature: 0.7
  remote_params:
    num_workers: 50
    politeness_policy: 60
```

---

## Implementation Plan

### Phase 1: Registry & Config Wiring

Wire tools into the synthesis config so they're available to downstream code.

| Task | File | Details |
|------|------|---------|
| Add `EnvironmentConfig` dataclass | `tool_params.py` | `entities: dict`, `context: str` |
| Add `tools` field to `GeneralSynthesisParams` | `synthesis_params.py` | `tools: list[ToolAttribute] \| None = None` |
| Add `environment` field to `MultiTurnAttribute` | `synthesis_params.py` | `environment: EnvironmentConfig \| None = None` |
| Add cross-validation | `synthesis_params.py` | `available_tools` ids must exist in `tools` list |
| Tests | `test_tool_params.py`, `test_synthesis_params.py` | Validation tests for new fields and cross-validation |

### Phase 2: ToolExecutor

Standalone class for parsing tool calls and resolving outputs.

| Task | File | Details |
|------|------|---------|
| `parse_tool_call()` | `tool_executor.py` | Parse `<tool_call>` tags, extract JSON, handle malformed |
| `sample_deterministic_outputs()` | `tool_executor.py` | Random weighted selection per tool per conversation |
| `resolve_output()` | `tool_executor.py` | DETERMINISTIC: return pre-selected; GENERATED: build simulator prompt + LLM call |
| `resolve_outputs_batch()` | `tool_executor.py` | Batch GENERATED resolutions into single inference |
| `build_tool_catalog()` | `tool_executor.py` | ToolAttribute list → formatted text for synthesis prompt |
| `build_tool_definitions()` | `tool_executor.py` | ToolAttribute list → OpenAI-format `tools` array for output |
| Tests | `test_tool_executor.py` | Unit tests for parsing, resolution, catalog generation |

### Phase 3: Agentic Turn Loop

Modify `conversation_synthesizer.py` to support tool-calling assistant turns.

| Task | File | Details |
|------|------|---------|
| Tool catalog auto-injection into ASSISTANT persona | `conversation_synthesizer.py` | Append tool catalog + format instructions when `available_tools` is non-empty |
| Guider prompt extension | `conversation_synthesizer.py` | Inject "write action-oriented instructions" when tools available |
| Agentic loop in `_synthesize_all_samples` | `conversation_synthesizer.py` | For ASSISTANT turns with tools: generate → parse → resolve → loop/break |
| Batched agentic loop | `conversation_synthesizer.py` | Batch active samples at each loop step, split completers vs continuers |
| `max_tool_calls_per_turn` safety limit | `conversation_synthesizer.py` | Default 5, prevents infinite tool-call loops |
| Zero-tool-use filtering | `conversation_synthesizer.py` | Filter conversations where tools were available but none were called |
| Output format: attach `tools` definitions to output | `conversation_synthesizer.py` | Output dict gets `"tools"` key alongside `"messages"` |
| Tests | `test_conversation_synthesizer.py` | Agentic loop tests, filtering tests, batching tests |

### Phase 4: EnvironmentManager (Optional)

Can be deferred — everything in Phases 1-3 works without it.

| Task | File | Details |
|------|------|---------|
| `EnvironmentManager` class | `environment_manager.py` | `get_snapshot()`, `apply_updates()`, placeholder resolution |
| Integration with ToolExecutor | `tool_executor.py` | Pass snapshot to GENERATED prompts |
| Integration with ConversationSynthesizer | `conversation_synthesizer.py` | Initialize env per conversation, pass to ToolExecutor |
| Tests | `test_environment_manager.py` | State mutation, snapshot, history tracking |

### Phase 5: Example Config & Docs

| Task | File | Details |
|------|------|---------|
| Example config | `configs/examples/synthesis/agentic_tool_synth.yaml` | Working example matching the config in this doc |
| Documentation | `docs/` | Usage guide for tool synthesis |

---

## Files Changed

| File | Change | Phase |
|------|--------|-------|
| `src/oumi/core/types/conversation.py` | `Role.TOOL` already exists — no change needed | — |
| `src/oumi/core/configs/params/tool_params.py` | Add `EnvironmentConfig` (existing: `ToolAttribute`, `ToolOutputStrategy`, `ToolOutputValue`) | 1 |
| `src/oumi/core/configs/params/synthesis_params.py` | Import from `tool_params`; add `tools` to `GeneralSynthesisParams`; add `environment` to `MultiTurnAttribute`; add cross-validation | 1 |
| `src/oumi/core/synthesis/tool_executor.py` | New — `ToolExecutor` class (output resolution + tool call parsing + catalog/definitions generation) | 2 |
| `src/oumi/core/synthesis/environment_manager.py` | New — `EnvironmentManager` class (optional) | 4 |
| `src/oumi/core/synthesis/conversation_synthesizer.py` | Add guider prompt extension; add agentic turn loop; add tool catalog injection; add output format attachment; add zero-tool-use filtering | 3 |
| `configs/examples/synthesis/agentic_tool_synth.yaml` | New example config | 5 |
| `tests/unit/core/synthesis/test_tool_executor.py` | New tests | 2 |
| `tests/unit/core/synthesis/test_environment_manager.py` | New tests | 4 |
| `tests/unit/core/synthesis/test_conversation_synthesizer.py` | Add agentic turn loop tests | 3 |
| `tests/unit/core/configs/params/test_tool_params.py` | New — ToolAttribute + EnvironmentConfig validation tests | 1 |
