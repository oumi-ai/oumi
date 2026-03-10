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
| **DETERMINISTIC** | User-defined outputs, one selected per conversation | Predictable tools, bounded output variety |
| **GENERATED** | LLM simulates output given schema + args + context | Open-ended tools, when realism matters |

### DETERMINISTIC

- Single output in the list: always returns that output
- Multiple outputs: one randomly selected per conversation (weighted by `sample_rate`, uniform if unset)
- Selection happens once at conversation initialization, not per-call

### GENERATED

- LLM receives: tool config + args + conversation history + optional environment snapshot
- `output_instruction` provides hints for realistic simulation
- `output_schema` gives the LLM structure guidance for the response format

---

## Config Schema

### ToolAttribute (flattened, in `tool_params.py`)

```python
class ToolOutputStrategy(str, Enum):
    DETERMINISTIC = "deterministic"
    GENERATED = "generated"

@dataclass
class ToolOutputValue:
    output: str
    sample_rate: float | None = None

@dataclass
class ToolAttribute:
    id: str                                          # referenced by available_tools
    name: str                                        # tool name (e.g., "SearchOrders")
    description: str                                 # what the tool does
    output_strategy: ToolOutputStrategy = ToolOutputStrategy.GENERATED

    # JSON Schema format — matches OpenAI/HuggingFace standard
    parameters: dict[str, Any] = {}
    # Example:
    # {
    #   "type": "object",
    #   "properties": {
    #     "customer_id": {"type": "string", "description": "The customer's ID"}
    #   },
    #   "required": ["customer_id"]
    # }

    # DETERMINISTIC fields
    deterministic_outputs: list[ToolOutputValue] = []

    # GENERATED fields
    output_instruction: str = ""
    output_schema: dict[str, Any] = {}

    # Optional environment interaction
    mutates_environment: bool = False
```

### EnvironmentConfig (Optional, in `tool_params.py`)

```python
@dataclass
class EnvironmentConfig:
    entities: dict[str, Any] = {}   # structured state template with {placeholders}
    context: str = ""               # domain rules/constraints with {placeholders}
```

### Changes to `GeneralSynthesisParams`

```python
tools: list[ToolAttribute] = []
```

### Changes to `MultiTurnAttribute`

```python
available_tools: list[str] = []                    # list of ToolAttribute ids
environment: EnvironmentConfig | None = None       # optional
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
    """Auto-generate tool catalog from ToolAttribute definitions."""
    # Produces formatted text like:
    # - SearchOrders: Look up order details by customer ID
    #   Parameters: customer_id (string, required) - The customer's ID
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

```python
# Inside an ASSISTANT turn
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
    tool_config = tools_by_id[tool_call["name"]]
    tool_result = resolve_output(tool_call, tool_config, deterministic_outputs, env)
    history.append(assistant_message(tool_calls=[tool_call]))
    history.append(tool_message(tool_call_id=..., content=tool_result))
    # Loop: agent sees tool result and decides next action
```

### Batching across samples

At each step of the agentic loop:
1. All "active" samples (still calling tools) are batched for generation
2. Parse results — samples with tool calls continue, samples without are done
3. Resolve tool outputs for continuing samples (DETERMINISTIC: instant, GENERATED: batched LLM)
4. Next step: batch only still-active samples

This mirrors the existing pattern where samples finish at different turn counts.

### Filtering

If a conversation with `available_tools` produces zero tool calls, it is filtered out post-synthesis (same pattern as existing empty message filtering). No retry — the combination of auto-injected framing and persona should be sufficient.

---

## Universal Output Format

Output follows the HuggingFace/OpenAI standard for maximum training compatibility. Works directly with `apply_chat_template(messages, tools=tools)`.

### Tool definitions (auto-generated from ToolAttribute)

```json
[
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
]
```

### Message format

```json
[
  {"role": "system", "content": "You are a customer support agent..."},
  {"role": "user", "content": "Can you check my order?"},
  {
    "role": "assistant",
    "tool_calls": [{
      "id": "call_001",
      "type": "function",
      "function": {
        "name": "SearchOrders",
        "arguments": {"customer_id": "CUST002"}
      }
    }]
  },
  {
    "role": "tool",
    "tool_call_id": "call_001",
    "name": "SearchOrders",
    "content": "{\"order_id\": \"ORD-001\", \"status\": \"delivered\"}"
  },
  {
    "role": "assistant",
    "tool_calls": [{
      "id": "call_002",
      "type": "function",
      "function": {
        "name": "ProcessReturn",
        "arguments": {"order_id": "ORD-001", "reason": "customer request"}
      }
    }]
  },
  {
    "role": "tool",
    "tool_call_id": "call_002",
    "name": "ProcessReturn",
    "content": "{\"return_id\": \"RET-001\", \"status\": \"approved\"}"
  },
  {"role": "assistant", "content": "Your return has been processed. Return ID: RET-001."}
]
```

Key details:
- `arguments` is a **dict** (not JSON string) — matches HuggingFace expectation
- `id`/`tool_call_id` included for compatibility (auto-generated, incrementing)
- `role: "tool"` for results, with both `name` and `tool_call_id`
- No custom `TOOL_CALL` role — tool calls are `assistant` messages with `tool_calls` field
- `role: "tool"` already exists or will be added to the Role enum

### Synthesis-time vs output format

During synthesis, the LLM generates `<tool_call>` tagged text. The system parses this and converts to the standard message format for output. These are separate concerns:

| Concern | Format |
|---------|--------|
| LLM generation (internal) | `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` |
| Output (training data) | Standard `assistant.tool_calls` / `tool` role messages |

---

## Pipeline Flow

```
1. DatasetPlanner.plan()            -> sampled attributes per sample
2. DataSynthesizer.synthesize()     -> generated attributes per sample
3. ConversationSynthesizer:
   a. Resolve DETERMINISTIC tool outputs per conversation (sample once)
   b. Resolve {placeholders} in environment template (if present)
   c. Guide conversation (guider sees tool availability, not specifics)
   d. Execute participant turns:
      - USER turn: existing behavior
      - ASSISTANT turn (no tools): existing behavior
      - ASSISTANT turn (tools available): agentic loop
        i.   Generate response (LLM sees persona + tool catalog + history + guide)
        ii.  Parse for <tool_call> tags
        iii. If tool call: resolve output, append messages, continue loop
        iv.  If natural language: append response, turn done
   e. Convert <tool_call> tagged messages to standard format
4. AttributeTransformer.transform() -> transformed attributes
5. Output (with tools definitions + standard messages)
```

---

## ToolExecutor

New class in `src/oumi/core/synthesis/tool_executor.py`. Simplified — only handles output resolution, not argument generation (the agent generates args naturally).

### Interface

```python
class ToolExecutor:
    def resolve_output(
        self,
        tool_call: dict,                    # {"name": "...", "arguments": {...}}
        tool_config: ToolAttribute,
        deterministic_output: str | None,   # pre-selected for this conversation
        environment: EnvironmentManager | None,
    ) -> str:
        """Resolve tool output.

        DETERMINISTIC: returns pre-selected output.
        GENERATED: runs LLM simulation conditioned on tool config + args
                   + optional environment snapshot.
        """

    def parse_tool_call(self, response: str) -> dict | None:
        """Parse <tool_call> tags from agent response.

        Returns {"name": "...", "arguments": {...}} or None.
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
```

---

## EnvironmentManager (Optional)

New class in `src/oumi/core/synthesis/environment_manager.py`. Only instantiated when `MultiTurnAttribute.environment` is defined.

```python
class EnvironmentManager:
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
      output_strategy: DETERMINISTIC
      deterministic_outputs:
        - output: '{"order_id": "ORD-001", "status": "delivered", "item": "Running Shoes"}'
          sample_rate: 0.5
        - output: '{"order_id": "ORD-002", "status": "shipped", "item": "Hiking Boots"}'
          sample_rate: 0.5

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

## Files Changed

| File | Change |
|------|--------|
| `src/oumi/core/types/conversation.py` | Add `Role.TOOL` to enum (if not present) |
| `src/oumi/core/configs/params/tool_params.py` | `ToolAttribute`, `ToolOutputStrategy`, `ToolOutputValue`, `EnvironmentConfig` (already created) |
| `src/oumi/core/configs/params/synthesis_params.py` | Import from `tool_params`; add `tools` to `GeneralSynthesisParams`; add `available_tools` and `environment` to `MultiTurnAttribute` |
| `src/oumi/core/synthesis/tool_executor.py` | New — `ToolExecutor` class (output resolution + tool call parsing) |
| `src/oumi/core/synthesis/environment_manager.py` | New — `EnvironmentManager` class (optional) |
| `src/oumi/core/synthesis/conversation_synthesizer.py` | Add guider prompt extension; add agentic turn loop; add tool catalog generation; add output format conversion; add zero-tool-use filtering |
| `configs/examples/synthesis/agentic_tool_synth.yaml` | New example config |
| `tests/unit/core/synthesis/test_tool_executor.py` | New tests |
| `tests/unit/core/synthesis/test_environment_manager.py` | New tests |
| `tests/unit/core/synthesis/test_conversation_synthesizer.py` | Add agentic turn loop tests |
| `tests/unit/core/configs/params/test_tool_params.py` | New — ToolAttribute validation tests |
