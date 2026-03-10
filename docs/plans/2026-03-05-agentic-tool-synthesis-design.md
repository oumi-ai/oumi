# Agentic Tool Synthesis Design

## Overview

Extend the multi-turn conversation synthesis system to support agentic tool use. Users define tools with output strategies (deterministic or LLM-generated), the conversation planner decides when the agent calls tools, and tool outputs are resolved and integrated into the conversation. An optional environment layer enables stateful tool interactions when needed.

**Core principle: Tools are the centerpiece. Environment is optional.**

---

## Goals

- Enable users to define tools with schemas, output strategies, and descriptions
- Support deterministic (user-defined) and generated (LLM-simulated) tool outputs
- Support multi-tool calls within a single assistant turn (chained tool use)
- Optionally support mutable environment state for stateful tool interactions
- Keep tool calls as sub-steps within an ASSISTANT turn (no impact on `min_turns`/`max_turns`)
- Produce standard chat-format output with `TOOL_CALL` and `TOOL` role messages inline
- Stay inline with the existing attribute pipeline (`{placeholder}` templates, sampled/generated attributes)

---

## Tool Output Strategies

| Strategy | Source | When to use |
|----------|--------|-------------|
| **DETERMINISTIC** | User-defined outputs, one selected per conversation | Predictable tools, bounded output variety |
| **GENERATED** | LLM simulates output given schema + args + context | Open-ended tools, when realism matters |

### DETERMINISTIC details

- Single output in the list: always returns that output
- Multiple outputs: one randomly selected per conversation (weighted by `sample_rate`, uniform if unset)
- Selection happens once at conversation initialization, not per-call

### GENERATED details

- LLM receives: tool config + args + conversation history + optional environment snapshot
- `output_instruction` provides hints for realistic simulation
- `output_schema` gives the LLM structure guidance for the response format

---

## Config Schema

### New: `ToolOutputStrategy`

```python
class ToolOutputStrategy(str, Enum):
    DETERMINISTIC = "deterministic"  # user-defined outputs
    GENERATED = "generated"          # LLM simulates output
```

### New: `ToolOutputValue`

```python
@dataclass
class ToolOutputValue:
    output: str                       # tool output (JSON string)
    sample_rate: float | None = None  # selection weight, uniform if unset
```

### New: `ToolConfig`

Defined at the `GeneralSynthesisParams` level. Referenced by name from `MultiTurnAttribute.available_tools`.

```python
@dataclass
class ToolConfig:
    id: str                                       # referenced by available_tools
    name: str                                     # tool name (e.g., "SearchOrders")
    description: str                              # what the tool does
    parameters: dict[str, Any] = {}               # parameter schema {name: {type, required, description}}

    output_strategy: ToolOutputStrategy = ToolOutputStrategy.GENERATED

    # DETERMINISTIC mode
    deterministic_outputs: list[ToolOutputValue] = []

    # GENERATED mode
    output_instruction: str = ""                  # hint for LLM simulator
    output_schema: dict[str, Any] = {}            # expected output structure

    # Optional environment interaction (only when environment is defined)
    mutates_environment: bool = False
```

### New: `EnvironmentConfig` (Optional)

```python
@dataclass
class EnvironmentConfig:
    entities: dict[str, Any] = {}   # structured state template with {placeholders}
    context: str = ""               # domain rules/constraints template with {placeholders}
```

Both fields support `{placeholder}` resolution from sampled/generated attributes.

### Changes to `GeneralSynthesisParams`

```python
tools: list[ToolConfig] = []
```

### Changes to `MultiTurnAttribute`

```python
available_tools: list[str] = []                    # list of ToolConfig ids
environment: EnvironmentConfig | None = None       # optional environment template
```

---

## Planner Extension

When `available_tools` is non-empty, the planner prompt is extended with:

1. **Tool catalog** — tool names, descriptions, and parameter summaries
2. **Environment overview** (only when environment is defined) — summary of available entities and context
3. **Extended instruction** — planner may annotate ASSISTANT turns with `tool_calls`

### Extended planner JSON schema

```json
[
  {"turn": 1, "role": "USER", "instruction": "ask about returning their order"},
  {
    "turn": 2,
    "role": "ASSISTANT",
    "tool_calls": [
      {"tool": "search_orders", "instruction": "look up the customer's order using their customer ID"},
      {"tool": "check_return_eligibility", "instruction": "check if the order is eligible for return based on search results"}
    ],
    "instruction": "summarize return eligibility and explain next steps"
  },
  {"turn": 3, "role": "USER", "instruction": "confirm they want to proceed with the return"},
  {
    "turn": 4,
    "role": "ASSISTANT",
    "tool_calls": [
      {"tool": "process_return", "instruction": "initiate the return for the order"}
    ],
    "instruction": "confirm the return has been processed and explain the timeline"
  }
]
```

- `tool_calls` is an optional list on any ASSISTANT turn
- Each entry has `tool` (ToolConfig id) and `instruction` (guides arg generation)
- `instruction` on the turn itself guides the ASSISTANT's final response after all tool results
- `_parse_plan` passes through the new fields transparently
- Turn count is unaffected — a tool-call turn still counts as one turn

---

## Pipeline Flow

```
1. DatasetPlanner.plan()            -> sampled attributes per sample
2. DataSynthesizer.synthesize()     -> generated attributes per sample
3. ConversationSynthesizer:
   a. Resolve DETERMINISTIC tool outputs per conversation (sample once)
   b. Resolve {placeholders} in environment template (if present) -> EnvironmentManager
   c. Plan conversation (planner sees tools + optional environment)
   d. Execute turns:
      - Regular turns: existing behavior
      - Tool-call turns:
        For each tool_call in tool_calls:
          i.   Generate tool_call args (batched LLM, sees prior tool results)
          ii.  Resolve tool output (DETERMINISTIC: use pre-selected output,
               GENERATED: batched LLM conditioned on args + optional environment)
          iii. Apply environment updates (if environment defined and tool mutates)
          iv.  Append TOOL_CALL + TOOL messages to conversation
        Then:
          v.   Generate ASSISTANT response (batched LLM, sees all tool results)
4. AttributeTransformer.transform() -> transformed attributes
5. Output
```

---

## ConversationSynthesizer Changes

### Initialization per sample

```python
# Resolve DETERMINISTIC tool outputs for this conversation
tool_outputs = {}
for tool_id in multiturn_attribute.available_tools:
    tool_config = tools_by_id[tool_id]
    if tool_config.output_strategy == ToolOutputStrategy.DETERMINISTIC:
        tool_outputs[tool_id] = select_output(tool_config.deterministic_outputs)

# Resolve environment (if present)
env_manager = None
if multiturn_attribute.environment:
    resolved_env = attribute_formatter.resolve(
        multiturn_attribute.environment, sample
    )
    env_manager = EnvironmentManager(
        entities=resolved_env["entities"],
        context=resolved_env["context"],
    )
```

### Extended turn loop

```python
for turn_idx in range(target_turns):
    turn_plan = parsed_turn_plans[turn_idx]

    if turn_plan.get("tool_calls"):
        # Process each tool call sequentially within this turn
        for tool_call_plan in turn_plan["tool_calls"]:
            tool_config = tools_by_id[tool_call_plan["tool"]]

            # Batch: Generate tool_call args for all active samples
            # LLM sees conversation history including prior tool results from this turn
            tool_call_msgs = batch_generate_args(
                tool_config, tool_call_plan, conversations, environments
            )

            # Resolve tool output based on strategy
            if tool_config.output_strategy == ToolOutputStrategy.DETERMINISTIC:
                tool_result_msgs = [
                    make_tool_message(tool_outputs[sample_idx][tool_config.id])
                    for sample_idx in active_sample_indices
                ]
            else:  # GENERATED
                tool_result_msgs = batch_simulate(
                    tool_config, tool_call_msgs, environments
                )

            # Append TOOL_CALL + TOOL to each conversation
            for i, conv in enumerate(active_conversations):
                conv.messages.append(tool_call_msgs[i])
                conv.messages.append(tool_result_msgs[i])

        # Batch: Generate ASSISTANT response with all tool results in context
        responses = batch_generate_response(
            conversations, turn_plan["instruction"], personas
        )
        for i, conv in enumerate(active_conversations):
            conv.messages.append(responses[i])
    else:
        # Existing behavior unchanged
        ...
```

---

## ToolExecutor

New class in `src/oumi/core/synthesis/tool_executor.py`.

### Responsibilities

**Argument generation** — Prompt the model with tool name/description, `instruction` from the plan, conversation history, and optional environment state. Model returns a tool_call message with arguments as JSON.

**Output resolution** — Either return the pre-selected DETERMINISTIC output, or run LLM simulation for GENERATED mode.

**GENERATED simulator prompt:**

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

Produce a realistic response based on the arguments{optional: and environment state}.

Return JSON:
{
  "status": "success" | "error",
  "output": { ... }
  {optional: "environment_updates": { ... }}
}
```

### Interface

```python
class ToolExecutor:
    def generate_args(
        self,
        tool_config: ToolConfig,
        tool_call_plan: dict,
        conversation: Conversation,
        environment: EnvironmentManager | None,
    ) -> Message:
        """Generate TOOL_CALL message with arguments."""

    def resolve_output(
        self,
        tool_config: ToolConfig,
        tool_call_msg: Message,
        environment: EnvironmentManager | None,
        deterministic_output: str | None,
    ) -> Message:
        """Resolve TOOL message. Uses deterministic_output if provided,
        otherwise runs LLM simulation."""
```

---

## EnvironmentManager (Optional)

New class in `src/oumi/core/synthesis/environment_manager.py`. Only instantiated when `MultiTurnAttribute.environment` is defined.

```python
class EnvironmentManager:
    """Manages mutable environment state for a single conversation/sample."""

    def __init__(self, entities: dict[str, Any], context: str):
        self._state = copy.deepcopy(entities)
        self._context = context
        self._history: list[dict] = []

    def get_snapshot(self) -> dict:
        """Return current state + context for injection into simulator prompt."""
        return {"entities": self._state, "context": self._context}

    def apply_updates(self, updates: dict[str, Any]) -> None:
        """Deep-merge updates into current state. Log the change."""
        self._history.append({"step": len(self._history), "updates": copy.deepcopy(updates)})
        deep_merge(self._state, updates)

    def get_history(self) -> list[dict]:
        """Return mutation history for debugging/observability."""
        return self._history
```

---

## Message Format

### Role enum change

Add `TOOL_CALL` to the `Role` enum in `conversation.py`:

```python
class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    TOOL_CALL = "tool_call"   # NEW
```

### Output format

A synthesized conversation with multi-tool chaining:

```
[SYSTEM]    "You are a helpful customer support agent"
[USER]      "Hi, I'd like to return my hiking boots from my recent order"
[TOOL_CALL] {"name": "SearchOrders", "arguments": {"customer_id": "CUST002"}}
[TOOL]      {"status": "success", "output": {"order_id": "ORD-001", "item": "Hiking Boots", "status": "delivered"}}
[TOOL_CALL] {"name": "CheckReturnEligibility", "arguments": {"order_id": "ORD-001"}}
[TOOL]      {"status": "success", "output": {"eligible": true, "window_remaining": 75}}
[ASSISTANT] "Your order with Hiking Boots is eligible for return. You have 75 days remaining..."
[USER]      "Great, please process the return"
[TOOL_CALL] {"name": "ProcessReturn", "arguments": {"order_id": "ORD-001", "reason": "customer request"}}
[TOOL]      {"status": "success", "output": {"return_id": "RET-001", "label_url": "https://..."}}
[ASSISTANT] "Your return has been processed! Return ID: RET-001. Here's your shipping label..."
```

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
        customer_id: {type: string, required: true}
      output_strategy: DETERMINISTIC
      deterministic_outputs:
        - output: '{"order_id": "ORD-001", "status": "delivered", "item": "Running Shoes", "price": 119.99}'
          sample_rate: 0.5
        - output: '{"order_id": "ORD-002", "status": "shipped", "item": "Hiking Boots", "price": 89.99}'
          sample_rate: 0.5

    - id: check_return_eligibility
      name: CheckReturnEligibility
      description: "Check if an order is eligible for return"
      parameters:
        order_id: {type: string, required: true}
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
        order_id: {type: string, required: true}
        reason: {type: string, required: true}
      output_strategy: GENERATED
      output_instruction: "Generate a return confirmation with return ID and shipping label"

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
          You are a helpful customer support agent. Use your tools to look up
          information and process requests. Always verify details before taking actions.

      output_system_prompt: |
        You are a helpful customer support agent with access to order management tools.

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
| `src/oumi/core/types/conversation.py` | Add `Role.TOOL_CALL` to enum |
| `src/oumi/core/configs/params/synthesis_params.py` | Add `ToolConfig`, `ToolOutputStrategy`, `ToolOutputValue`, `EnvironmentConfig`; extend `GeneralSynthesisParams` with `tools`; extend `MultiTurnAttribute` with `available_tools` and `environment` |
| `src/oumi/core/synthesis/environment_manager.py` | New file — `EnvironmentManager` class |
| `src/oumi/core/synthesis/tool_executor.py` | New file — `ToolExecutor` class |
| `src/oumi/core/synthesis/conversation_synthesizer.py` | Extend planner prompt with tool catalog + optional environment; extend turn loop for tool-call turns with multi-tool support; per-sample tool output selection and optional environment initialization |
| `configs/examples/synthesis/agentic_tool_synth.yaml` | New example config |
| `tests/unit/core/synthesis/test_environment_manager.py` | New tests |
| `tests/unit/core/synthesis/test_tool_executor.py` | New tests |
| `tests/unit/core/synthesis/test_conversation_synthesizer.py` | Add tool-call turn tests |
| `tests/unit/core/configs/params/test_synthesis_params.py` | Add ToolConfig, EnvironmentConfig validation tests |
