# Agentic Tool Synthesis Design

## Overview

Extend the multi-turn conversation synthesis system to support agentic tool use with stateful environment simulation. Users define tools and a structured environment that tools interact with. The planner decides which turns involve tool calls, the tool executor simulates tool behavior conditioned on environment state, and tool outputs mutate the environment for subsequent calls. The output is a standard chat-format conversation with `TOOL_CALL` and `TOOL` role messages inline.

Inspired by the SynthTools paper (ICML 2025 submission), which introduces a pipeline for generating synthetic tool ecosystems with LLM-based tool simulation conditioned on mutable metadata/environment state.

---

## Goals

- Enable users to define tools with rich schemas (parameters, output schemas, error modes)
- Introduce a structured, mutable environment that tools read from and write to
- Environment structure is user-defined; data values are populated via `{placeholder}` resolution from sampled/generated attributes
- Tool simulation is LLM-based: the simulator receives tool config + args + environment state and produces realistic outputs + state updates
- Keep tool calls as sub-steps within an ASSISTANT turn (no impact on `min_turns`/`max_turns`)
- Produce standard chat-format output compatible with the existing pipeline
- Stay inline with the existing attribute pipeline (`{placeholder}` templates, sampled/generated attributes)

---

## Architecture

### Approach: Extend existing synthesizer + new ToolExecutor + EnvironmentManager

Extend `MultiTurnAttribute`, `ConversationSynthesizer`, and `GeneralSynthesisParams` minimally. Extract tool execution and environment management into new classes. The environment is populated through the standard attribute pipeline using `{placeholder}` resolution.

### Pipeline flow

```
1. DatasetPlanner.plan()            -> sampled attributes per sample
2. DataSynthesizer.synthesize()     -> generated attributes per sample
3. ConversationSynthesizer:
   a. Resolve {placeholders} in environment template -> populated environment per sample
   b. Plan conversation (planner sees tools + environment)
   c. Execute turns:
      - Regular turns: existing behavior
      - Tool-call turns:
        i.   Generate tool_call args (batched LLM)
        ii.  Simulate tool output (batched LLM, conditioned on environment)
        iii. Apply environment updates
        iv.  Generate assistant response (batched LLM)
4. AttributeTransformer.transform() -> transformed attributes
5. Output
```

---

## Config Schema

### New: `ToolOutputMode`

```python
class ToolOutputMode(str, Enum):
    DETERMINISTIC = "deterministic"   # round-robin through fixed outputs
    SIMULATED = "simulated"           # LLM simulates tool behavior given environment
```

### New: `ToolConfig`

Defined at the `GeneralSynthesisParams` level. Referenced by name from `MultiTurnAttribute.available_tools`.

```python
class ToolConfig(BaseModel):
    id: str                              # referenced by available_tools
    name: str                            # tool name (e.g., "ReturnRequestValidator")
    description: str                     # what the tool does
    parameters: dict[str, Any] = {}      # parameter schema {name: {type, required, description}}
    output_schema: dict[str, Any] = {}   # expected output fields
    error_modes: list[str] = []          # documented error conditions
    output_mode: ToolOutputMode = ToolOutputMode.SIMULATED

    # DETERMINISTIC mode
    deterministic_outputs: list[str] = []

    # SIMULATED mode
    output_instruction: str = ""         # optional hint for the LLM simulator
    mutates_environment: bool = True     # whether this tool updates environment state
```

### New: `EnvironmentConfig`

```python
class EnvironmentConfig(BaseModel):
    entities: dict[str, Any] = {}   # structured state template with {placeholders}
    context: str = ""               # domain rules/constraints template with {placeholders}
```

The `entities` and `context` fields are templates. They contain `{placeholder}` references to sampled/generated attributes, which are resolved per-sample before the conversation synthesizer uses them.

### Changes to `GeneralSynthesisParams`

```python
tools: list[ToolConfig] = []
```

### Changes to `MultiTurnAttribute`

```python
available_tools: list[str] = []                    # list of ToolConfig ids
environment: EnvironmentConfig | None = None       # environment template for this conversation
```

---

## Environment Design

### User defines structure, attributes fill values

The environment is a user-authored template. Users define the full structure (entity types, fields, relationships). Data values come from the existing attribute pipeline via `{placeholder}` syntax.

```yaml
strategy_params:
  sampled_attributes:
    - id: customer_id
      values: [{value: "CUST001"}, {value: "CUST002"}, {value: "CUST003"}]
    - id: order_status
      values: [{value: "delivered"}, {value: "shipped"}]
    - id: product_name
      values: [{value: "Running Shoes"}, {value: "Hiking Boots"}]
    - id: product_price
      values: [{value: "119.99"}, {value: "89.99"}]

  generated_attributes:
    - id: customer_name
      instruction: "Generate a realistic customer name"
    - id: customer_email
      instruction: "Generate a realistic email for a customer named {customer_name}"
    - id: return_policy
      instruction: "Generate a return policy for footwear products"

  multiturn_attributes:
    - id: support_conv
      min_turns: 3
      max_turns: 6
      available_tools: [return_validator, search_orders, status_tracker]
      environment:
        entities:
          customers:
            "{customer_id}":
              name: "{customer_name}"
              email: "{customer_email}"
              orders: ["{customer_id}_ORD001"]
          orders:
            "{customer_id}_ORD001":
              customer_id: "{customer_id}"
              items:
                - id: "ITEM001"
                  name: "{product_name}"
                  price: "{product_price}"
              status: "{order_status}"
          returns: {}
        context: |
          {return_policy}
      role_instruction_messages:
        user: "You are {customer_name}, contacting support about your order"
        assistant: "You are a helpful customer support agent"
```

### How it works per sample

1. `DatasetPlanner` samples: `customer_id=CUST002`, `order_status=delivered`, etc.
2. `DataSynthesizer` generates: `customer_name="Maria Garcia"`, `customer_email="maria@example.com"`, etc.
3. `ConversationSynthesizer` resolves placeholders in the environment template:

```python
# Sample-specific populated environment:
{
  "entities": {
    "customers": {
      "CUST002": {
        "name": "Maria Garcia",
        "email": "maria@example.com",
        "orders": ["CUST002_ORD001"]
      }
    },
    "orders": {
      "CUST002_ORD001": {
        "customer_id": "CUST002",
        "items": [{"id": "ITEM001", "name": "Hiking Boots", "price": "89.99"}],
        "status": "delivered"
      }
    },
    "returns": {}
  },
  "context": "90-day return policy for footwear..."
}
```

4. This populated environment is passed to the `EnvironmentManager` for this sample's conversation.

### Environment compatibility with tools

Since the user defines both the tool configs and the environment structure, compatibility is ensured by design. The user knows that `ReturnRequestValidator` needs `orders` and `customers` in the environment, so they include them. The tool's `parameters`, `output_schema`, and `error_modes` give the LLM tool simulator enough context to produce outputs consistent with the environment state.

---

## EnvironmentManager

New class in `src/oumi/core/synthesis/environment_manager.py`.

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
        self._history.append({
            "step": len(self._history),
            "updates": copy.deepcopy(updates),
        })
        deep_merge(self._state, updates)

    def get_history(self) -> list[dict]:
        """Return mutation history for debugging/observability."""
        return self._history
```

One `EnvironmentManager` instance per sample. Created after placeholder resolution, lives for the duration of the conversation synthesis.

---

## ToolExecutor

New class in `src/oumi/core/synthesis/tool_executor.py`.

### Responsibilities

**Step 1 -- Argument generation**

Prompt the model with the tool name/description, `tool_args_instruction` from the plan, current conversation history, and environment state. Model returns a tool_call message with arguments as a JSON dict.

**Step 2 -- Tool simulation**

| Mode | Behavior |
|------|----------|
| `DETERMINISTIC` | Round-robin through `deterministic_outputs`. No environment interaction. |
| `SIMULATED` | LLM-based two-stage simulation (following SynthTools paper): |

For SIMULATED mode, the simulator prompt follows a two-stage pattern:

1. **Parameter validation**: Check if args are valid given tool schema + environment state. If invalid, return an appropriate error from `error_modes`.
2. **Response generation**: Given valid args + environment state, produce realistic output matching `output_schema` and return `environment_updates` if `mutates_environment` is true.

Simulator prompt template:

```
You are simulating the tool "{name}".

Description: {description}
Parameters: {parameters}
Output schema: {output_schema}
Error modes: {error_modes}
{output_instruction}

Current environment state:
{environment_snapshot}

The agent called this tool with arguments:
{tool_call_args}

First, validate the parameters against the tool schema and environment state.
If invalid, return an error response.
If valid, produce a realistic response based on the environment state.

Return JSON:
{
  "status": "success" | "error",
  "status_code": 200 | 400 | 422,
  "output": { ... },
  "environment_updates": { ... }
}
```

**Step 3 -- Apply environment updates**

If the tool has `mutates_environment: true` and returned `environment_updates`, apply them to the `EnvironmentManager`.

### Interface

```python
class ToolExecutor:
    def execute(
        self,
        tool_config: ToolConfig,
        turn_plan: dict,
        conversation: Conversation,
        environment: EnvironmentManager,
        sample: dict,
        deterministic_index: int,
    ) -> tuple[Message, Message]:
        """
        Returns (tool_call_message, tool_result_message).
        Side effect: may mutate environment via apply_updates.
        """
```

---

## Planner Extension

When `available_tools` is non-empty, two additions are injected into the planner prompt:

1. **Tool catalog** -- tool names, descriptions, and parameter summaries:
   ```
   Available tools:
   - ReturnRequestValidator: Validates a return request against order data
     Parameters: return_request_id (string), order_id (string), customer_id (string), return_reason (string)
   - SearchOrders: Look up order details by order ID
     Parameters: order_id (string)
   ```

2. **Environment overview** -- summary of available entities and context:
   ```
   Environment state contains:
   - customers: 1 customer (CUST002)
   - orders: 1 order (CUST002_ORD001, status: delivered)
   - returns: empty
   Context: 90-day return policy for footwear...
   ```

3. **Extended instruction** -- planner is told it may annotate ASSISTANT turns with tool calls.

### Extended planner JSON schema

```json
[
  {"turn": 1, "role": "USER", "instruction": "ask about returning their order"},
  {
    "turn": 2,
    "role": "ASSISTANT",
    "tool_call": "search_orders",
    "tool_args_instruction": "look up the customer's order using their customer ID from the environment",
    "instruction": "after seeing the order details, confirm the order and ask about the return reason"
  },
  {
    "turn": 3,
    "role": "USER",
    "instruction": "explain they changed their mind about the product"
  },
  {
    "turn": 4,
    "role": "ASSISTANT",
    "tool_call": "return_validator",
    "tool_args_instruction": "validate the return request with the order ID, customer ID, and reason 'changed mind'",
    "instruction": "confirm the return has been approved and explain next steps"
  }
]
```

- `tool_call` and `tool_args_instruction` are optional fields on any ASSISTANT turn.
- `instruction` is always present and guides the ASSISTANT's final response after the tool result.
- `_parse_plan` passes through the new fields transparently.
- Turn count is unaffected -- a tool-call turn still counts as one turn.

---

## ConversationSynthesizer Changes

### Environment initialization

Before planning, resolve `{placeholders}` in the environment template for each sample using `AttributeFormatter`, then create an `EnvironmentManager` per sample.

```python
# Per sample:
resolved_env = attribute_formatter.resolve(environment_config, sample)
env_manager = EnvironmentManager(
    entities=resolved_env["entities"],
    context=resolved_env["context"],
)
```

### Extended turn loop

```python
for turn_idx in range(target_turns):
    turn_plan = parsed_turn_plans[turn_idx]

    if turn_plan.get("tool_call"):
        tool_config = tools_by_id[turn_plan["tool_call"]]

        # Batch 1: Generate tool_call args for all active samples
        tool_call_msgs = batch_generate_args(
            tool_config, turn_plan, conversations, environments
        )

        # Batch 2: Simulate tool outputs for all active samples
        # Each sample's prompt includes its own environment snapshot
        tool_result_msgs = batch_simulate(
            tool_config, tool_call_msgs, environments
        )
        # Side effect: environment_updates applied per sample

        # Append tool_call and tool_result to each conversation
        for i, conv in enumerate(active_conversations):
            conv.messages.append(tool_call_msgs[i])
            conv.messages.append(tool_result_msgs[i])

        # Batch 3: Generate ASSISTANT response with tool result in context
        responses = batch_generate_response(
            conversations, turn_plan["instruction"], personas
        )
        for i, conv in enumerate(active_conversations):
            conv.messages.append(responses[i])
    else:
        # Existing behavior unchanged
        response = batch_generate_response(...)
        ...
```

### Batching strategy for tool-call turns

Tool-call turns run as up to three sequential batched inference passes:

1. **Args generation batch** -- generate tool_call messages for all samples at this turn position. Each prompt includes the sample's conversation history + environment snapshot.
2. **Simulation batch** -- for SIMULATED mode: generate tool outputs for all samples. Each prompt includes the sample's environment snapshot + tool call args. DETERMINISTIC mode skips this batch.
3. **Response batch** -- ASSISTANT responds with tool result in context (existing batch logic).

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

A synthesized conversation with tool-call turns:

```
[SYSTEM]    "You are a helpful customer support agent"
[USER]      "Hi, I'd like to return my hiking boots from my recent order"
[TOOL_CALL] {"name": "SearchOrders", "arguments": {"customer_id": "CUST002"}}
[TOOL]      {"status": "success", "status_code": 200, "output": {"order_id": "CUST002_ORD001", "items": [{"name": "Hiking Boots", "price": "89.99"}], "status": "delivered"}}
[ASSISTANT] "I can see your order with Hiking Boots for $89.99. It was delivered. I can help you with a return. What's the reason?"
[USER]      "I changed my mind about them"
[TOOL_CALL] {"name": "ReturnRequestValidator", "arguments": {"return_request_id": "RET001", "order_id": "CUST002_ORD001", "customer_id": "CUST002", "return_reason": "changed mind"}}
[TOOL]      {"status": "success", "status_code": 200, "output": {"is_valid": true, "validation_status": "approved", "days_remaining": 75}}
[ASSISTANT] "Your return has been approved! You have 75 days remaining in the return window..."
```

The output is a standard `Conversation` object. The `TOOL_CALL` and `TOOL` messages are inline between USER and ASSISTANT messages.

---

## Files Changed

| File | Change |
|------|--------|
| `src/oumi/core/types/conversation.py` | Add `Role.TOOL_CALL` to enum |
| `src/oumi/core/configs/params/synthesis_params.py` | Add `ToolConfig`, `ToolOutputMode`, `EnvironmentConfig`; extend `GeneralSynthesisParams` with `tools`; extend `MultiTurnAttribute` with `available_tools` and `environment` |
| `src/oumi/core/synthesis/environment_manager.py` | New file -- `EnvironmentManager` class |
| `src/oumi/core/synthesis/tool_executor.py` | New file -- `ToolExecutor` class |
| `src/oumi/core/synthesis/conversation_synthesizer.py` | Extend planner prompt injection with tool catalog + environment overview; extend turn loop to handle tool-call turns; per-sample environment initialization and tracking |
| `configs/examples/synthesis/agentic_tool_synth.yaml` | New example config demonstrating tool use with environment |
| `tests/unit/core/synthesis/test_environment_manager.py` | New tests |
| `tests/unit/core/synthesis/test_tool_executor.py` | New tests |
| `tests/unit/core/synthesis/test_conversation_synthesizer.py` | Add tool-call turn tests |
| `tests/unit/core/configs/params/test_synthesis_params.py` | Add ToolConfig, EnvironmentConfig validation tests |
