"""Convert tool definitions in Hermes dataset to OpenAI JSON Schema format.

The dataset has two formats for tool definitions in metadata["tools"]:
1. OpenAI format: {"type": "function", "function": {"name": ..., "parameters": ...}}
2. Flat format: {"name": ..., "parameters": {"param_name": {"type": "str", ...}}}

This script normalizes all tools to OpenAI format and optionally applies a
chat template to render the tools into the system prompt.

Usage:
    # Convert tools in-place (adds metadata["tools_openai"]):
    python format_tools.py --input data/hermes_reasoning_tool_use_train_split_clean.jsonl \
                           --output data/hermes_reasoning_tool_use_train_split_formatted.jsonl

    # Also apply chat template:
    python format_tools.py --input data/hermes_reasoning_tool_use_train_split_clean.jsonl \
                           --output data/hermes_reasoning_tool_use_train_split_formatted.jsonl \
                           --apply_chat_template Qwen/Qwen2.5-1.5B-Instruct
"""

import argparse
import json
from pathlib import Path

# Map Python-style type strings to JSON Schema types
TYPE_MAP = {
    "str": "string",
    "string": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
    "list": "array",
    "List": "array",
    "set": "array",
    "dict": "object",
    "Dict": "object",
}


def normalize_type(type_str: str) -> str:
    """Map Python-style types to JSON Schema types."""
    base = type_str.replace(", optional", "").strip()
    # Handle parameterized types like List[int], Tuple[float, float]
    bracket = base.find("[")
    if bracket != -1:
        base = base[:bracket]
    return TYPE_MAP.get(base, "string")


def convert_tool(tool: dict) -> dict:
    """Convert a single tool definition to OpenAI function calling format."""
    # Already OpenAI format
    if tool.get("type") == "function" and "function" in tool:
        return tool

    name = tool.get("name", "")
    desc = tool.get("description", "")
    params = tool.get("parameters", {})

    if not isinstance(params, dict):
        return {
            "type": "function",
            "function": {"name": name, "description": desc, "parameters": {}},
        }

    # Semi-structured: has "properties" key
    if "properties" in params:
        properties = {}
        for k, v in params.get("properties", {}).items():
            if not isinstance(v, dict):
                continue
            prop = dict(v)
            if "type" in prop and isinstance(prop["type"], str):
                prop["type"] = normalize_type(prop["type"])
            properties[k] = prop

        json_params = {"type": "object", "properties": properties}
        if "required" in params:
            json_params["required"] = params["required"]
    else:
        # Flat key-value: each key is a param name
        properties = {}
        required = []
        for k, v in params.items():
            if not isinstance(v, dict):
                continue
            raw_type = v.get("type", "string")
            if isinstance(raw_type, str):
                optional = "optional" in raw_type.lower()
                prop = {
                    "type": normalize_type(raw_type),
                    "description": v.get("description", ""),
                }
                if v.get("default") is not None:
                    prop["default"] = v["default"]
                if not optional:
                    required.append(k)
            else:
                prop = {"type": "string", "description": v.get("description", "")}
                required.append(k)
            properties[k] = prop

        json_params = {"type": "object", "properties": properties}
        if required:
            json_params["required"] = required

    return {
        "type": "function",
        "function": {"name": name, "description": desc, "parameters": json_params},
    }


def convert_tools(tools_json_str: str) -> list[dict]:
    """Parse and convert all tools from a metadata["tools"] JSON string."""
    tools = json.loads(tools_json_str)
    return [convert_tool(t) for t in tools]


def main():
    parser = argparse.ArgumentParser(description="Convert tool definitions to OpenAI format")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    parser.add_argument(
        "--apply_chat_template",
        type=str,
        default=None,
        help="Tokenizer name/path. If set, applies chat template with tools "
        "and replaces messages with the rendered conversation.",
    )
    args = parser.parse_args()

    tokenizer = None
    if args.apply_chat_template:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.apply_chat_template, trust_remote_code=True
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    kept, errors = 0, 0
    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            record = json.loads(line)
            tools_raw = record["metadata"].get("tools", "[]")

            try:
                openai_tools = convert_tools(tools_raw)
            except (json.JSONDecodeError, KeyError):
                errors += 1
                continue

            record["metadata"]["tools_openai"] = json.dumps(openai_tools)

            if tokenizer:
                # Build messages for chat template
                messages = [
                    {"role": m["role"], "content": m["content"]}
                    for m in record["messages"]
                ]
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tools=openai_tools,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                record["rendered"] = rendered

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Converted {kept} records ({errors} errors) -> {args.output}")


if __name__ == "__main__":
    main()
