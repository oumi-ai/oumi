from oumi_verl_ext.tool_episode import full_tool_schemas


def test_full_tool_schemas_replace_verls_by_name():
    full = {
        "type": "function",
        "function": {
            "name": "get_stock",
            "description": "Counts an item's stock.",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    }
                },
            },
        },
    }
    verl_schema = {
        "type": "function",
        "function": {
            "name": "get_stock",
            "description": "Counts an item's stock.",
            "parameters": {
                "type": "object",
                "properties": {"items": {"type": "array"}},
            },
        },
    }
    function_tool = {
        "type": "function",
        "function": {"name": "add", "description": "Adds two numbers."},
    }

    assert full_tool_schemas([full], [verl_schema, function_tool]) == [
        full,
        function_tool,
    ]
