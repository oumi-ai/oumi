def test_template_field_access_on_tool_call_and_function_call():
    """Gemma 4's template reads these as mappings, so a model with neither `get` nor
    `__getitem__` fails to render at all. `arguments` decodes, since the template
    requires a mapping there and raises on the wire string this class stores."""
    from oumi.core.types.tool_call import FunctionCall, ToolCall

    call = ToolCall(
        id="c1", function=FunctionCall(name="search", arguments='{"query": "mixer"}')
    )
    assert call.get("function").name == "search"
    assert call["id"] == "c1"
    assert call.get("nonexistent") is None

    fn = call["function"]
    assert fn["arguments"] == {"query": "mixer"}  # the template needs the mapping
    assert fn.arguments == '{"query": "mixer"}'  # storage stays the wire string
    assert fn["name"] == "search"
    assert fn.get("missing", "fallback") == "fallback"

    # A dict on the way in still normalizes to the wire string, so both forms agree.
    from_dict = FunctionCall(name="search", arguments={"query": "mixer"})  # type: ignore[arg-type]
    assert from_dict.arguments == '{"query": "mixer"}'
    assert from_dict["arguments"] == {"query": "mixer"}
