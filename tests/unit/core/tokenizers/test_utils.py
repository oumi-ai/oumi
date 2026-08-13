from typing import Any

import jinja2
import jinja2.sandbox
import pytest

from oumi.core.tokenizers import utils as tokenizer_utils
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.core.types.tool_call import ToolCall

# Real templates disagree on the shape of tool-call `arguments`: Qwen2.5 pipes it
# through `tojson`, Llama 3.1 iterates it, DeepSeek-V3 concatenates it onto a string.
_TOJSON_TEMPLATE = "{{ messages[0].tool_calls[0].function.arguments | tojson }}"
_ITEMS_TEMPLATE = (
    "{% for k, v in messages[0].tool_calls[0].function.arguments.items() %}"
    "{{ k }}={{ v }}{% endfor %}"
)
_CONCAT_TEMPLATE = "{{ '<>' + messages[0].tool_calls[0].function.arguments }}"
_TOOLS_TEMPLATE = "{{ tools[0]['function']['name'] }}"


class _FakeTokenizer:
    """Renders a chat template the way `transformers` does: sandboxed Jinja."""

    def __init__(self, template: str):
        env = jinja2.sandbox.ImmutableSandboxedEnvironment()
        self._template = env.from_string(template)

    def apply_chat_template(self, messages, tools=None, **kwargs) -> str:
        return self._template.render(messages=messages, tools=tools)


def _conversation(
    arguments: Any = '{"city": "SF"}', tools: list | None = None
) -> Conversation:
    tool_call = ToolCall.model_validate(
        {
            "id": "call_abc",
            "type": "function",
            "function": {"name": "get_weather", "arguments": arguments},
        }
    )
    return Conversation(
        messages=[Message(role=Role.ASSISTANT, content=None, tool_calls=[tool_call])],
        tools=tools,
    )


def _render(template: str, conversation: Conversation) -> str:
    return tokenizer_utils.apply_chat_template(
        _FakeTokenizer(template),  # type: ignore[arg-type]
        conversation,
    )


@pytest.mark.parametrize("arguments", [{"city": "SF"}, '{"city": "SF"}'])
@pytest.mark.parametrize(
    "template,expected",
    [(_TOJSON_TEMPLATE, '{"city": "SF"}'), (_ITEMS_TEMPLATE, "city=SF")],
)
def test_object_shaped_templates_get_an_object(template, expected, arguments):
    assert _render(template, _conversation(arguments)) == expected


def test_string_shaped_template_falls_back_to_the_wire_form():
    """A dict raises `TypeError` on concat-style templates; the string form works."""
    assert (
        _render(_CONCAT_TEMPLATE, _conversation({"city": "SF"})) == '<>{"city": "SF"}'
    )


def test_malformed_arguments_survive_both_attempts():
    """Providers return invalid JSON; it must render rather than raise."""
    assert _render(_CONCAT_TEMPLATE, _conversation("not json{")) == "<>not json{"


def test_tools_are_forwarded_as_a_separate_kwarg():
    """Without `tools` the model sees no schema and invents function names."""
    tool = {
        "type": "function",
        "function": {"name": "get_weather", "parameters": {"type": "object"}},
    }
    assert _render(_TOOLS_TEMPLATE, _conversation(tools=[tool])) == "get_weather"
