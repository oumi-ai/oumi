import pytest

from oumi_verl_ext.tool_episode import judge_transcript


@pytest.mark.parametrize("trailing_tool_messages", [0, 1, 2])
def test_transcript_ends_at_the_last_assistant_message(trailing_tool_messages):
    messages = [
        {"role": "user", "content": "Stock of pens?"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "3"},
        {"role": "assistant", "content": "Checking.<|im_end|>"},
        *[{"role": "tool", "content": "4"}] * trailing_tool_messages,
    ]

    transcript = judge_transcript(messages, None, ["<|im_end|>"])

    assert transcript == [*messages[:3], {"role": "assistant", "content": "Checking."}]
