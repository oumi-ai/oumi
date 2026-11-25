import pytest

from oumi_verl_ext.tool_episode import call_problems, judge_transcript


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


@pytest.mark.parametrize(
    ("call_start", "parsed_calls", "expected"),
    [
        pytest.param("<tool_call>", 0, (1, 0), id="unreadable call"),
        pytest.param("<tool_call>", 2, (0, 1), id="two calls from one marker"),
        pytest.param(None, 1, (0, 0), id="parser without a marker"),
    ],
)
def test_call_problems_counts_malformed_and_dropped_calls(
    call_start, parsed_calls, expected
):
    text = '<tool_call>{"name": "get_stock"</tool_call>'

    assert call_problems(text, call_start, parsed_calls, max_parallel_calls=1) == (
        expected
    )
