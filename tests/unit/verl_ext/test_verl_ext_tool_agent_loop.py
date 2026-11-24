import pytest

from oumi_verl_ext.tool_episode import (
    call_problems,
    judge_transcript,
    stopped_by_turn_limit,
)


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


@pytest.mark.parametrize(
    ("response_tokens", "max_assistant_turns", "expected"),
    [
        pytest.param(100, 2, False, id="response budget, checked first"),
        pytest.param(99, 2, True, id="assistant turn limit"),
        pytest.param(99, None, False, id="user turn limit"),
    ],
)
def test_stopped_by_turn_limit_follows_verls_check_order(
    response_tokens, max_assistant_turns, expected
):
    assert (
        stopped_by_turn_limit(
            response_tokens,
            response_length=100,
            assistant_turns=2,
            max_assistant_turns=max_assistant_turns,
        )
        is expected
    )
