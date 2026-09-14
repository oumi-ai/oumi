from oumi.core.types.conversation import Role
from oumi.utils.canonical_tool_conversations import (
    ARGUMENT_SENTINEL,
    ASSISTANT_TEXT_1,
    ASSISTANT_TEXT_2,
    FIRST_TOOL_CALL_INDEX,
    SYSTEM_TEXT,
    TOOL_RESULT_1,
    TOOL_RESULT_2,
    TOOL_RESULT_3,
    USER_TEXT_1,
    USER_TEXT_2,
    canonical_tool_conversation,
)

# The conversation is shared by the chat-template probe and by the template and
# collator tests, each of which relies on a documented property of its shape.
# These pin those properties, so a change to the conversation shows up here
# rather than as a puzzling failure in one of the callers.


def test_first_tool_call_index_points_at_parallel_calls_with_no_content():
    """What the probe and the collator tests both index into."""
    message = canonical_tool_conversation().messages[FIRST_TOOL_CALL_INDEX]

    assert message.role == Role.ASSISTANT
    assert message.content is None
    assert message.tool_calls is not None
    assert len(message.tool_calls) == 2


def test_tool_calls_omit_an_explicit_type():
    """Left at its default on purpose; `to_dict()` then drops the key.

    That is what a conversation built in code rather than parsed from
    OpenAI-format JSON looks like, and some templates read `type` directly.
    """
    conversation = canonical_tool_conversation()
    raw = conversation.to_dict()

    assert "type" not in raw["messages"][FIRST_TOOL_CALL_INDEX]["tool_calls"][0]
    assert "type" not in raw["tools"][0]


def test_the_first_argument_carries_the_sentinel():
    """The probe looks for this value, and for quotes escaped around it."""
    message = canonical_tool_conversation().messages[FIRST_TOOL_CALL_INDEX]
    assert message.tool_calls is not None

    arguments = message.tool_calls[0].function.arguments

    assert ARGUMENT_SENTINEL in arguments


def test_every_text_value_is_distinct():
    """Tests locate these in rendered output, so none may be a substring."""
    texts = [
        SYSTEM_TEXT,
        USER_TEXT_1,
        USER_TEXT_2,
        ASSISTANT_TEXT_1,
        ASSISTANT_TEXT_2,
        TOOL_RESULT_1,
        TOOL_RESULT_2,
        TOOL_RESULT_3,
        ARGUMENT_SENTINEL,
    ]

    assert len(set(texts)) == len(texts)
    for text in texts:
        others = [t for t in texts if t != text]
        assert not any(text in other for other in others), f"{text} is a substring"


def test_tool_results_are_matched_to_the_calls_that_requested_them():
    """Interleaved results, so a template that reorders them is detectable."""
    conversation = canonical_tool_conversation()

    call_ids = [
        call.id
        for message in conversation.messages
        for call in (message.tool_calls or [])
    ]
    result_ids = [
        message.tool_call_id
        for message in conversation.messages
        if message.role == Role.TOOL
    ]

    assert call_ids == result_ids
    assert len(call_ids) == 3
