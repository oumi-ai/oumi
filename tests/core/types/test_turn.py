import pytest

from oumi.core.types.turn import Conversation, Message, Role


@pytest.fixture
def test_conversation():
    role_user = Role.USER
    role_assistant = Role.ASSISTANT

    message1 = Message(role=role_user, content="Hello", id="1")
    message2 = Message(role=role_assistant, content="Hi, how can I help you?")
    message3 = Message(role=role_user, content="I need assistance with my account.")

    conversation = Conversation(messages=[message1, message2, message3])
    return conversation, role_user, role_assistant, message1, message2, message3


def test_first_message_no_role(test_conversation):
    conversation, _, _, message1, _, _ = test_conversation
    assert conversation.first_message() == message1


def test_first_message_with_role(test_conversation):
    conversation, _, role_assistant, _, message2, _ = test_conversation
    assert conversation.first_message(role_assistant) == message2


def test_first_message_with_nonexistent_role(test_conversation):
    conversation, _, _, _, _, _ = test_conversation
    role_nonexistent = Role.TOOL
    assert conversation.first_message(role_nonexistent) is None


def test_last_message_no_role(test_conversation):
    conversation, _, _, _, _, message3 = test_conversation
    assert conversation.last_message() == message3


def test_last_message_with_role(test_conversation):
    conversation, role_user, _, _, _, message3 = test_conversation
    assert conversation.last_message(role_user) == message3


def test_last_message_with_nonexistent_role(test_conversation):
    conversation, _, _, _, _, _ = test_conversation
    role_nonexistent = Role.TOOL
    assert conversation.last_message(role_nonexistent) is None


def test_filter_messages_no_role(test_conversation):
    conversation, _, _, message1, message2, message3 = test_conversation
    assert conversation.filter_messages() == [message1, message2, message3]


def test_filter_messages_with_role(test_conversation):
    conversation, role_user, _, message1, _, message3 = test_conversation
    assert conversation.filter_messages(role_user) == [message1, message3]


def test_filter_messages_with_nonexistent_role(test_conversation):
    conversation, _, _, _, _, _ = test_conversation
    role_nonexistent = Role.TOOL
    assert conversation.filter_messages(role_nonexistent) == []


def test_repr(test_conversation):
    conversation, _, _, message1, message2, message3 = test_conversation
    assert repr(message1) == "1 - USER: Hello"
    assert repr(message2) == "ASSISTANT: Hi, how can I help you?"
    assert repr(message3) == "USER: I need assistance with my account."
    assert repr(conversation) == (
        "1 - USER: Hello\n"
        "ASSISTANT: Hi, how can I help you?\n"
        "USER: I need assistance with my account."
    )
