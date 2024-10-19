from pathlib import Path
from typing import Final

import pytest

from oumi.core.types.conversation import Conversation, Message, Role, Type
from oumi.utils.image_utils import load_image_png_bytes_from_path

OUMI_ROOT_DIR: Final[Path] = Path(__file__).resolve().parent.parent.parent.parent
TEST_IMAGE_DIR: Final[Path] = OUMI_ROOT_DIR / "tests" / "testdata" / "images"


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


def test_conversation_to_dict():
    conv = Conversation(
        messages=[
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="Hi there!"),
        ],
        metadata={"test": "metadata"},
    )
    conv_dict = conv.to_dict()

    assert isinstance(conv_dict, dict)
    assert "messages" in conv_dict
    assert len(conv_dict["messages"]) == 2
    assert conv_dict["metadata"] == {"test": "metadata"}
    assert conv_dict["messages"][0]["role"] == "user"
    assert conv_dict["messages"][0]["content"] == "Hello"
    assert conv_dict["messages"][1]["role"] == "assistant"
    assert conv_dict["messages"][1]["content"] == "Hi there!"


def test_conversation_from_dict():
    conv_dict = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ],
        "metadata": {"test": "metadata"},
    }
    conv = Conversation.from_dict(conv_dict)

    assert isinstance(conv, Conversation)
    assert len(conv.messages) == 2
    assert conv.metadata == {"test": "metadata"}
    assert conv.messages[0].role == Role.USER
    assert conv.messages[0].content == "Hello"
    assert conv.messages[1].role == Role.ASSISTANT
    assert conv.messages[1].content == "Hi there!"


def test_conversation_to_json():
    conv = Conversation(
        messages=[
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="Hi there!"),
        ],
        metadata={"test": "metadata"},
    )
    json_str = conv.to_json()

    assert isinstance(json_str, str)
    assert '"role":"user"' in json_str
    assert '"content":"Hello"' in json_str
    assert '"role":"assistant"' in json_str
    assert '"content":"Hi there!"' in json_str
    assert '"test":"metadata"' in json_str


def test_conversation_from_json():
    json_str = '{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}], "metadata": {"test": "metadata"}}'  # noqa: E501
    conv = Conversation.from_json(json_str)

    assert isinstance(conv, Conversation)
    assert len(conv.messages) == 2
    assert conv.metadata == {"test": "metadata"}
    assert conv.messages[0].role == Role.USER
    assert conv.messages[0].content == "Hello"
    assert conv.messages[1].role == Role.ASSISTANT
    assert conv.messages[1].content == "Hi there!"


def test_roundtrip_dict():
    png_image_bytes = load_image_png_bytes_from_path(TEST_IMAGE_DIR / "math.png")

    original = Conversation(
        messages=[
            Message(id="001", role=Role.SYSTEM, content="Behave!"),
            Message(id="", role=Role.ASSISTANT, content="Hi there!", type=Type.TEXT),
            Message(role=Role.USER, binary=png_image_bytes, type=Type.IMAGE_BINARY),
            Message(role=Role.USER, binary=b"", type=Type.IMAGE_BINARY),
            Message(
                role=Role.ASSISTANT,
                content="https://www.oumi.ai/logo.png",
                type=Type.IMAGE_URL,
            ),
            Message(
                id="xyz",
                role=Role.TOOL,
                content=str(TEST_IMAGE_DIR / "math.png"),
                type=Type.IMAGE_PATH,
            ),
        ],
        metadata={"test": "metadata"},
    )
    conv_dict = original.to_dict()
    reconstructed = Conversation.from_dict(conv_dict)

    assert original == reconstructed


def test_roundtrip_json():
    png_image_bytes = load_image_png_bytes_from_path(TEST_IMAGE_DIR / "math.png")

    original = Conversation(
        messages=[
            Message(id="001", role=Role.SYSTEM, content="Behave!"),
            Message(id="", role=Role.ASSISTANT, content="Hi there!", type=Type.TEXT),
            Message(role=Role.USER, binary=png_image_bytes, type=Type.IMAGE_BINARY),
            Message(role=Role.USER, binary=b"", type=Type.IMAGE_BINARY),
            Message(
                role=Role.ASSISTANT,
                content="https://www.oumi.ai/logo.png",
                type=Type.IMAGE_URL,
            ),
            Message(
                id="xyz",
                role=Role.TOOL,
                content=str(TEST_IMAGE_DIR / "math.png"),
                type=Type.IMAGE_PATH,
            ),
        ],
        metadata={"test": "metadata"},
    )
    json_str = original.to_json()
    reconstructed = Conversation.from_json(json_str)

    assert original == reconstructed


def test_from_dict_with_invalid_field():
    with pytest.raises(ValueError, match="Field required"):
        Conversation.from_dict({"invalid": "data"})


def test_from_json_with_invalid_field():
    with pytest.raises(ValueError, match="Invalid JSON"):
        Conversation.from_json('{"invalid": json')


def test_from_dict_with_invalid_base64():
    with pytest.raises(ValueError, match="Invalid base64-encoded string"):
        Conversation.from_dict(
            {
                "messages": [
                    {
                        "binary": "INVALID_BASE64!",
                        "role": "user",
                        "type": "image_binary",
                    },
                ],
                "metadata": {"test": "metadata"},
            }
        )
