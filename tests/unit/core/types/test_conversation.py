import base64
from typing import Final

import pytest

from oumi.core.types.conversation import (
    Conversation,
    Message,
    MessageContentItem,
    MessageContentItemCounts,
    Role,
    Type,
)
from oumi.utils.image_utils import load_image_png_bytes_from_path

_SMALL_B64_IMAGE: Final[str] = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _create_test_image_bytes() -> bytes:
    return base64.b64decode(_SMALL_B64_IMAGE)


@pytest.fixture
def test_conversation():
    role_user = Role.USER
    role_assistant = Role.ASSISTANT

    message1 = Message(role=role_user, content="Hello", id="1")
    message2 = Message(role=role_assistant, content="Hi, how can I help you?")
    message3 = Message(
        type=Type.COMPOUND,
        role=role_user,
        content=[
            MessageContentItem(
                type=Type.TEXT, content="I need assistance with my account."
            ),
            MessageContentItem(
                type=Type.IMAGE_BINARY, binary=_create_test_image_bytes()
            ),
        ],
    )

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
    assert repr(message3) == "USER: I need assistance with my account. | <IMAGE_BINARY>"
    assert repr(conversation) == (
        "1 - USER: Hello\n"
        "ASSISTANT: Hi, how can I help you?\n"
        "USER: I need assistance with my account. | <IMAGE_BINARY>"
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


def test_roundtrip_dict(root_testdata_dir):
    png_image_bytes = load_image_png_bytes_from_path(
        root_testdata_dir / "images" / "oumi_logo_dark.png"
    )

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
                content=str(root_testdata_dir / "images" / "oumi_logo_dark.png"),
                type=Type.IMAGE_PATH,
            ),
        ],
        metadata={"test": "metadata"},
    )
    conv_dict = original.to_dict()
    reconstructed = Conversation.from_dict(conv_dict)

    assert original == reconstructed


def test_roundtrip_json(root_testdata_dir):
    png_image_bytes = load_image_png_bytes_from_path(
        root_testdata_dir / "images" / "oumi_logo_light.png"
    )

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
                content=str(root_testdata_dir / "images" / "oumi_logo_dark.png"),
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


def test_compound_content_incorrect_message_type():
    with pytest.raises(RuntimeError, match="Unexpected content type"):
        Message(
            role=Role.ASSISTANT,
            content=[
                MessageContentItem(
                    type=Type.TEXT, content="I need assistance with my account."
                )
            ],
        )
    with pytest.raises(RuntimeError, match="Unexpected content type"):
        Message(
            type=Type.TEXT,
            role=Role.ASSISTANT,
            content=[
                MessageContentItem(
                    type=Type.TEXT, content="I need assistance with my account."
                )
            ],
        )
    with pytest.raises(RuntimeError, match="Unexpected content type"):
        Message(
            type=Type.IMAGE_PATH,
            role=Role.ASSISTANT,
            content=[],
        )

    with pytest.raises(RuntimeError, match="Unexpected content type"):
        Message(
            type=Type.COMPOUND,
            role=Role.USER,
            content="Hello!",
        )


@pytest.mark.parametrize(
    "role",
    [Role.USER, Role.ASSISTANT, Role.TOOL, Role.SYSTEM],
)
def test_content_item_methods_mixed_items(role: Role):
    text_item1 = MessageContentItem(type=Type.TEXT, content="aaa")
    image_item1 = MessageContentItem(
        type=Type.IMAGE_BINARY, binary=_create_test_image_bytes()
    )
    text_item2 = MessageContentItem(type=Type.TEXT, content=" B B ")
    image_item2 = MessageContentItem(
        type=Type.IMAGE_PATH,
        content="/tmp/test/dummy.jpeg",
        binary=_create_test_image_bytes(),
    )
    text_item3 = MessageContentItem(type=Type.TEXT, content="CC")

    message = Message(
        type=Type.COMPOUND,
        role=role,
        content=[
            text_item1,
            image_item1,
            text_item2,
            image_item2,
            text_item3,
        ],
    )

    assert message.contains_text()
    assert not message.contains_single_text_content_item_only()
    assert not message.contains_text_content_items_only()

    assert message.contains_images()
    assert not message.contains_single_image_content_item_only()
    assert not message.contains_image_content_items_only()

    assert message.compute_flattened_text_content() == "aaa  B B  CC"
    assert message.compute_flattened_text_content("||") == "aaa|| B B ||CC"

    assert message.content_items == [
        text_item1,
        image_item1,
        text_item2,
        image_item2,
        text_item3,
    ]
    assert message.image_content_items == [image_item1, image_item2]
    assert message.text_content_items == [text_item1, text_item2, text_item3]

    assert message.count_content_items() == MessageContentItemCounts(
        total_items=5, image_items=2, text_items=3
    )


@pytest.mark.parametrize(
    "image_type",
    [Type.IMAGE_BINARY, Type.IMAGE_PATH, Type.IMAGE_URL],
)
def test_content_item_methods_legacy_image(image_type):
    test_image_item = MessageContentItem(
        type=image_type,
        content=(None if image_type == Type.IMAGE_BINARY else "foo"),
        binary=(
            _create_test_image_bytes() if image_type == Type.IMAGE_BINARY else None
        ),
    )
    message = Message(
        type=image_type,
        role=Role.ASSISTANT,
        content=test_image_item.content,
        binary=test_image_item.binary,
    )

    assert not message.contains_text()
    assert not message.contains_single_text_content_item_only()
    assert not message.contains_text_content_items_only()

    assert message.contains_images()
    assert message.contains_single_image_content_item_only()
    assert message.contains_image_content_items_only()

    assert message.compute_flattened_text_content() == ""
    assert message.compute_flattened_text_content("Z") == ""

    assert message.content_items == [
        test_image_item,
    ]
    assert message.image_content_items == [test_image_item]
    assert message.text_content_items == []

    assert message.count_content_items() == MessageContentItemCounts(
        total_items=1, image_items=1, text_items=0
    )


@pytest.mark.parametrize(
    "image_type",
    [Type.IMAGE_BINARY, Type.IMAGE_PATH, Type.IMAGE_URL],
)
def test_content_item_methods_single_image(image_type):
    test_image_item = MessageContentItem(
        type=image_type,
        content=(None if image_type == Type.IMAGE_BINARY else "foo"),
        binary=(
            _create_test_image_bytes() if image_type == Type.IMAGE_BINARY else None
        ),
    )
    message = Message(
        type=Type.COMPOUND,
        role=Role.ASSISTANT,
        content=[test_image_item],
    )

    assert not message.contains_text()
    assert not message.contains_single_text_content_item_only()
    assert not message.contains_text_content_items_only()

    assert message.contains_images()
    assert message.contains_single_image_content_item_only()
    assert message.contains_image_content_items_only()

    assert message.compute_flattened_text_content() == ""
    assert message.compute_flattened_text_content("Z") == ""

    assert message.content_items == [
        test_image_item,
    ]
    assert message.image_content_items == [test_image_item]
    assert message.text_content_items == []

    assert message.count_content_items() == MessageContentItemCounts(
        total_items=1, image_items=1, text_items=0
    )


def test_content_item_methods_triple_image():
    test_image_item1 = MessageContentItem(
        type=Type.IMAGE_BINARY,
        binary=(_create_test_image_bytes()),
    )
    test_image_item2 = MessageContentItem(
        type=Type.IMAGE_URL,
        content="http://oumi.ai/a.png",
    )
    test_image_item3 = MessageContentItem(
        type=Type.IMAGE_PATH,
        content="/tmp/oumi.ai/b.gif",
    )
    message = Message(
        type=Type.COMPOUND,
        role=Role.ASSISTANT,
        content=[test_image_item1, test_image_item2, test_image_item3],
    )

    assert not message.contains_text()
    assert not message.contains_single_text_content_item_only()
    assert not message.contains_text_content_items_only()

    assert message.contains_images()
    assert not message.contains_single_image_content_item_only()
    assert message.contains_image_content_items_only()

    assert message.compute_flattened_text_content() == ""
    assert message.compute_flattened_text_content("Z") == ""

    assert message.content_items == [
        test_image_item1,
        test_image_item2,
        test_image_item3,
    ]
    assert message.image_content_items == [
        test_image_item1,
        test_image_item2,
        test_image_item3,
    ]
    assert message.text_content_items == []

    assert message.count_content_items() == MessageContentItemCounts(
        total_items=3, image_items=3, text_items=0
    )


def test_content_item_methods_legacy_text():
    test_text_item = MessageContentItem(type=Type.TEXT, content="bzzz")
    message = Message(
        role=Role.USER,
        type=Type.TEXT,
        content=test_text_item.content,
        binary=test_text_item.binary,
    )

    assert message.contains_text()
    assert message.contains_single_text_content_item_only()
    assert message.contains_text_content_items_only()

    assert not message.contains_images()
    assert not message.contains_single_image_content_item_only()
    assert not message.contains_image_content_items_only()

    assert message.compute_flattened_text_content() == "bzzz"
    assert message.compute_flattened_text_content("X") == "bzzz"

    assert message.content_items == [
        test_text_item,
    ]
    assert message.image_content_items == []
    assert message.text_content_items == [test_text_item]

    assert message.count_content_items() == MessageContentItemCounts(
        total_items=1, image_items=0, text_items=1
    )


def test_content_item_methods_double_text():
    test_text_item = MessageContentItem(type=Type.TEXT, content="bzzz")
    message = Message(
        role=Role.USER,
        type=Type.COMPOUND,
        content=[test_text_item, test_text_item],
    )

    assert message.contains_text()
    assert not message.contains_single_text_content_item_only()
    assert message.contains_text_content_items_only()

    assert not message.contains_images()
    assert not message.contains_single_image_content_item_only()
    assert not message.contains_image_content_items_only()

    assert message.compute_flattened_text_content() == "bzzz bzzz"
    assert message.compute_flattened_text_content("^") == "bzzz^bzzz"

    assert message.content_items == [
        test_text_item,
        test_text_item,
    ]
    assert message.image_content_items == []
    assert message.text_content_items == [test_text_item, test_text_item]

    assert message.count_content_items() == MessageContentItemCounts(
        total_items=2, image_items=0, text_items=2
    )
