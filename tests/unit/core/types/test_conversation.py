import base64
from typing import Final, cast

import pydantic
import pytest

from oumi.core.types.conversation import (
    ContentItem,
    ContentItemCounts,
    Conversation,
    Message,
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
        role=role_user,
        content=[
            ContentItem(type=Type.TEXT, content="I need assistance with my account."),
            ContentItem(type=Type.IMAGE_BINARY, binary=_create_test_image_bytes()),
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
    assert conversation.filter_messages(role=role_user) == [message1, message3]
    assert conversation.filter_messages(filter_fn=lambda m: m.role == role_user) == [
        message1,
        message3,
    ]


def test_filter_messages_with_filter_fn(test_conversation):
    conversation, role_user, _, message1, message2, message3 = test_conversation
    assert conversation.filter_messages(filter_fn=lambda m: m.role == role_user) == [
        message1,
        message3,
    ]
    assert conversation.filter_messages(filter_fn=lambda m: m.role != role_user) == [
        message2,
    ]


def test_filter_messages_with_role_and_filter_fn(test_conversation):
    conversation, role_user, _, message1, message2, message3 = test_conversation
    assert conversation.filter_messages(
        role=role_user, filter_fn=lambda m: m.role == role_user
    ) == [
        message1,
        message3,
    ]
    assert (
        conversation.filter_messages(
            role=role_user, filter_fn=lambda m: m.role != role_user
        )
        == []
    )


def test_filter_messages_with_nonexistent_role(test_conversation):
    conversation, _, _, _, _, _ = test_conversation
    role_nonexistent = Role.TOOL
    assert conversation.filter_messages(role=role_nonexistent) == []
    assert (
        conversation.filter_messages(role=role_nonexistent, filter_fn=lambda m: True)
        == []
    )
    assert (
        conversation.filter_messages(filter_fn=lambda m: m.role == role_nonexistent)
        == []
    )


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


def test_conversation_to_dict_legacy():
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


def test_conversation_to_dict_compound_text_content():
    conv = Conversation(
        messages=[
            Message(
                role=Role.USER,
                content=[ContentItem(type=Type.TEXT, content="Hello")],
            ),
            Message(
                role=Role.ASSISTANT,
                content=[ContentItem(type=Type.TEXT, content="Hi there!")],
            ),
        ],
        metadata={"test": "metadata"},
    )
    # Default format is now OUMI (HuggingFace-compatible)
    conv_dict = conv.to_dict()

    assert isinstance(conv_dict, dict)
    assert "messages" in conv_dict
    assert len(conv_dict["messages"]) == 2
    assert conv_dict["metadata"] == {"test": "metadata"}
    assert conv_dict["messages"][0]["role"] == "user"
    assert conv_dict["messages"][0]["content"] == [{"text": "Hello", "type": "text"}]
    assert conv_dict["messages"][1]["role"] == "assistant"
    assert conv_dict["messages"][1]["content"] == [
        {"text": "Hi there!", "type": "text"}
    ]


def test_conversation_to_dict_compound_mixed_content():
    png_bytes = _create_test_image_bytes()
    conv = Conversation(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    ContentItem(type=Type.IMAGE_BINARY, binary=png_bytes),
                    ContentItem(type=Type.TEXT, content="Hello"),
                ],
            ),
            Message(
                role=Role.ASSISTANT,
                content=[
                    ContentItem(type=Type.TEXT, content="Hi there!"),
                    ContentItem(
                        type=Type.IMAGE_URL,
                        content="/tmp/foo.png",
                        binary=png_bytes,
                    ),
                ],
            ),
        ],
        metadata={"test": "metadata"},
    )
    # Default format is now OUMI (HuggingFace-compatible)
    conv_dict = conv.to_dict()

    assert isinstance(conv_dict, dict)
    assert "messages" in conv_dict
    assert len(conv_dict["messages"]) == 2
    assert conv_dict["metadata"] == {"test": "metadata"}
    assert conv_dict["messages"][0]["role"] == "user"
    # IMAGE_BINARY -> {"type": "image", "url": "data:..."}
    # TEXT -> {"type": "text", "text": "..."}
    assert conv_dict["messages"][0]["content"] == [
        {
            "type": "image",
            "url": f"data:image/png;base64,{_SMALL_B64_IMAGE}",
        },
        {"type": "text", "text": "Hello"},
    ]
    assert conv_dict["messages"][1]["role"] == "assistant"
    # IMAGE_URL -> {"type": "image", "url": "...", "binary": "..."}
    assert conv_dict["messages"][1]["content"] == [
        {"type": "text", "text": "Hi there!"},
        {
            "type": "image",
            "url": "/tmp/foo.png",
            "binary": _SMALL_B64_IMAGE,
        },
    ]


def test_conversation_from_dict_legacy():
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


def test_conversation_from_dict_with_unknown_fields():
    conv_dict = {
        "messages": [
            {"role": "user", "content": "Hello", "foo_unknown": "bar"},
            {"role": "assistant", "content": "Hi there!", "type": "text"},
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


def test_conversation_from_dict_compound_mixed_content():
    png_bytes = _create_test_image_bytes()
    conv_dict = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "binary": _SMALL_B64_IMAGE,
                        "type": "image_binary",
                    },
                    {"content": "Hello", "type": "text"},
                ],
            },
            {"role": "assistant", "content": "Hi there!"},
        ],
        "metadata": {"test": "metadata"},
    }
    conv = Conversation.from_dict(conv_dict)

    assert isinstance(conv, Conversation)
    assert len(conv.messages) == 2
    assert conv.metadata == {"test": "metadata"}
    assert conv.messages[0].role == Role.USER
    assert isinstance(conv.messages[0].content, list)
    assert conv.messages[0].content == [
        ContentItem(type=Type.IMAGE_BINARY, binary=png_bytes),
        ContentItem(type=Type.TEXT, content="Hello"),
    ]
    assert conv.messages[1].role == Role.ASSISTANT
    assert conv.messages[1].content == "Hi there!"


def test_conversation_to_json_legacy():
    conv = Conversation(
        messages=[
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="Hi there!"),
        ],
        metadata={"test": "metadata"},
    )
    # String content is preserved as-is in all formats
    json_str = conv.to_json()

    assert isinstance(json_str, str)
    assert '"role": "user"' in json_str
    assert '"content": "Hello"' in json_str
    assert '"role": "assistant"' in json_str
    assert '"content": "Hi there!"' in json_str
    assert '"test": "metadata"' in json_str


def test_conversation_to_json_mixed_content():
    png_bytes = _create_test_image_bytes()
    conv = Conversation(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    ContentItem(type=Type.IMAGE_BINARY, binary=png_bytes),
                    ContentItem(type=Type.TEXT, content="Hello"),
                ],
            ),
            Message(
                role=Role.ASSISTANT,
                content=[
                    ContentItem(type=Type.TEXT, content="Hi there!"),
                    ContentItem(
                        type=Type.IMAGE_URL,
                        content="/tmp/foo.png",
                        binary=png_bytes,
                    ),
                ],
            ),
        ],
        metadata={"test": "_MY_METADATA_"},
    )
    # Default format is now OUMI (HuggingFace-compatible)
    json_str = conv.to_json()

    assert isinstance(json_str, str)
    assert '"role": "user"' in json_str, json_str
    # IMAGE_BINARY -> {"type": "image", "url": "data:..."}
    assert '"type": "image"' in json_str
    assert f'"url": "data:image/png;base64,{_SMALL_B64_IMAGE}"' in json_str, json_str
    # TEXT -> {"type": "text", "text": "..."}
    assert '"text": "Hello"' in json_str, json_str
    assert '"type": "text"' in json_str
    assert json_str.count('"type": "text"') == 2, json_str
    assert '"role": "assistant"' in json_str
    # IMAGE_URL with binary -> {"type": "image", "url": "...", "binary": "..."}
    assert '"url": "/tmp/foo.png"' in json_str
    assert f'"binary": "{_SMALL_B64_IMAGE}"' in json_str
    assert '"text": "Hi there!"' in json_str
    assert '"test": "_MY_METADATA_"' in json_str


def test_conversation_from_json_legacy():
    json_str = '{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}], "metadata": {"test": "metadata"}}'  # noqa: E501
    conv = Conversation.from_json(json_str)

    assert isinstance(conv, Conversation)
    assert len(conv.messages) == 2
    assert conv.metadata == {"test": "metadata"}
    assert conv.messages[0].role == Role.USER
    assert conv.messages[0].content == "Hello"
    assert conv.messages[1].role == Role.ASSISTANT
    assert conv.messages[1].content == "Hi there!"


def test_conversation_from_json_with_unknown_fields():
    json_str = '{"messages": [{"role": "user", "content": "Hello", "foo_unknown": "Z"}, {"role": "assistant", "content": "Hi there!"}], "metadata": {"test": "metadata"}}'  # noqa: E501
    conv = Conversation.from_json(json_str)

    assert isinstance(conv, Conversation)
    assert len(conv.messages) == 2
    assert conv.metadata == {"test": "metadata"}
    assert conv.messages[0].role == Role.USER
    assert conv.messages[0].content == "Hello"
    assert conv.messages[1].role == Role.ASSISTANT
    assert conv.messages[1].content == "Hi there!"


def test_conversation_from_json_compound_simple():
    json_str = '{"messages": [{"role": "user", "content": [{"type": "text", "content": "Hello"}]}, {"role": "assistant", "content": "Hi there!"}], "metadata": {"test": "metadata"}}'  # noqa: E501
    conv = Conversation.from_json(json_str)

    assert isinstance(conv, Conversation)
    assert len(conv.messages) == 2
    assert conv.metadata == {"test": "metadata"}
    assert conv.messages[0].role == Role.USER
    assert isinstance(conv.messages[0].content, list)
    assert conv.messages[0].content == [ContentItem(type=Type.TEXT, content="Hello")]
    assert conv.messages[1].role == Role.ASSISTANT
    assert conv.messages[1].content == "Hi there!"


def test_roundtrip_dict_legacy(root_testdata_dir):
    from oumi.core.types.conversation import OutputFormat

    original = Conversation(
        messages=[
            Message(id="001", role=Role.SYSTEM, content="Behave!"),
            Message(id="", role=Role.ASSISTANT, content="Hi there!"),
            Message(
                role=Role.TOOL,
                content="lalala",
            ),
            Message(
                id="xyz",
                role=Role.USER,
                content="oumi_logo_dark",
            ),
        ],
        metadata={"test": "metadata"},
    )
    # Use OUMI_LEGACY format to preserve exact equality
    conv_dict = original.to_dict(output_format=OutputFormat.OUMI_LEGACY)
    reconstructed = Conversation.from_dict(conv_dict)

    assert original == reconstructed


def test_roundtrip_dict_compound_mixed_content(root_testdata_dir):
    from oumi.core.types.conversation import OutputFormat

    png_logo_bytes = load_image_png_bytes_from_path(
        root_testdata_dir / "images" / "oumi_logo_dark.png"
    )
    png_small_image_bytes = _create_test_image_bytes()

    original = Conversation(
        messages=[
            Message(id="001", role=Role.SYSTEM, content="Behave!"),
            Message(id="", role=Role.ASSISTANT, content="Hi there!"),
            Message(
                id="z072",
                role=Role.USER,
                content=[
                    ContentItem(binary=png_logo_bytes, type=Type.IMAGE_BINARY),
                    ContentItem(binary=png_small_image_bytes, type=Type.IMAGE_BINARY),
                    ContentItem(
                        content="https://www.oumi.ai/logo.png",
                        type=Type.IMAGE_URL,
                    ),
                ],
            ),
            Message(
                id="_xyz",
                role=Role.TOOL,
                content=[
                    ContentItem(
                        content=str(
                            root_testdata_dir / "images" / "oumi_logo_dark.png"
                        ),
                        binary=png_logo_bytes,
                        type=Type.IMAGE_PATH,
                    ),
                    ContentItem(
                        content="http://oumi.ai/bzz.png",
                        binary=png_small_image_bytes,
                        type=Type.IMAGE_URL,
                    ),
                    ContentItem(content="<@>", type=Type.TEXT),
                ],
            ),
        ],
        metadata={"a": "b", "b": "c"},
    )
    # Use OUMI_LEGACY format to preserve exact equality with legacy types
    conv_dict = original.to_dict(output_format=OutputFormat.OUMI_LEGACY)
    reconstructed = Conversation.from_dict(conv_dict)

    assert original == reconstructed


def test_roundtrip_json_legacy(root_testdata_dir):
    from oumi.core.types.conversation import OutputFormat

    original = Conversation(
        messages=[
            Message(id="001", role=Role.SYSTEM, content="Behave!"),
            Message(id="", role=Role.ASSISTANT, content="Hi there!"),
            Message(
                role=Role.USER,
                content="",
            ),
            Message(
                id="xyz",
                role=Role.TOOL,
                content="oumi_logo_dark",
            ),
        ],
        metadata={"test": "metadata"},
    )
    # Use OUMI_LEGACY format to preserve exact equality
    json_str = original.to_json(output_format=OutputFormat.OUMI_LEGACY)
    reconstructed = Conversation.from_json(json_str)

    assert original == reconstructed


def test_roundtrip_json_compound_mixed_content(root_testdata_dir):
    from oumi.core.types.conversation import OutputFormat

    png_logo_bytes = load_image_png_bytes_from_path(
        root_testdata_dir / "images" / "oumi_logo_light.png"
    )
    png_small_image_bytes = _create_test_image_bytes()

    original = Conversation(
        messages=[
            Message(id="001", role=Role.SYSTEM, content="Behave!"),
            Message(id="", role=Role.ASSISTANT, content="Hi there!"),
            Message(
                id="z072",
                role=Role.USER,
                content=[
                    ContentItem(binary=png_logo_bytes, type=Type.IMAGE_BINARY),
                    ContentItem(binary=png_small_image_bytes, type=Type.IMAGE_BINARY),
                    ContentItem(
                        content="https://www.oumi.ai/logo.png",
                        type=Type.IMAGE_URL,
                    ),
                ],
            ),
            Message(
                id="_xyz",
                role=Role.TOOL,
                content=[
                    ContentItem(
                        content=str(
                            root_testdata_dir / "images" / "oumi_logo_dark.png"
                        ),
                        binary=png_logo_bytes,
                        type=Type.IMAGE_PATH,
                    ),
                    ContentItem(
                        content="http://oumi.ai/bzz.png",
                        binary=png_small_image_bytes,
                        type=Type.IMAGE_URL,
                    ),
                    ContentItem(content="<@>", type=Type.TEXT),
                ],
            ),
        ],
        metadata={"a": "b", "b": "c"},
    )
    # Use OUMI_LEGACY format to preserve exact equality with legacy types
    json_str = original.to_json(output_format=OutputFormat.OUMI_LEGACY)
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
                        "role": "user",
                        "content": [
                            {
                                "binary": "INVALID_BASE64!",
                                "type": "image_binary",
                            }
                        ],
                    },
                ],
                "metadata": {"test": "metadatazzz"},
            }
        )


def test_empty_content_list():
    message = Message(role=Role.ASSISTANT, content=[])
    assert isinstance(message.content, list)
    assert len(message.content) == 0


def test_empty_content_string():
    message = Message(role=Role.USER, content="")
    assert isinstance(message.content, str)
    assert len(message.content) == 0


def test_incorrect_message_content_item_type():
    with pytest.raises(ValueError, match="Input should be a valid string"):
        ContentItem(
            type=Type.TEXT,
            content=cast(str, 12345.7),  # Hacky way to pass a number as content.
        )
    with pytest.raises(ValueError, match="'content' or 'text' must be provided"):
        ContentItem(
            type=Type.TEXT,
        )
    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValueError, match="No image bytes in message content item"):
            ContentItem(type=Type.IMAGE_BINARY, binary=b"")
    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValueError, match="Content not provided"):
            ContentItem(type=Type.IMAGE_URL, binary=b"")
    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValueError, match="Content not provided"):
            ContentItem(type=Type.IMAGE_PATH, binary=b"")
    with pytest.raises(ValueError, match="Binary data cannot be provided for TEXT"):
        ContentItem(type=Type.TEXT, content="hello", binary=b"")


@pytest.mark.parametrize(
    "role",
    [Role.USER, Role.ASSISTANT, Role.TOOL, Role.SYSTEM],
)
def test_content_item_methods_mixed_items(role: Role):
    text_item1 = ContentItem(type=Type.TEXT, content="aaa")
    image_item1 = ContentItem(type=Type.IMAGE_BINARY, binary=_create_test_image_bytes())
    text_item2 = ContentItem(type=Type.TEXT, content=" B B ")
    image_item2 = ContentItem(
        type=Type.IMAGE_PATH,
        content="/tmp/test/dummy.jpeg",
        binary=_create_test_image_bytes(),
    )
    text_item3 = ContentItem(type=Type.TEXT, content="CC")

    message = Message(
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

    assert message.count_content_items() == ContentItemCounts(
        total_items=5, image_items=2, text_items=3
    )


@pytest.mark.parametrize(
    "image_type",
    [Type.IMAGE_BINARY, Type.IMAGE_PATH, Type.IMAGE_URL],
)
def test_content_item_methods_single_image(image_type):
    test_image_item = ContentItem(
        type=image_type,
        content=(None if image_type == Type.IMAGE_BINARY else "foo"),
        binary=(
            _create_test_image_bytes() if image_type == Type.IMAGE_BINARY else None
        ),
    )
    message = Message(
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

    assert message.count_content_items() == ContentItemCounts(
        total_items=1, image_items=1, text_items=0
    )


def test_content_item_methods_triple_image():
    test_image_item1 = ContentItem(
        type=Type.IMAGE_BINARY,
        binary=(_create_test_image_bytes()),
    )
    test_image_item2 = ContentItem(
        type=Type.IMAGE_URL,
        content="http://oumi.ai/a.png",
    )
    test_image_item3 = ContentItem(
        type=Type.IMAGE_PATH,
        content="/tmp/oumi.ai/b.gif",
    )
    message = Message(
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

    assert message.count_content_items() == ContentItemCounts(
        total_items=3, image_items=3, text_items=0
    )


def test_content_item_methods_legacy_text():
    test_text_item = ContentItem(type=Type.TEXT, content="bzzz")
    message = Message(
        role=Role.USER,
        content=(test_text_item.content or ""),
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

    assert message.count_content_items() == ContentItemCounts(
        total_items=1, image_items=0, text_items=1
    )


def test_content_item_methods_double_text():
    test_text_item = ContentItem(type=Type.TEXT, content="bzzz")
    message = Message(
        role=Role.USER,
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

    assert message.count_content_items() == ContentItemCounts(
        total_items=2, image_items=0, text_items=2
    )


def test_role_str_repr():
    assert str(Role.ASSISTANT) == "assistant"
    assert "assistant" in repr(Role.ASSISTANT)
    assert str(Role.USER) == "user"
    assert "user" in repr(Role.USER)
    assert str(Role.TOOL) == "tool"
    assert "tool" in repr(Role.TOOL)


def test_type_str_repr():
    assert str(Type.TEXT) == "text"
    assert "text" in repr(Type.TEXT)
    assert str(Type.IMAGE_BINARY) == "image_binary"
    assert "image_binary" in repr(Type.IMAGE_BINARY)
    assert str(Type.IMAGE_URL) == "image_url"
    assert "image_url" in repr(Type.IMAGE_URL)
    assert str(Type.IMAGE_PATH) == "image_path"
    assert "image_path" in repr(Type.IMAGE_PATH)


def test_frozen_message_content_item():
    test_item = ContentItem(type=Type.TEXT, content="init")
    with pytest.raises(pydantic.ValidationError, match="Instance is frozen"):
        test_item.content = "foo"
    assert test_item.content == "init"

    with pytest.raises(pydantic.ValidationError, match="Instance is frozen"):
        test_item.type = Type.IMAGE_BINARY
    assert test_item.type == Type.TEXT


def test_frozen_message():
    test_item = ContentItem(type=Type.TEXT, content="bzzz")
    message = Message(
        id="007",
        role=Role.ASSISTANT,
        content=[test_item, test_item],
    )

    with pytest.raises(pydantic.ValidationError, match="Instance is frozen"):
        message.id = "001"
    assert message.id == "007"

    with pytest.raises(pydantic.ValidationError, match="Instance is frozen"):
        message.role = Role.TOOL
    assert message.role == Role.ASSISTANT

    with pytest.raises(pydantic.ValidationError, match="Instance is frozen"):
        message.content = "Hey"
    assert isinstance(message.content, list)
    assert len(message.content) == 2

    # Pydantic "frozen" only ensures that `message.content` can't be re-assigned
    # but it doesn't enforce the field object itself is immutable.
    message.content.append(test_item)
    assert isinstance(message.content, list)
    assert len(message.content) == 3


def test_conversation_metadata_independence():
    conv1 = Conversation(messages=[Message(role=Role.USER, content="hi")])
    conv2 = Conversation(messages=[Message(role=Role.USER, content="hello")])

    conv1.metadata["foo"] = "bar"

    assert conv2.metadata == {}
    assert conv1.metadata == {"foo": "bar"}


# ============================================================================
# New HuggingFace format tests (Type.IMAGE, text/url/path fields)
# ============================================================================


class TestImageType:
    """Tests for the new unified IMAGE type."""

    def test_image_with_url(self):
        """Test creating IMAGE content item with URL."""
        item = ContentItem(type=Type.IMAGE, url="https://example.com/image.jpg")
        assert item.type == Type.IMAGE
        assert item.url == "https://example.com/image.jpg"
        assert item.path is None
        assert item.is_image()
        assert not item.is_text()

    def test_image_with_path(self):
        """Test creating IMAGE content item with local path."""
        item = ContentItem(type=Type.IMAGE, path="/path/to/image.jpg")
        assert item.type == Type.IMAGE
        assert item.path == "/path/to/image.jpg"
        assert item.url is None
        assert item.is_image()

    def test_image_with_binary(self):
        """Test creating IMAGE content item with binary data."""
        png_bytes = _create_test_image_bytes()
        item = ContentItem(type=Type.IMAGE, url="data:image/png;base64,test", binary=png_bytes)
        assert item.type == Type.IMAGE
        assert item.binary == png_bytes
        assert item.is_image()

    def test_image_with_data_uri(self):
        """Test creating IMAGE content item with data URI."""
        data_uri = f"data:image/png;base64,{_SMALL_B64_IMAGE}"
        item = ContentItem(type=Type.IMAGE, url=data_uri)
        assert item.type == Type.IMAGE
        assert item.url == data_uri
        assert item.is_image()

    def test_image_repr(self):
        """Test IMAGE type string representation."""
        item = ContentItem(type=Type.IMAGE, url="https://example.com/image.jpg")
        assert repr(item) == "<IMAGE>"

    def test_image_validation_requires_url_or_path(self):
        """Test that IMAGE type requires url, path, or binary."""
        with pytest.raises(ValueError, match="provide 'url', 'path', or 'binary'"):
            ContentItem(type=Type.IMAGE)

    def test_image_validation_cannot_have_both_url_and_path(self):
        """Test that IMAGE type cannot have both url and path."""
        with pytest.raises(ValueError, match="Cannot provide both 'url' and 'path'"):
            ContentItem(
                type=Type.IMAGE,
                url="https://example.com/image.jpg",
                path="/path/to/image.jpg",
            )

    def test_image_validation_cannot_have_content(self):
        """Test that IMAGE type cannot use content field."""
        with pytest.raises(ValueError, match="'content' and 'text' fields cannot be used"):
            ContentItem(
                type=Type.IMAGE,
                url="https://example.com/image.jpg",
                content="should not be here",
            )

    def test_get_image_url(self):
        """Test get_image_url helper method."""
        item = ContentItem(type=Type.IMAGE, url="https://example.com/image.jpg")
        assert item.get_image_url() == "https://example.com/image.jpg"

    def test_get_image_path(self):
        """Test get_image_path helper method."""
        item = ContentItem(type=Type.IMAGE, path="/path/to/image.jpg")
        assert item.get_image_path() == "/path/to/image.jpg"


class TestTextFieldAlias:
    """Tests for the 'text' field alias (HuggingFace format)."""

    def test_text_field_for_text_type(self):
        """Test creating TEXT content item with text field."""
        item = ContentItem(type=Type.TEXT, text="Hello, world!")
        assert item.type == Type.TEXT
        assert item.text == "Hello, world!"
        assert item.is_text()

    def test_text_field_repr(self):
        """Test TEXT type with text field string representation."""
        item = ContentItem(type=Type.TEXT, text="Hello")
        assert repr(item) == "Hello"

    def test_get_text_prefers_text_field(self):
        """Test that get_text() returns text field when both are set."""
        item = ContentItem(type=Type.TEXT, text="text_field", content="content_field")
        assert item.get_text() == "text_field"

    def test_get_text_falls_back_to_content(self):
        """Test that get_text() falls back to content field."""
        item = ContentItem(type=Type.TEXT, content="content_field")
        assert item.get_text() == "content_field"

    def test_text_validation_requires_content_or_text(self):
        """Test that TEXT type requires content or text field."""
        with pytest.raises(ValueError, match="'content' or 'text' must be provided"):
            ContentItem(type=Type.TEXT)

    def test_text_cannot_have_url_or_path(self):
        """Test that TEXT type cannot use url or path fields."""
        with pytest.raises(ValueError, match="'url' and 'path' fields cannot be used"):
            ContentItem(type=Type.TEXT, text="Hello", url="https://example.com")


class TestDeprecationWarnings:
    """Tests for deprecation warnings on legacy types."""

    def test_image_url_deprecation_warning(self):
        """Test that IMAGE_URL emits deprecation warning."""
        with pytest.warns(DeprecationWarning, match="IMAGE_URL is deprecated"):
            ContentItem(type=Type.IMAGE_URL, content="https://example.com/image.jpg")

    def test_image_path_deprecation_warning(self):
        """Test that IMAGE_PATH emits deprecation warning."""
        with pytest.warns(DeprecationWarning, match="IMAGE_PATH is deprecated"):
            ContentItem(type=Type.IMAGE_PATH, content="/path/to/image.jpg")

    def test_image_binary_deprecation_warning(self):
        """Test that IMAGE_BINARY emits deprecation warning."""
        with pytest.warns(DeprecationWarning, match="IMAGE_BINARY is deprecated"):
            ContentItem(type=Type.IMAGE_BINARY, binary=_create_test_image_bytes())

    def test_is_deprecated_method(self):
        """Test is_deprecated() method on Type enum."""
        assert Type.IMAGE_URL.is_deprecated()
        assert Type.IMAGE_PATH.is_deprecated()
        assert Type.IMAGE_BINARY.is_deprecated()
        assert not Type.IMAGE.is_deprecated()
        assert not Type.TEXT.is_deprecated()


class TestHuggingFaceFormatParsing:
    """Tests for parsing HuggingFace format input."""

    def test_parse_hf_text_format(self):
        """Test parsing HuggingFace text format."""
        conv_dict = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello from HF format"}],
                }
            ]
        }
        conv = Conversation.from_dict(conv_dict)
        assert len(conv.messages) == 1
        assert isinstance(conv.messages[0].content, list)
        item = conv.messages[0].content[0]
        assert item.type == Type.TEXT
        assert item.text == "Hello from HF format"

    def test_parse_hf_image_url_format(self):
        """Test parsing HuggingFace image URL format."""
        conv_dict = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": "https://example.com/image.jpg"}
                    ],
                }
            ]
        }
        conv = Conversation.from_dict(conv_dict)
        item = conv.messages[0].content[0]
        assert item.type == Type.IMAGE
        assert item.url == "https://example.com/image.jpg"

    def test_parse_hf_image_path_format(self):
        """Test parsing HuggingFace image path format."""
        conv_dict = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image", "path": "/path/to/image.jpg"}],
                }
            ]
        }
        conv = Conversation.from_dict(conv_dict)
        item = conv.messages[0].content[0]
        assert item.type == Type.IMAGE
        assert item.path == "/path/to/image.jpg"

    def test_parse_mixed_hf_format(self):
        """Test parsing mixed HuggingFace format message."""
        conv_dict = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this image?"},
                        {"type": "image", "url": "https://example.com/image.jpg"},
                    ],
                }
            ]
        }
        conv = Conversation.from_dict(conv_dict)
        assert len(conv.messages[0].content) == 2
        assert conv.messages[0].content[0].type == Type.TEXT
        assert conv.messages[0].content[0].text == "What is in this image?"
        assert conv.messages[0].content[1].type == Type.IMAGE
        assert conv.messages[0].content[1].url == "https://example.com/image.jpg"


class TestImageTypeRoundtrip:
    """Tests for IMAGE type roundtrip serialization."""

    def test_roundtrip_dict_image_url(self):
        """Test roundtrip for IMAGE with URL."""
        original = Conversation(
            messages=[
                Message(
                    role=Role.USER,
                    content=[
                        ContentItem(type=Type.IMAGE, url="https://example.com/img.jpg"),
                        ContentItem(type=Type.TEXT, text="What is this?"),
                    ],
                )
            ]
        )
        conv_dict = original.to_dict()
        reconstructed = Conversation.from_dict(conv_dict)
        assert original == reconstructed

    def test_roundtrip_dict_image_path(self):
        """Test roundtrip for IMAGE with path."""
        original = Conversation(
            messages=[
                Message(
                    role=Role.USER,
                    content=[ContentItem(type=Type.IMAGE, path="/path/to/image.jpg")],
                )
            ]
        )
        conv_dict = original.to_dict()
        reconstructed = Conversation.from_dict(conv_dict)
        assert original == reconstructed

    def test_roundtrip_dict_image_binary(self):
        """Test roundtrip for IMAGE with binary data."""
        png_bytes = _create_test_image_bytes()
        data_uri = f"data:image/png;base64,{_SMALL_B64_IMAGE}"
        original = Conversation(
            messages=[
                Message(
                    role=Role.USER,
                    content=[ContentItem(type=Type.IMAGE, url=data_uri, binary=png_bytes)],
                )
            ]
        )
        conv_dict = original.to_dict()
        reconstructed = Conversation.from_dict(conv_dict)
        assert original == reconstructed

    def test_roundtrip_json_image_type(self):
        """Test JSON roundtrip for IMAGE type."""
        original = Conversation(
            messages=[
                Message(
                    role=Role.USER,
                    content=[
                        ContentItem(type=Type.TEXT, text="Look at this"),
                        ContentItem(type=Type.IMAGE, url="https://example.com/img.jpg"),
                    ],
                )
            ]
        )
        json_str = original.to_json()
        reconstructed = Conversation.from_json(json_str)
        assert original == reconstructed


class TestOutputFormat:
    """Tests for OutputFormat serialization options."""

    def test_oumi_format_text(self):
        """Test OUMI format for text content."""
        from oumi.core.types.conversation import OutputFormat

        item = ContentItem(type=Type.TEXT, content="Hello")
        result = item.to_dict(output_format=OutputFormat.OUMI)
        assert result == {"type": "text", "text": "Hello"}

    def test_oumi_format_image_url(self):
        """Test OUMI format for IMAGE with URL."""
        from oumi.core.types.conversation import OutputFormat

        item = ContentItem(type=Type.IMAGE, url="https://example.com/img.jpg")
        result = item.to_dict(output_format=OutputFormat.OUMI)
        assert result == {"type": "image", "url": "https://example.com/img.jpg"}

    def test_oumi_format_image_path(self):
        """Test OUMI format for IMAGE with path."""
        from oumi.core.types.conversation import OutputFormat

        item = ContentItem(type=Type.IMAGE, path="/path/to/img.jpg")
        result = item.to_dict(output_format=OutputFormat.OUMI)
        assert result == {"type": "image", "path": "/path/to/img.jpg"}

    def test_openai_format_text(self):
        """Test OPENAI format for text content."""
        from oumi.core.types.conversation import OutputFormat

        item = ContentItem(type=Type.TEXT, content="Hello")
        result = item.to_dict(output_format=OutputFormat.OPENAI)
        assert result == {"type": "text", "text": "Hello"}

    def test_openai_format_image(self):
        """Test OPENAI format for IMAGE with URL."""
        from oumi.core.types.conversation import OutputFormat

        item = ContentItem(type=Type.IMAGE, url="https://example.com/img.jpg")
        result = item.to_dict(output_format=OutputFormat.OPENAI)
        assert result == {
            "type": "image_url",
            "image_url": {"url": "https://example.com/img.jpg"},
        }

    def test_openai_format_legacy_image_binary(self):
        """Test OPENAI format for legacy IMAGE_BINARY."""
        from oumi.core.types.conversation import OutputFormat

        png_bytes = _create_test_image_bytes()
        item = ContentItem(type=Type.IMAGE_BINARY, binary=png_bytes)
        result = item.to_dict(output_format=OutputFormat.OPENAI)
        assert result["type"] == "image_url"
        assert "image_url" in result
        assert result["image_url"]["url"].startswith("data:image/png;base64,")

    def test_legacy_format_text(self):
        """Test OUMI_LEGACY format for text content."""
        from oumi.core.types.conversation import OutputFormat

        item = ContentItem(type=Type.TEXT, content="Hello")
        result = item.to_dict(output_format=OutputFormat.OUMI_LEGACY)
        assert result == {"type": "text", "content": "Hello"}

    def test_legacy_format_image_url(self):
        """Test OUMI_LEGACY format for legacy IMAGE_URL."""
        from oumi.core.types.conversation import OutputFormat

        item = ContentItem(type=Type.IMAGE_URL, content="https://example.com/img.jpg")
        result = item.to_dict(output_format=OutputFormat.OUMI_LEGACY)
        assert result == {"type": "image_url", "content": "https://example.com/img.jpg"}

    def test_legacy_format_image_path(self):
        """Test OUMI_LEGACY format for legacy IMAGE_PATH."""
        from oumi.core.types.conversation import OutputFormat

        item = ContentItem(type=Type.IMAGE_PATH, content="/path/to/img.jpg")
        result = item.to_dict(output_format=OutputFormat.OUMI_LEGACY)
        assert result == {"type": "image_path", "content": "/path/to/img.jpg"}

    def test_legacy_format_image_binary(self):
        """Test OUMI_LEGACY format for legacy IMAGE_BINARY."""
        from oumi.core.types.conversation import OutputFormat

        png_bytes = _create_test_image_bytes()
        item = ContentItem(type=Type.IMAGE_BINARY, binary=png_bytes)
        result = item.to_dict(output_format=OutputFormat.OUMI_LEGACY)
        assert result["type"] == "image_binary"
        assert "binary" in result
        assert result["binary"] == _SMALL_B64_IMAGE

    def test_conversation_to_dict_all_formats(self):
        """Test Conversation.to_dict with all formats."""
        from oumi.core.types.conversation import OutputFormat

        conv = Conversation(
            messages=[
                Message(
                    role=Role.USER,
                    content=[
                        ContentItem(type=Type.TEXT, content="What is this?"),
                        ContentItem(type=Type.IMAGE, url="https://example.com/img.jpg"),
                    ],
                )
            ]
        )

        # OUMI format (default)
        oumi_dict = conv.to_dict(output_format=OutputFormat.OUMI)
        assert oumi_dict["messages"][0]["content"][0] == {"type": "text", "text": "What is this?"}
        assert oumi_dict["messages"][0]["content"][1] == {"type": "image", "url": "https://example.com/img.jpg"}

        # OPENAI format
        openai_dict = conv.to_dict(output_format=OutputFormat.OPENAI)
        assert openai_dict["messages"][0]["content"][0] == {"type": "text", "text": "What is this?"}
        assert openai_dict["messages"][0]["content"][1] == {
            "type": "image_url",
            "image_url": {"url": "https://example.com/img.jpg"},
        }

        # OUMI_LEGACY format - IMAGE type maps to image_url
        legacy_dict = conv.to_dict(output_format=OutputFormat.OUMI_LEGACY)
        assert legacy_dict["messages"][0]["content"][0] == {"type": "text", "content": "What is this?"}
        assert legacy_dict["messages"][0]["content"][1] == {"type": "image_url", "content": "https://example.com/img.jpg"}
