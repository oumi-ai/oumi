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
    conv_dict = conv.to_dict()

    assert isinstance(conv_dict, dict)
    assert "messages" in conv_dict
    assert len(conv_dict["messages"]) == 2
    assert conv_dict["metadata"] == {"test": "metadata"}
    assert conv_dict["messages"][0]["role"] == "user"
    assert conv_dict["messages"][0]["content"] == [{"content": "Hello", "type": "text"}]
    assert conv_dict["messages"][1]["role"] == "assistant"
    assert conv_dict["messages"][1]["content"] == [
        {"content": "Hi there!", "type": "text"}
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
    conv_dict = conv.to_dict()

    assert isinstance(conv_dict, dict)
    assert "messages" in conv_dict
    assert len(conv_dict["messages"]) == 2
    assert conv_dict["metadata"] == {"test": "metadata"}
    assert conv_dict["messages"][0]["role"] == "user"
    assert conv_dict["messages"][0]["content"] == [
        {
            "binary": _SMALL_B64_IMAGE,
            "type": "image_binary",
        },
        {"content": "Hello", "type": "text"},
    ]
    assert conv_dict["messages"][1]["role"] == "assistant"
    assert conv_dict["messages"][1]["content"] == [
        {"content": "Hi there!", "type": "text"},
        {
            "binary": _SMALL_B64_IMAGE,
            "content": "/tmp/foo.png",
            "type": "image_url",
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
    json_str = conv.to_json()

    assert isinstance(json_str, str)
    assert '"role":"user"' in json_str
    assert '"content":"Hello"' in json_str
    assert '"role":"assistant"' in json_str
    assert '"content":"Hi there!"' in json_str
    assert '"test":"metadata"' in json_str


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
    json_str = conv.to_json()

    assert isinstance(json_str, str)
    assert '"role":"user"' in json_str, json_str
    assert '"type":"image_binary"' in json_str
    assert ('"binary":"' + _SMALL_B64_IMAGE + '"') in json_str, json_str
    assert json_str.count('"binary":"' + _SMALL_B64_IMAGE + '"') == 2, json_str
    assert '"content":"Hello"' in json_str, json_str
    assert '"type":"text"' in json_str
    assert json_str.count('"type":"text"') == 2, json_str
    assert '"role":"assistant"' in json_str
    assert '"type":"image_url"' in json_str
    assert '"content":"Hi there!"' in json_str
    assert '"test":"_MY_METADATA_"' in json_str


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
    conv_dict = original.to_dict()
    reconstructed = Conversation.from_dict(conv_dict)

    assert original == reconstructed


def test_roundtrip_dict_compound_mixed_content(root_testdata_dir):
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
    conv_dict = original.to_dict()
    reconstructed = Conversation.from_dict(conv_dict)

    assert original == reconstructed


def test_roundtrip_json_legacy(root_testdata_dir):
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
    json_str = original.to_json()
    reconstructed = Conversation.from_json(json_str)

    assert original == reconstructed


def test_roundtrip_json_compound_mixed_content(root_testdata_dir):
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
    json_str = original.to_json()
    reconstructed = Conversation.from_json(json_str)

    assert original == reconstructed


def test_roundtrip_proto_legacy(root_testdata_dir):
    original = Conversation(
        messages=[
            Message(id="001", role=Role.SYSTEM, content="Behave!"),
            Message(role=Role.ASSISTANT, content="Hi there!"),
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
    convo_proto = original.to_proto()
    reconstructed = Conversation.from_proto(convo_proto)

    assert original == reconstructed, (
        f"\n\noriginal: {original.to_dict()}\n"
        f"\n\nreconstructed: {reconstructed.to_dict()}\n"
        f"\n\nproto: {convo_proto.SerializeToString()}"
    )


def test_roundtrip_proto_compound_mixed_content(root_testdata_dir):
    png_logo_bytes = load_image_png_bytes_from_path(
        root_testdata_dir / "images" / "oumi_logo_light.png"
    )
    png_small_image_bytes = _create_test_image_bytes()

    original = Conversation(
        messages=[
            Message(id="001", role=Role.SYSTEM, content="Behave!"),
            Message(role=Role.ASSISTANT, content="Hi there!"),
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
    convo_proto = original.to_proto()
    reconstructed = Conversation.from_proto(convo_proto)

    assert original == reconstructed, (
        f"\n\noriginal: {original.to_dict()}\n"
        f"\n\nreconstructed: {reconstructed.to_dict()}\n"
        f"\n\nproto: {convo_proto.SerializeToString()}"
    )


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
    with pytest.raises(ValueError, match="Either content or binary must be provided"):
        ContentItem(
            type=Type.TEXT,
        )
    with pytest.raises(ValueError, match="No image bytes in message content item"):
        ContentItem(type=Type.IMAGE_BINARY, binary=b"")
    with pytest.raises(ValueError, match="Content not provided"):
        ContentItem(type=Type.IMAGE_URL, binary=b"")
    with pytest.raises(ValueError, match="Content not provided"):
        ContentItem(type=Type.IMAGE_PATH, binary=b"")
    with pytest.raises(ValueError, match="Binary can only be provided for images"):
        ContentItem(type=Type.TEXT, binary=b"")


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
