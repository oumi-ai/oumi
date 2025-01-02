import copy
import tempfile
from pathlib import Path
from typing import Final

import PIL.Image
import pytest
import responses

from oumi.core.types.conversation import ContentItem, Conversation, Message, Role, Type
from oumi.utils.conversation_utils import (
    base64encode_content_item_image_bytes,
    convert_message_to_json_content,
    convert_message_to_json_content_list,
    create_list_of_message_json_dicts,
    load_image_bytes_to_content_item,
)
from oumi.utils.image_utils import (
    create_png_bytes_from_image,
)
from oumi.utils.io_utils import get_oumi_root_directory

_TEST_IMAGE_DIR: Final[Path] = (
    get_oumi_root_directory().parent.parent.resolve() / "tests" / "testdata" / "images"
)


def create_test_text_only_conversation():
    return Conversation(
        messages=[
            Message(content="You are an assistant!", role=Role.SYSTEM),
            Message(content="Hello", role=Role.USER),
            Message(content="Hi there!", role=Role.ASSISTANT),
            Message(content="How are you?", role=Role.USER),
        ]
    )


def create_test_png_image_bytes() -> bytes:
    pil_image = PIL.Image.new(mode="RGB", size=(32, 48))
    return create_png_bytes_from_image(pil_image)


def create_test_png_image_base64_str() -> str:
    return base64encode_content_item_image_bytes(
        ContentItem(binary=create_test_png_image_bytes(), type=Type.IMAGE_BINARY),
        add_mime_prefix=True,
    )


def create_test_multimodal_text_image_conversation():
    png_bytes = create_test_png_image_bytes()
    return Conversation(
        messages=[
            Message(content="You are an assistant!", role=Role.SYSTEM),
            Message(
                role=Role.USER,
                content=[
                    ContentItem(binary=png_bytes, type=Type.IMAGE_BINARY),
                    ContentItem(content="Hello", type=Type.TEXT),
                    ContentItem(content="there", type=Type.TEXT),
                ],
            ),
            Message(
                role=Role.ASSISTANT,
                content=[
                    ContentItem(content="Greetings!", type=Type.TEXT),
                    ContentItem(
                        content="http://oumi.ai/test.png",
                        type=Type.IMAGE_URL,
                    ),
                ],
            ),
            Message(
                role=Role.USER,
                content=[
                    ContentItem(content="Describe this image", type=Type.TEXT),
                    ContentItem(
                        content=str(
                            _TEST_IMAGE_DIR / "the_great_wave_off_kanagawa.jpg"
                        ),
                        type=Type.IMAGE_PATH,
                    ),
                ],
            ),
        ]
    )


def test_load_image_bytes_to_message_noop_text():
    input_item = ContentItem(type=Type.TEXT, content="hello")
    saved_input_item = copy.deepcopy(input_item)

    output_item = load_image_bytes_to_content_item(input_item)
    assert id(output_item) == id(input_item)
    assert output_item == saved_input_item


def test_load_image_bytes_to_message_noop_image_binary():
    pil_image = PIL.Image.new(mode="RGB", size=(32, 48))
    input_item = ContentItem(
        type=Type.IMAGE_BINARY,
        binary=create_png_bytes_from_image(pil_image),
    )
    saved_input_item = copy.deepcopy(input_item)

    output_item = load_image_bytes_to_content_item(input_item)
    assert id(output_item) == id(input_item)
    assert output_item == saved_input_item


def test_load_image_bytes_to_message_image_path():
    pil_image = PIL.Image.new(mode="RGB", size=(32, 48))
    png_bytes = create_png_bytes_from_image(pil_image)

    with tempfile.TemporaryDirectory() as output_temp_dir:
        png_filename: Path = Path(output_temp_dir) / "test.png"
        with png_filename.open(mode="wb") as f:
            f.write(png_bytes)

        input_item = ContentItem(type=Type.IMAGE_PATH, content=str(png_filename))

        output_item = load_image_bytes_to_content_item(input_item)
        assert id(output_item) != id(input_item)

        expected_output_item = ContentItem(type=Type.IMAGE_BINARY, binary=png_bytes)
        assert output_item == expected_output_item


def test_load_image_bytes_to_message_image_url():
    pil_image = PIL.Image.new(mode="RGB", size=(32, 48))
    png_bytes = create_png_bytes_from_image(pil_image)

    with responses.RequestsMock() as m:
        m.add(responses.GET, "http://oumi.ai/logo.png", body=png_bytes, stream=True)

        input_item = ContentItem(type=Type.IMAGE_URL, content="http://oumi.ai/logo.png")

        output_item = load_image_bytes_to_content_item(input_item)
        assert id(output_item) != id(input_item)

        expected_output_item = ContentItem(type=Type.IMAGE_BINARY, binary=png_bytes)
        assert output_item == expected_output_item


@pytest.mark.parametrize(
    "message_type",
    [Type.IMAGE_BINARY, Type.IMAGE_PATH, Type.IMAGE_URL],
)
def test_base64encode_image_bytes(message_type: Type):
    pil_image = PIL.Image.new(mode="RGB", size=(32, 48))
    png_bytes = create_png_bytes_from_image(pil_image)

    base64_str = base64encode_content_item_image_bytes(
        ContentItem(
            type=message_type,
            binary=png_bytes,
            content=(None if message_type == Type.IMAGE_BINARY else "foo"),
        )
    )
    assert base64_str
    assert base64_str.startswith("data:image/png;base64,iVBOR")
    assert len(base64_str) >= ((4 * len(png_bytes)) / 3) + len("data:image/png;base64,")
    assert len(base64_str) <= ((4 * len(png_bytes) + 2) / 3) + len(
        "data:image/png;base64,"
    )


def test_base64encode_image_bytes_invalid_arguments():
    with pytest.raises(ValueError, match="Message type is not an image"):
        base64encode_content_item_image_bytes(
            ContentItem(type=Type.TEXT, content="hello")
        )
    with pytest.raises(ValueError, match="No image bytes in message"):
        base64encode_content_item_image_bytes(
            ContentItem(type=Type.IMAGE_BINARY, content="hi")
        )
    with pytest.raises(ValueError, match="No image bytes in message"):
        base64encode_content_item_image_bytes(
            ContentItem(type=Type.IMAGE_PATH, content="hi")
        )
    with pytest.raises(ValueError, match="No image bytes in message"):
        base64encode_content_item_image_bytes(
            ContentItem(type=Type.IMAGE_URL, content="hi")
        )


def test_convert_message_to_json_content_or_list():
    test_message = Message(role=Role.ASSISTANT, content="")
    assert convert_message_to_json_content(test_message) == ""
    assert convert_message_to_json_content_list(test_message) == [
        {
            "type": "text",
            "text": "",
        }
    ]

    test_message = Message(role=Role.ASSISTANT, content="pear peach")
    assert convert_message_to_json_content(test_message) == "pear peach"
    assert convert_message_to_json_content_list(test_message) == [
        {
            "type": "text",
            "text": "pear peach",
        }
    ]

    assert (
        convert_message_to_json_content(Message(role=Role.ASSISTANT, content=[])) == []
    )
    assert (
        convert_message_to_json_content_list(Message(role=Role.ASSISTANT, content=[]))
        == []
    )

    test_message = Message(
        role=Role.ASSISTANT, content=[ContentItem(type=Type.TEXT, content="hi")]
    )
    assert convert_message_to_json_content(test_message) == [
        {
            "text": "hi",
            "type": "text",
        },
    ]
    assert convert_message_to_json_content(
        test_message
    ) == convert_message_to_json_content_list(test_message)

    test_message = Message(
        role=Role.ASSISTANT,
        content=[
            ContentItem(type=Type.TEXT, content="hi"),
            ContentItem(type=Type.TEXT, content="there"),
        ],
    )
    assert convert_message_to_json_content(test_message) == [
        {
            "type": "text",
            "text": "hi",
        },
        {
            "type": "text",
            "text": "there",
        },
    ]
    assert convert_message_to_json_content(
        test_message
    ) == convert_message_to_json_content_list(test_message)

    png_bytes = create_test_png_image_bytes()
    png_bytes_b64 = create_test_png_image_base64_str()
    test_message = Message(
        role=Role.ASSISTANT,
        content=[
            ContentItem(type=Type.TEXT, content="hi"),
            ContentItem(type=Type.IMAGE_BINARY, binary=png_bytes),
        ],
    )
    assert convert_message_to_json_content(test_message) == [
        {
            "type": "text",
            "text": "hi",
        },
        {
            "type": "image_url",
            "image_url": {
                "url": png_bytes_b64,
            },
        },
    ]
    assert convert_message_to_json_content(
        test_message
    ) == convert_message_to_json_content_list(test_message)

    test_message = Message(
        role=Role.ASSISTANT,
        content=[
            ContentItem(
                content="http://oumi.ai/test.png",
                type=Type.IMAGE_URL,
            ),
            ContentItem(type=Type.TEXT, content="Describe this picture"),
        ],
    )
    assert convert_message_to_json_content(test_message) == [
        {
            "type": "image_url",
            "image_url": {"url": "http://oumi.ai/test.png"},
        },
        {
            "type": "text",
            "text": "Describe this picture",
        },
    ]
    assert convert_message_to_json_content(
        test_message
    ) == convert_message_to_json_content_list(test_message)

    with tempfile.TemporaryDirectory() as output_temp_dir:
        png_filename: Path = Path(output_temp_dir) / "test.png"
        with png_filename.open(mode="wb") as f:
            f.write(png_bytes)

        test_message = Message(
            role=Role.ASSISTANT,
            content=[
                ContentItem(
                    content=str(png_filename),
                    type=Type.IMAGE_PATH,
                ),
                ContentItem(type=Type.TEXT, content="Describe this picture"),
            ],
        )
        assert convert_message_to_json_content(test_message) == [
            {
                "type": "image_url",
                "image_url": {"url": png_bytes_b64},
            },
            {
                "type": "text",
                "text": "Describe this picture",
            },
        ]


def test_create_list_of_message_json_dicts_multimodal_with_grouping():
    conversation = create_test_multimodal_text_image_conversation()
    assert len(conversation.messages) == 4
    expected_base64_str = create_test_png_image_base64_str()
    assert expected_base64_str.startswith("data:image/png;base64,")

    result = create_list_of_message_json_dicts(
        conversation.messages, group_adjacent_same_role_turns=True
    )

    assert len(result) == 4
    assert [m["role"] for m in result] == ["system", "user", "assistant", "user"]

    assert result[0] == {"role": "system", "content": "You are an assistant!"}

    assert result[1]["role"] == "user"
    assert isinstance(result[1]["content"], list) and len(result[1]["content"]) == 3
    assert all([isinstance(item, dict) for item in result[1]["content"]])
    assert result[1]["content"][0] == {
        "type": "image_url",
        "image_url": {"url": expected_base64_str},
    }
    assert result[1]["content"][1] == {"type": "text", "text": "Hello"}
    assert result[1]["content"][2] == {"type": "text", "text": "there"}

    assert result[2]["role"] == "assistant"
    assert isinstance(result[2]["content"], list) and len(result[2]["content"]) == 2
    assert all([isinstance(item, dict) for item in result[2]["content"]])
    assert result[2]["content"][0] == {"type": "text", "text": "Greetings!"}
    assert result[2]["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "http://oumi.ai/test.png"},
    }

    assert result[3]["role"] == "user"
    assert isinstance(result[3]["content"], list) and len(result[3]["content"]) == 2
    assert all([isinstance(item, dict) for item in result[3]["content"]])
    assert result[3]["content"][0] == {"type": "text", "text": "Describe this image"}
    content = result[3]["content"][1]
    assert isinstance(content, dict)
    assert "image_url" in content
    image_url = content["image_url"]
    assert isinstance(image_url, dict)
    assert "url" in image_url
    tsunami_base64_image_str = image_url["url"]

    assert isinstance(tsunami_base64_image_str, str)
    assert tsunami_base64_image_str.startswith("data:image/png;base64,")
    assert content == {
        "type": "image_url",
        "image_url": {"url": tsunami_base64_image_str},
    }


@pytest.mark.parametrize(
    "conversation,group_adjacent_same_role_turns",
    [
        (create_test_multimodal_text_image_conversation(), False),
        (create_test_text_only_conversation(), False),
        (create_test_text_only_conversation(), True),
    ],
)
def test_get_list_of_message_json_dicts_multimodal_no_grouping(
    conversation: Conversation, group_adjacent_same_role_turns: bool
):
    result = create_list_of_message_json_dicts(
        conversation.messages,
        group_adjacent_same_role_turns=group_adjacent_same_role_turns,
    )

    assert len(result) == len(conversation.messages)
    assert [m["role"] for m in result] == [m.role for m in conversation.messages]

    for i in range(len(result)):
        json_dict = result[i]
        message = conversation.messages[i]
        debug_info = f"Index: {i} JSON: {json_dict} Message: {message}"
        if len(debug_info) > 1024:
            debug_info = debug_info[:1024] + " ..."

        assert "role" in json_dict, debug_info
        assert message.role == json_dict["role"], debug_info
        if isinstance(message.content, str):
            assert isinstance(json_dict["content"], str), debug_info
            assert message.content == json_dict["content"], debug_info
        else:
            assert isinstance(message.content, list), debug_info
            assert "content" in json_dict, debug_info
            assert isinstance(json_dict["content"], list), debug_info
            assert len(message.content) == len(json_dict["content"]), debug_info

            assert message.contains_images(), debug_info

            for idx, item in enumerate(message.content):
                json_item = json_dict["content"][idx]
                assert isinstance(json_item, dict)
                assert "type" in json_item, debug_info

                if item.is_text():
                    assert json_item["type"] == "text", debug_info
                    assert json_item["text"] == item.content, debug_info
                elif item.is_image():
                    assert json_item["type"] == "image_url", debug_info
                    assert "image_url" in json_item, debug_info
                    assert isinstance(json_item["image_url"], dict), debug_info
                    assert "url" in json_item["image_url"], debug_info
                    assert isinstance(json_item["image_url"]["url"], str), debug_info
                    if item.type == Type.IMAGE_BINARY:
                        assert "image_url" in json_item
                        image_url = json_item["image_url"]
                        assert isinstance(image_url, dict)
                        assert "url" in image_url
                        expected_base64_bytes_str = (
                            base64encode_content_item_image_bytes(
                                message.image_content_items[-1], add_mime_prefix=True
                            )
                        )
                        assert len(expected_base64_bytes_str) == len(image_url["url"])
                        assert image_url == {
                            "url": expected_base64_bytes_str
                        }, debug_info
                    elif item.type == Type.IMAGE_URL:
                        assert json_item["image_url"] == {
                            "url": item.content
                        }, debug_info
                    elif item.type == Type.IMAGE_PATH:
                        assert json_item["image_url"]["url"].startswith(
                            "data:image/png;base64,"
                        ), debug_info
