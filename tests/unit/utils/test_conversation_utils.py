import copy
import tempfile
from pathlib import Path
from typing import Final, Optional

import PIL.Image
import pytest
import responses

from oumi.builders import build_tokenizer
from oumi.core.configs import ModelParams
from oumi.core.tokenizers import BaseTokenizer
from oumi.core.types.conversation import ContentItem, Conversation, Message, Role, Type
from oumi.utils.conversation_utils import (
    base64encode_content_item_image_bytes,
    convert_message_to_json_content,
    convert_message_to_json_content_list,
    create_list_of_message_json_dicts,
    load_image_bytes_to_content_item,
    load_pil_image_from_content_item,
    remove_excessive_images_from_conversation,
    truncate_text_in_content_items,
)
from oumi.utils.image_utils import (
    create_png_bytes_from_image,
)
from tests import get_testdata_dir

_TEST_IMAGE_DIR: Final[Path] = get_testdata_dir() / "images"


@pytest.fixture
def gpt2_tokenizer():
    tokenizer = build_tokenizer(
        ModelParams(
            model_name="openai-community/gpt2",
            torch_dtype_str="float16",
            trust_remote_code=False,
            chat_template="default",
            tokenizer_pad_token="<|endoftext|>",
        )
    )
    assert tokenizer.pad_token_id is not None
    assert isinstance(tokenizer.pad_token_id, int)
    return tokenizer


def create_test_text_only_conversation():
    return Conversation(
        conversation_id="text_convo",
        messages=[
            Message(content="You are an assistant!", role=Role.SYSTEM),
            Message(content="Hello", role=Role.USER),
            Message(content="Hi there!", role=Role.ASSISTANT),
            Message(content="How are you?", role=Role.USER),
        ],
        metadata={"foo": "bar_text"},
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
        conversation_id="mm_convo",
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
        ],
        metadata={"foo": "bar_mm"},
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
                        assert image_url == {"url": expected_base64_bytes_str}, (
                            debug_info
                        )
                    elif item.type == Type.IMAGE_URL:
                        assert json_item["image_url"] == {"url": item.content}, (
                            debug_info
                        )
                    elif item.type == Type.IMAGE_PATH:
                        assert json_item["image_url"]["url"].startswith(
                            "data:image/png;base64,"
                        ), debug_info


def test_load_pil_image_from_content_item():
    test_pil_image = PIL.Image.new(mode="RGB", size=(32, 48))
    test_png_bytes = create_png_bytes_from_image(test_pil_image)

    pil_image = load_pil_image_from_content_item(
        ContentItem(type=Type.IMAGE_BINARY, binary=test_png_bytes)
    )
    assert pil_image.mode == "RGB"
    assert pil_image.size == test_pil_image.size

    pil_image = load_pil_image_from_content_item(
        ContentItem(type=Type.IMAGE_BINARY, binary=test_png_bytes), mode="RGBA"
    )
    assert pil_image.mode == "RGBA"
    assert pil_image.size == test_pil_image.size

    with tempfile.TemporaryDirectory() as output_temp_dir:
        png_filename: Path = Path(output_temp_dir) / "test.png"
        with png_filename.open(mode="wb") as f:
            f.write(test_png_bytes)

        pil_image = load_pil_image_from_content_item(
            ContentItem(type=Type.IMAGE_PATH, content=str(png_filename))
        )
        assert pil_image.mode == "RGB"
        assert pil_image.size == test_pil_image.size

        pil_image = load_pil_image_from_content_item(
            ContentItem(type=Type.IMAGE_PATH, content=str(png_filename)), mode=""
        )
        assert pil_image.mode == "RGB"
        assert pil_image.size == test_pil_image.size

        pil_image = load_pil_image_from_content_item(
            ContentItem(type=Type.IMAGE_PATH, content=str(png_filename)), mode="RGBA"
        )
        assert pil_image.mode == "RGBA"
        assert pil_image.size == test_pil_image.size

    with responses.RequestsMock() as m:
        m.add(
            responses.GET, "http://oumi.ai/logo.png", body=test_png_bytes, stream=True
        )

        pil_image = load_pil_image_from_content_item(
            ContentItem(type=Type.IMAGE_URL, content="http://oumi.ai/logo.png")
        )
        assert pil_image.mode == "RGB"
        assert pil_image.size == test_pil_image.size

        pil_image = load_pil_image_from_content_item(
            ContentItem(type=Type.IMAGE_URL, content="http://oumi.ai/logo.png"), mode=""
        )
        assert pil_image.mode == "RGB"
        assert pil_image.size == test_pil_image.size

        pil_image = load_pil_image_from_content_item(
            ContentItem(type=Type.IMAGE_URL, content="http://oumi.ai/logo.png"),
            mode="RGBA",
        )
        assert pil_image.mode == "RGBA"
        assert pil_image.size == test_pil_image.size


def test_remove_excessive_images_from_conversation():
    png_bytes = create_test_png_image_bytes()

    input = create_test_multimodal_text_image_conversation()
    assert len(input.messages) == 4

    output = remove_excessive_images_from_conversation(input, max_images=-1)
    assert output == input
    output = remove_excessive_images_from_conversation(input, max_images=100)
    assert output == input
    output = remove_excessive_images_from_conversation(input, max_images=3)
    assert output == input
    output = remove_excessive_images_from_conversation(input, max_images=2)
    assert output == Conversation(
        conversation_id="mm_convo",
        metadata={"foo": "bar_mm"},
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
            Message(role=Role.USER, content="Describe this image"),
        ],
    )

    output = remove_excessive_images_from_conversation(input, max_images=1)
    assert output == Conversation(
        conversation_id="mm_convo",
        metadata={"foo": "bar_mm"},
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
            Message(role=Role.ASSISTANT, content="Greetings!"),
            Message(role=Role.USER, content="Describe this image"),
        ],
    )

    output = remove_excessive_images_from_conversation(input, max_images=0)
    assert output == Conversation(
        conversation_id="mm_convo",
        metadata={"foo": "bar_mm"},
        messages=[
            Message(content="You are an assistant!", role=Role.SYSTEM),
            Message(
                role=Role.USER,
                content=[
                    ContentItem(content="Hello", type=Type.TEXT),
                    ContentItem(content="there", type=Type.TEXT),
                ],
            ),
            Message(
                role=Role.ASSISTANT,
                content="Greetings!",
            ),
            Message(role=Role.USER, content="Describe this image"),
        ],
    )

    input = create_test_text_only_conversation()
    assert len(input.messages) == 4
    output = remove_excessive_images_from_conversation(input, max_images=-1)
    assert output == input
    output = remove_excessive_images_from_conversation(input, max_images=100)
    assert output == input
    output = remove_excessive_images_from_conversation(input, max_images=3)
    assert output == input
    output = remove_excessive_images_from_conversation(input, max_images=1)
    assert output == input
    output = remove_excessive_images_from_conversation(input, max_images=0)
    assert output == input


@pytest.mark.parametrize(
    "messages,max_tokens,truncation_side,expected_messages",
    [
        (
            [
                Message(role=Role.USER, content="Hello"),
                Message(role=Role.USER, content="Wonderful World!"),
            ],
            100,
            "right",
            None,
        ),
        (
            [
                Message(
                    role=Role.USER,
                    content=[ContentItem(type=Type.TEXT, content="Hello")],
                ),
                Message(role=Role.USER, content="Wonderful World!"),
            ],
            100,
            "right",
            None,
        ),
        (
            [
                Message(role=Role.USER, content="Hello"),
                Message(role=Role.USER, content="Wonderful World!"),
            ],
            2,
            "right",
            [
                Message(role=Role.USER, content="Hello"),
                Message(role=Role.USER, content="Wonder"),
            ],
        ),
        (
            [
                Message(role=Role.USER, content="Hello"),
                Message(
                    role=Role.USER,
                    content=[ContentItem(type=Type.TEXT, content="Wonderful World!")],
                ),
            ],
            2,
            "right",
            [
                Message(role=Role.USER, content="Hello"),
                Message(
                    role=Role.USER,
                    content=[ContentItem(type=Type.TEXT, content="Wonder")],
                ),
            ],
        ),
        (
            [
                Message(role=Role.USER, content="Hello"),
                Message(
                    role=Role.USER,
                    content=[
                        ContentItem(
                            type=Type.IMAGE_URL, content="https://oumi.ai/z.png"
                        ),
                        ContentItem(type=Type.TEXT, content="Wonderful World!"),
                        ContentItem(type=Type.IMAGE_PATH, content="/a/b.png"),
                    ],
                ),
            ],
            2,
            "right",
            [
                Message(role=Role.USER, content="Hello"),
                Message(
                    role=Role.USER,
                    content=[
                        ContentItem(
                            type=Type.IMAGE_URL, content="https://oumi.ai/z.png"
                        ),
                        ContentItem(type=Type.TEXT, content="Wonder"),
                        ContentItem(type=Type.IMAGE_PATH, content="/a/b.png"),
                    ],
                ),
            ],
        ),
        (
            [
                Message(role=Role.USER, content="Hello"),
                Message(role=Role.USER, content="Wonderful World!"),
            ],
            2,
            "left",
            [
                Message(role=Role.USER, content=""),
                Message(role=Role.USER, content=" World!"),
            ],
        ),
        (
            [
                Message(
                    role=Role.USER,
                    content=[
                        ContentItem(type=Type.TEXT, content=""),
                        ContentItem(
                            type=Type.IMAGE_URL, content="https://oumi.ai/x.png"
                        ),
                        ContentItem(type=Type.TEXT, content="Hello"),
                        ContentItem(type=Type.IMAGE_PATH, content="/a/x.png"),
                        ContentItem(type=Type.TEXT, content=""),
                    ],
                ),
                Message(
                    role=Role.USER,
                    content=[
                        ContentItem(
                            type=Type.IMAGE_URL, content="https://oumi.ai/z.png"
                        ),
                        ContentItem(type=Type.TEXT, content="Wonderful World!"),
                        ContentItem(type=Type.IMAGE_PATH, content="/a/b.png"),
                    ],
                ),
            ],
            2,
            "left",
            [
                Message(
                    role=Role.USER,
                    content=[
                        ContentItem(type=Type.TEXT, content=""),
                        ContentItem(
                            type=Type.IMAGE_URL, content="https://oumi.ai/x.png"
                        ),
                        ContentItem(type=Type.TEXT, content=""),
                        ContentItem(type=Type.IMAGE_PATH, content="/a/x.png"),
                        ContentItem(type=Type.TEXT, content=""),
                    ],
                ),
                Message(
                    role=Role.USER,
                    content=[
                        ContentItem(
                            type=Type.IMAGE_URL, content="https://oumi.ai/z.png"
                        ),
                        ContentItem(type=Type.TEXT, content=" World!"),
                        ContentItem(type=Type.IMAGE_PATH, content="/a/b.png"),
                    ],
                ),
            ],
        ),
    ],
)
def test_truncate_text_in_content_items(
    messages: list[Message],
    max_tokens: int,
    truncation_side: str,
    expected_messages: Optional[list[Message]],
    gpt2_tokenizer: BaseTokenizer,
):
    truncated_messages = truncate_text_in_content_items(
        messages,
        tokenizer=gpt2_tokenizer,
        max_tokens=max_tokens,
        truncation_side=truncation_side,
    )
    if expected_messages is not None:
        assert truncated_messages == expected_messages
    else:
        assert truncated_messages == messages
