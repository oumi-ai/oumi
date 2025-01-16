import base64
import copy
import functools
import random
import string
from typing import Final, NamedTuple, Optional

import pytest

from oumi.builders.models import build_chat_template, build_tokenizer
from oumi.core.configs import ModelParams
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer
from oumi.core.types.conversation import (
    ContentItem,
    Conversation,
    Message,
    Role,
    Type,
)
from oumi.utils.io_utils import get_oumi_root_directory
from oumi.utils.logging import logger


class ChatTemplateTestSpec(NamedTuple):
    chat_template_name: str
    model_name: str
    test_image: bool = False
    trust_remote_code: bool = False
    image_placeholder: Optional[str] = None


class ConversationTuple(NamedTuple):
    convo: Conversation
    unique_text_pieces: list[str]


_ALL_TEST_CHARS: Final[str] = string.ascii_uppercase + string.digits


_SMALL_B64_IMAGE: Final[str] = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _create_test_image_bytes() -> bytes:
    return base64.b64decode(_SMALL_B64_IMAGE)


def _generate_unique_text_piece(idx: int) -> str:
    return f"x{idx:03}" + "".join(random.choices(_ALL_TEST_CHARS, k=8))


def create_test_conversation(
    num_messages: int, include_image: bool
) -> ConversationTuple:
    messages = []
    unique_text_pieces = []
    if include_image:
        png_bytes = _create_test_image_bytes()
        s = _generate_unique_text_piece(len(unique_text_pieces))
        messages.append(
            Message(
                role=Role.USER,
                content=[
                    ContentItem(binary=png_bytes, type=Type.IMAGE_BINARY),
                    ContentItem(content=s, type=Type.TEXT),
                ],
            )
        )
        unique_text_pieces.append(s)

    for i in range(num_messages - 1 if include_image else num_messages):
        idx = len(unique_text_pieces)
        s = _generate_unique_text_piece(idx)
        messages.append(
            Message(
                role=(Role.USER if (idx % 2 == 0) else Role.ASSISTANT),
                content=s,
            )
        )
        unique_text_pieces.append(s)
    return ConversationTuple(
        convo=Conversation(messages=messages), unique_text_pieces=unique_text_pieces
    )


@functools.cache
def create_test_tokenizer(
    model_name: str, chat_template_name: str, trust_remote_code: bool
) -> BaseTokenizer:
    return build_tokenizer(
        model_params=ModelParams(
            model_name=model_name,
            chat_template=chat_template_name,
            trust_remote_code=trust_remote_code,
        ),
    )


_ALL_CHAT_TEMPLATE_TESTS: Final[list[ChatTemplateTestSpec]] = [
    ChatTemplateTestSpec(
        chat_template_name="chat_ml", model_name="openai-community/gpt2"
    ),
    ChatTemplateTestSpec(
        chat_template_name="default", model_name="openai-community/gpt2"
    ),
    ChatTemplateTestSpec(chat_template_name="gpt2", model_name="openai-community/gpt2"),
    ChatTemplateTestSpec(
        chat_template_name="llama3-instruct",
        model_name="openai-community/gpt2",
        test_image=True,
        image_placeholder="<|image|>",
    ),
    ChatTemplateTestSpec(
        chat_template_name="llava",
        model_name="llava-hf/llava-1.5-7b-hf",
        test_image=True,
        image_placeholder="<image>",
    ),
    ChatTemplateTestSpec(
        chat_template_name="phi3-instruct",
        model_name="microsoft/Phi-3-vision-128k-instruct",
        test_image=True,
        trust_remote_code=True,
        image_placeholder="<|image_1|>",
    ),
    ChatTemplateTestSpec(
        chat_template_name="qwen2-vl-instruct",
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        test_image=True,
        image_placeholder="<|vision_start|><|image_pad|><|vision_end|>",
    ),
    ChatTemplateTestSpec(
        chat_template_name="zephyr",
        model_name="openai-community/gpt2",
    ),
]


def _generate_all_test_specs() -> list[ChatTemplateTestSpec]:
    result = copy.deepcopy(_ALL_CHAT_TEMPLATE_TESTS)

    # Backfill with templates for which there is no explicit test defined yet.
    known_template_names = {t.chat_template_name for t in result}
    chat_template_dir = get_oumi_root_directory() / "datasets" / "chat_templates"
    for f in chat_template_dir.glob("*.jinja"):
        template_name = f.stem
        if template_name in known_template_names:
            continue
        logger.warning(
            f"No explicit chat template test is configured for '{f}' yet! "
            "Consider adding a new entry to _ALL_CHAT_TEMPLATE_TESTS."
        )
        result.append(
            ChatTemplateTestSpec(
                chat_template_name=template_name, model_name="openai-community/gpt2"
            )
        )
    return result


@pytest.mark.parametrize(
    "test_spec",
    _generate_all_test_specs(),
)
def test_chat_template(test_spec: ChatTemplateTestSpec):
    random.seed(hash(test_spec))

    chat_template: str = build_chat_template(test_spec.chat_template_name)
    tokenizer = create_test_tokenizer(
        test_spec.model_name,
        test_spec.chat_template_name,
        trust_remote_code=test_spec.trust_remote_code,
    )
    tokenizer.chat_template = chat_template

    for include_image in (False, True) if test_spec.test_image else (False,):
        test_convo_tuple: ConversationTuple = create_test_conversation(
            5, include_image=include_image
        )
        for add_generation_prompt in (False, True):
            debug_tag = (
                f"\ninclude_image: {include_image} "
                f"\nadd_generation_prompt: {add_generation_prompt} "
                f"\ntest_spec: {test_spec}"
            )

            prompt = tokenizer.apply_chat_template(
                test_convo_tuple.convo,  # type: ignore
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

            logger.info(
                f"prompt ({test_spec.chat_template_name}):\n=====\n{prompt}\n====="
            )
            for text_piece in test_convo_tuple.unique_text_pieces:
                assert (
                    text_piece in prompt
                ), f"Text piece '{text_piece}' not found in '{prompt}' ({debug_tag})"

                if include_image and test_spec.image_placeholder:
                    assert test_spec.image_placeholder in prompt, (
                        f"Image tag {test_spec.image_placeholder} "
                        f"not found in '{prompt}' ({debug_tag})"
                    )

            # Same test but using JSON dict.
            convo_dict = test_convo_tuple.convo.to_dict()
            prompt = tokenizer.apply_chat_template(
                convo_dict["messages"],  # type: ignore
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

            for text_piece in test_convo_tuple.unique_text_pieces:
                assert (
                    text_piece in prompt
                ), f"Text piece '{text_piece}' not found in '{prompt}' ({debug_tag})"

                if include_image and test_spec.image_placeholder:
                    assert test_spec.image_placeholder in prompt, (
                        f"Image tag {test_spec.image_placeholder} "
                        f"not found in '{prompt}' ({debug_tag})"
                    )
