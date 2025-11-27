import base64
import copy
import functools
import random
import string
import warnings
from typing import Final, NamedTuple, Optional

import pytest
import transformers
from packaging import version

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
from tests.markers import requires_hf_token


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
    num_messages: int, num_with_images: int = 0
) -> ConversationTuple:
    assert num_with_images >= 0 and num_with_images <= num_messages
    messages = []
    unique_text_pieces = []
    for i in range(num_with_images):
        idx = len(unique_text_pieces)
        png_bytes = _create_test_image_bytes()
        s = _generate_unique_text_piece(len(unique_text_pieces))
        messages.append(
            Message(
                role=(Role.USER if (idx % 2 == 0) else Role.ASSISTANT),
                content=[
                    ContentItem(binary=png_bytes, type=Type.IMAGE_BINARY),
                    ContentItem(content=s, type=Type.TEXT),
                ],
            )
        )
        unique_text_pieces.append(s)

    for i in range(num_messages - num_with_images):
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
    model_name: str,
    *,
    chat_template_name: Optional[str] = None,
    inline_chat_template: Optional[str] = None,
    trust_remote_code: bool = False,
) -> BaseTokenizer:
    if chat_template_name is None:
        assert inline_chat_template is not None
    else:
        assert inline_chat_template is None

    tokenizer = build_tokenizer(
        model_params=ModelParams(
            model_name=model_name,
            chat_template=chat_template_name,
            trust_remote_code=trust_remote_code,
        ),
    )
    if inline_chat_template is not None:
        tokenizer.chat_template = inline_chat_template
    return tokenizer


@functools.cache
def get_hf_chat_template(
    tokenizer_name: str, *, trust_remote_code: bool = False
) -> Optional[str]:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=trust_remote_code
    )
    if tokenizer.chat_template:
        assert isinstance(tokenizer.chat_template, str), (
            f"tokenizer_name: {tokenizer_name}"
        )
        return tokenizer.chat_template
    return None


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
    # TODO: Uncomment when we update to a newer version of transformers.
    # ChatTemplateTestSpec(
    #     chat_template_name="internvl3",
    #     model_name="OpenGVLab/InternVL3-1B-hf",
    #     test_image=True,
    #     image_placeholder="<IMG_CONTEXT>",
    # ),
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


def _get_chat_template_test_spec_id(spec):
    assert isinstance(spec, ChatTemplateTestSpec)
    return f"{spec.chat_template_name}_{spec.model_name}"


@pytest.mark.parametrize(
    "test_spec",
    _generate_all_test_specs(),
    ids=_get_chat_template_test_spec_id,
)
def test_chat_template(test_spec: ChatTemplateTestSpec):
    random.seed(hash(test_spec))

    oumi_chat_template: str = build_chat_template(test_spec.chat_template_name)
    tokenizer = create_test_tokenizer(
        test_spec.model_name,
        chat_template_name=test_spec.chat_template_name,
        trust_remote_code=test_spec.trust_remote_code,
    )
    assert tokenizer.chat_template == oumi_chat_template

    for include_image in (False, True) if test_spec.test_image else (False,):
        test_convo_tuple: ConversationTuple = create_test_conversation(
            5, num_with_images=(1 if include_image else 0)
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
                assert text_piece in prompt, (
                    f"Text piece '{text_piece}' not found in '{prompt}' ({debug_tag})"
                )

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
                assert text_piece in prompt, (
                    f"Text piece '{text_piece}' not found in '{prompt}' ({debug_tag})"
                )

                if include_image and test_spec.image_placeholder:
                    assert test_spec.image_placeholder in prompt, (
                        f"Image tag {test_spec.image_placeholder} "
                        f"not found in '{prompt}' ({debug_tag})"
                    )


@pytest.mark.parametrize(
    "model_name, is_vision",
    [
        ("microsoft/Phi-3-vision-128k-instruct", True),
        ("microsoft/Phi-3-mini-4k-instruct", False),
    ],
)
def test_phi3_chat_template(model_name: str, is_vision: bool):
    oumi_chat_template: str = build_chat_template("phi3-instruct")
    hf_chat_template = get_hf_chat_template(model_name)
    assert hf_chat_template != oumi_chat_template

    oumi_tokenizer = create_test_tokenizer(
        model_name,
        chat_template_name="phi3-instruct",
        trust_remote_code=True,
    )
    assert oumi_chat_template == oumi_tokenizer.chat_template

    hf_tokenizer = create_test_tokenizer(
        model_name,
        inline_chat_template=hf_chat_template,
        trust_remote_code=True,
    )
    assert hf_chat_template == hf_tokenizer.chat_template

    # Text only: Verify that oumi template leads to the same result as HF template.
    test_convo_tuple: ConversationTuple = create_test_conversation(3, num_with_images=0)
    for add_generation_prompt in (False, True):
        debug_tag = f"Text-only: add_generation_prompt: {add_generation_prompt} "
        oumi_result = oumi_tokenizer.apply_chat_template(
            test_convo_tuple.convo.messages,  # type: ignore
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        assert isinstance(oumi_result, str), debug_tag
        hf_result = hf_tokenizer.apply_chat_template(
            test_convo_tuple.convo.messages,  # type: ignore
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        assert isinstance(hf_result, str), debug_tag
        unique_text_pieces = test_convo_tuple.unique_text_pieces
        assert len(unique_text_pieces) == 3
        expected_lines = (
            [
                "<|user|>",
                f"{unique_text_pieces[0]}<|end|>",
                "<|assistant|>",
                f"{unique_text_pieces[1]}<|end|>",
                "<|user|>",
                f"{unique_text_pieces[2]}<|end|>",
            ]
            + (["<|assistant|>"] if add_generation_prompt else [])
            + [""]
        )
        expected = "\n".join(expected_lines)
        assert oumi_result == expected, debug_tag
        assert hf_result.startswith(expected), (
            f"{debug_tag}\n\n{hf_result}\n\n{expected}"
        )
        if not is_vision:
            # Why the difference for text-only Phi3?
            hf_result.endswith("\n<|endoftext|>")

    if is_vision:  # With images.
        test_convo_tuple: ConversationTuple = create_test_conversation(
            3, num_with_images=2
        )
        for add_generation_prompt in (False, True):
            debug_tag = f"Multi-modal: add_generation_prompt: {add_generation_prompt} "
            oumi_result = oumi_tokenizer.apply_chat_template(
                test_convo_tuple.convo.messages,  # type: ignore
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
            unique_text_pieces = test_convo_tuple.unique_text_pieces
            assert len(unique_text_pieces) == 3
            expected_lines = (
                [
                    "<|user|>",
                    f"<|image_1|>{unique_text_pieces[0]}<|end|>",
                    "<|assistant|>",
                    f"<|image_2|>{unique_text_pieces[1]}<|end|>",
                    "<|user|>",
                    f"{unique_text_pieces[2]}<|end|>",
                ]
                + (["<|assistant|>"] if add_generation_prompt else [])
                + [""]
            )
            expected = "\n".join(expected_lines)
            assert oumi_result == expected, debug_tag


def _strip_llama3_system_prefix(s: str) -> str:
    if s.startswith("<|begin_of_text|><|start_header_id|>system<|end_header_id|>"):
        pos: int = s.index("<|eot_id|>")
        pos += len("<|eot_id|>")
        return "<|begin_of_text|>" + s[pos:]
    return s


@pytest.mark.parametrize(
    "model_name, is_vision",
    [
        ("meta-llama/Llama-3.2-1B-Instruct", False),
        ("meta-llama/Llama-3.2-11B-Vision-Instruct", True),
    ],
)
@requires_hf_token()
def test_llama3_chat_template(model_name: str, is_vision: bool):
    oumi_chat_template: str = build_chat_template("llama3-instruct")
    hf_chat_template = get_hf_chat_template(model_name)
    assert hf_chat_template != oumi_chat_template

    oumi_tokenizer = create_test_tokenizer(
        model_name,
        chat_template_name="llama3-instruct",
        trust_remote_code=True,
    )
    assert oumi_chat_template == oumi_tokenizer.chat_template

    hf_tokenizer = create_test_tokenizer(
        model_name,
        inline_chat_template=hf_chat_template,
        trust_remote_code=True,
    )
    assert hf_chat_template == hf_tokenizer.chat_template

    # Text only: Verify that oumi template leads to the same result as HF template.
    test_convo_tuple: ConversationTuple = create_test_conversation(3, num_with_images=0)
    for add_generation_prompt in (False, True):
        debug_tag = (
            f"Text-only ({model_name}): add_generation_prompt: {add_generation_prompt} "
        )
        oumi_result = oumi_tokenizer.apply_chat_template(
            test_convo_tuple.convo.messages,  # type: ignore
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        assert isinstance(oumi_result, str), debug_tag
        hf_result = hf_tokenizer.apply_chat_template(
            test_convo_tuple.convo.messages,  # type: ignore
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        assert isinstance(hf_result, str), debug_tag
        hf_result = _strip_llama3_system_prefix(hf_result)
        unique_text_pieces = test_convo_tuple.unique_text_pieces
        assert len(unique_text_pieces) == 3
        expected_lines = [
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>",
            "",
            f"{unique_text_pieces[0]}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
            "",
            f"{unique_text_pieces[1]}<|eot_id|><|start_header_id|>user<|end_header_id|>",
            (
                f"\n{unique_text_pieces[2]}<|eot_id|>"
                + (
                    "<|start_header_id|>assistant<|end_header_id|>"
                    if add_generation_prompt
                    else ""
                )
            ),
        ] + (["", ""] if add_generation_prompt else [])
        expected = "\n".join(expected_lines)
        assert hf_result == expected, (
            f"{debug_tag}\nHF result:\n{hf_result}\nExpected:\n{expected}"
        )
        assert oumi_result == expected, (
            f"{debug_tag}\nOUMI result:\n{oumi_result}\nExpected:\n{expected}"
        )

    if is_vision:  # With images.
        test_convo_tuple: ConversationTuple = create_test_conversation(
            3, num_with_images=2
        )
        for add_generation_prompt in (False, True):
            debug_tag = f"Multi-modal: add_generation_prompt: {add_generation_prompt} "
            oumi_result = oumi_tokenizer.apply_chat_template(
                test_convo_tuple.convo.messages,  # type: ignore
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
            assert isinstance(oumi_result, str), debug_tag
            unique_text_pieces = test_convo_tuple.unique_text_pieces
            assert len(unique_text_pieces) == 3
            expected_lines = [
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>",
                "",
                f"<|image|>{unique_text_pieces[0]}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
                "",
                f"<|image|>{unique_text_pieces[1]}<|eot_id|><|start_header_id|>user<|end_header_id|>",
                (
                    f"\n{unique_text_pieces[2]}<|eot_id|>"
                    + (
                        "<|start_header_id|>assistant<|end_header_id|>"
                        if add_generation_prompt
                        else ""
                    )
                ),
            ] + (["", ""] if add_generation_prompt else [])
            expected = "\n".join(expected_lines)
            assert oumi_result == expected, (
                f"{debug_tag}\nOUMI result:\n{oumi_result}\nExpected:\n{expected}"
            )


@pytest.mark.parametrize(
    "model_name, is_vision",
    [
        ("Qwen/Qwen2-VL-2B-Instruct", True),
        ("Qwen/Qwen2.5-VL-3B-Instruct", True),
    ],
)
def test_qwen2_chat_template(model_name: str, is_vision: bool):
    oumi_chat_template: str = build_chat_template("qwen2-vl-instruct")
    hf_chat_template = get_hf_chat_template(model_name)
    assert hf_chat_template != oumi_chat_template

    oumi_tokenizer = create_test_tokenizer(
        model_name,
        chat_template_name="qwen2-vl-instruct",
        trust_remote_code=True,
    )
    assert oumi_chat_template == oumi_tokenizer.chat_template

    hf_tokenizer = create_test_tokenizer(
        model_name,
        inline_chat_template=hf_chat_template,
        trust_remote_code=True,
    )
    assert hf_chat_template == hf_tokenizer.chat_template

    # Text only: Verify that oumi template leads to the same result as HF template.
    test_convo_tuple: ConversationTuple = create_test_conversation(3, num_with_images=0)
    for add_generation_prompt in (False, True):
        debug_tag = f"Text-only: add_generation_prompt: {add_generation_prompt} "
        oumi_result = oumi_tokenizer.apply_chat_template(
            test_convo_tuple.convo.messages,  # type: ignore
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        assert isinstance(oumi_result, str), debug_tag
        hf_result = hf_tokenizer.apply_chat_template(
            test_convo_tuple.convo.messages,  # type: ignore
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        assert isinstance(hf_result, str), debug_tag
        unique_text_pieces = test_convo_tuple.unique_text_pieces
        assert len(unique_text_pieces) == 3
        expected_lines = (
            [
                "<|im_start|>system",
                "You are a helpful assistant.<|im_end|>",
                "<|im_start|>user",
                f"{unique_text_pieces[0]}<|im_end|>",
                "<|im_start|>assistant",
                f"{unique_text_pieces[1]}<|im_end|>",
                "<|im_start|>user",
                f"{unique_text_pieces[2]}<|im_end|>",
            ]
            + (["<|im_start|>assistant"] if add_generation_prompt else [])
            + [""]
        )
        expected = "\n".join(expected_lines)
        assert oumi_result == expected, debug_tag
        assert hf_result.startswith(expected), (
            f"{debug_tag}\n\n{hf_result}\n\n{expected}"
        )

    # With images.
    if is_vision:
        test_convo_tuple: ConversationTuple = create_test_conversation(
            3, num_with_images=2
        )
        for add_generation_prompt in (False, True):
            debug_tag = f"Multi-modal: add_generation_prompt: {add_generation_prompt} "
            oumi_result = oumi_tokenizer.apply_chat_template(
                test_convo_tuple.convo.messages,  # type: ignore
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
            unique_text_pieces = test_convo_tuple.unique_text_pieces
            assert len(unique_text_pieces) == 3
            expected_lines = (
                [
                    "<|im_start|>system",
                    "You are a helpful assistant.<|im_end|>",
                    "<|im_start|>user",
                    f"<|vision_start|><|image_pad|><|vision_end|>{unique_text_pieces[0]}<|im_end|>",
                    "<|im_start|>assistant",
                    f"<|vision_start|><|image_pad|><|vision_end|>{unique_text_pieces[1]}<|im_end|>",
                    "<|im_start|>user",
                    f"{unique_text_pieces[2]}<|im_end|>",
                ]
                + (["<|im_start|>assistant"] if add_generation_prompt else [])
                + [""]
            )
            expected = "\n".join(expected_lines)
            assert oumi_result == expected, debug_tag


# # TODO: Update to more internvl versions when we incorporate them.
# Uncomment when we have a newer version of transformers.
# @pytest.mark.parametrize(
#     "model_name, is_vision, oumi_template_name",
#     [
#         ("OpenGVLab/InternVL3-1B-hf", True, "internvl3"),
#     ],
# )
def _future_test_internvl_chat_template(
    model_name: str, is_vision: bool, oumi_template_name: str
):
    # this model *requires* >=4.52.0.dev0 version of transformers.
    if version.parse(transformers.__version__) < version.parse("4.52.0.dev0"):
        warnings.warn(
            f"Skipping test_internvl_chat_template for {model_name} "
            f"because transformers version is too old: {transformers.__version__}"
        )
        return True

    oumi_chat_template: str = build_chat_template(oumi_template_name)
    hf_chat_template = get_hf_chat_template(model_name)
    assert hf_chat_template != oumi_chat_template

    oumi_tokenizer = create_test_tokenizer(
        model_name,
        chat_template_name=oumi_template_name,
        trust_remote_code=True,
    )
    assert oumi_chat_template == oumi_tokenizer.chat_template

    hf_tokenizer = create_test_tokenizer(
        model_name,
        inline_chat_template=hf_chat_template,
        trust_remote_code=True,
    )
    assert hf_chat_template == hf_tokenizer.chat_template

    # Text only: Verify that oumi template leads to the same result as HF template.
    test_convo_tuple: ConversationTuple = create_test_conversation(3, num_with_images=0)
    for add_generation_prompt in (False, True):
        debug_tag = f"Text-only: add_generation_prompt: {add_generation_prompt} "
        oumi_result = oumi_tokenizer.apply_chat_template(
            test_convo_tuple.convo.messages,  # type: ignore
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        assert isinstance(oumi_result, str), debug_tag
        hf_result = hf_tokenizer.apply_chat_template(
            test_convo_tuple.convo.messages,  # type: ignore
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        assert isinstance(hf_result, str), debug_tag
        unique_text_pieces = test_convo_tuple.unique_text_pieces
        assert len(unique_text_pieces) == 3
        expected_lines = (
            [
                "<|im_start|>user",
                f"{unique_text_pieces[0]}<|im_end|>",
                "<|im_start|>assistant",
                f"{unique_text_pieces[1]}<|im_end|>",
                "<|im_start|>user",
                f"{unique_text_pieces[2]}<|im_end|>",
            ]
            + (["<|im_start|>assistant"] if add_generation_prompt else [])
            + [""]
        )
        expected = "\n".join(expected_lines)
        assert oumi_result == expected, debug_tag
        assert hf_result.startswith(expected), (
            f"{debug_tag}\n\n{hf_result}\n\n{expected}"
        )

    # With images.
    if is_vision:
        test_convo_tuple: ConversationTuple = create_test_conversation(
            3, num_with_images=2
        )
        for add_generation_prompt in (False, True):
            debug_tag = f"Multi-modal: add_generation_prompt: {add_generation_prompt} "
            oumi_result = oumi_tokenizer.apply_chat_template(
                test_convo_tuple.convo.messages,  # type: ignore
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
            unique_text_pieces = test_convo_tuple.unique_text_pieces
            assert len(unique_text_pieces) == 3
            expected_lines = (
                [
                    "<|im_start|>user",
                    f"<IMG_CONTEXT>\n{unique_text_pieces[0]}<|im_end|>",
                    "<|im_start|>assistant",
                    f"<IMG_CONTEXT>\n{unique_text_pieces[1]}<|im_end|>",
                    "<|im_start|>user",
                    f"{unique_text_pieces[2]}<|im_end|>",
                ]
                + (["<|im_start|>assistant"] if add_generation_prompt else [])
                + [""]
            )
            expected = "\n".join(expected_lines)
            assert oumi_result == expected, debug_tag
