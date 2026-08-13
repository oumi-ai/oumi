# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
from unittest.mock import patch

import pytest
import transformers

from oumi.builders.collators import (
    build_collator_from_config,
    build_data_collator,
    resolve_collator_templates,
)
from oumi.core.configs import (
    DataParams,
    DatasetParams,
    DatasetSplitParams,
    ModelParams,
    TrainingConfig,
    TrainTarget,
)
from oumi.core.constants import LABEL_IGNORE_INDEX
from oumi.core.tokenizers.utils import (
    _probe_renders_sentinel,
    detect_chat_template_tool_format,
)
from oumi.core.types import Conversation
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
from oumi.utils.conversation_utils import create_chat_template_inputs
from oumi.utils.packaging import is_transformers_v5
from tests.markers import requires_hf_token


def _normalize(s: str) -> str:
    """Normalize sentencepiece ▁ (U+2581) to space for cross-version comparison."""
    return s.replace("\u2581", " ")


@functools.cache
def _load_tokenizer(
    model_name: str, trust_remote_code: bool = False
) -> transformers.PreTrainedTokenizerBase:
    return transformers.AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )


# -- Public models (no HF token required) ------------------------------------


@pytest.mark.parametrize(
    "model_name,trust_remote_code,expected_response,expected_eot",
    [
        pytest.param(
            "Qwen/Qwen2.5-0.5B-Instruct",
            False,
            "<|im_start|>assistant",
            "<|im_end|>\n",
            id="qwen2.5-chatml",
        ),
        pytest.param(
            "Qwen/Qwen3-0.6B",
            False,
            "<|im_start|>assistant",
            "<|im_end|>\n",
            id="qwen3-chatml-think",
        ),
        pytest.param(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            False,
            "<｜Assistant｜>",
            "<｜end▁of▁sentence｜>",
            id="deepseek-r1-qwen",
        ),
        pytest.param(
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            False,
            "<｜Assistant｜>",
            "<｜end▁of▁sentence｜>",
            id="deepseek-r1-llama",
        ),
        pytest.param(
            "allenai/Olmo-3-7B-Instruct",
            True,
            "\n<|im_start|>assistant",
            "<|im_end|>",
            id="olmo3",
        ),
        pytest.param(
            "HuggingFaceTB/SmolLM2-135M-Instruct",
            False,
            "<|im_start|>assistant",
            "<|im_end|>\n",
            id="smollm2",
        ),
        pytest.param(
            "HuggingFaceTB/SmolLM3-3B",
            False,
            "<|im_start|>assistant",
            "<|im_end|>\n",
            id="smollm3",
        ),
        pytest.param(
            "Qwen/Qwen3.5-0.8B",
            True,
            "<|im_start|>assistant",
            "<|im_end|>\n",
            id="qwen3.5",
        ),
        pytest.param(
            "Qwen/Qwen3.6-35B-A3B",
            True,
            "<|im_start|>assistant",
            "<|im_end|>\n",
            id="qwen3.6",
        ),
        pytest.param(
            "MiniMaxAI/MiniMax-M2.5",
            True,
            "]~b]ai",
            "[e~[\n",
            id="minimax-m2.5",
        ),
    ],
)
def test_template_detection_public(
    model_name, trust_remote_code, expected_response, expected_eot
):
    tokenizer = _load_tokenizer(model_name, trust_remote_code)
    response_template, end_of_turn_template = resolve_collator_templates(tokenizer)
    assert _normalize(response_template) == _normalize(expected_response)
    assert _normalize(end_of_turn_template) == _normalize(expected_eot)
    assert response_template.strip()
    assert end_of_turn_template.strip()
    assert "<think>" not in response_template


# -- Gated models (require HF token) -----------------------------------------


@pytest.mark.parametrize(
    "model_name,trust_remote_code,expected_response,expected_eot",
    [
        pytest.param(
            "meta-llama/Llama-3.2-1B-Instruct",
            False,
            "<|start_header_id|>assistant<|end_header_id|>",
            "<|eot_id|>",
            id="llama3",
        ),
        pytest.param(
            "google/gemma-3-4b-it",
            False,
            "<start_of_turn>model",
            "<end_of_turn>\n",
            id="gemma3",
        ),
        pytest.param(
            "microsoft/Phi-4-reasoning-plus",
            True,
            "<|im_start|>assistant<|im_sep|>",
            "<|im_end|>",
            id="phi4-reasoning",
        ),
        pytest.param(
            "openai/gpt-oss-20b",
            True,
            "<|start|>assistant<|channel|>final<|message|>",
            "<|end|>",
            id="gpt-oss",
        ),
        pytest.param(
            "mistralai/Mistral-7B-Instruct-v0.3",
            False,
            "[/INST] ",
            "</s>",
            id="mistral",
        ),
        pytest.param(
            "Qwen/Qwen3-Next-80B-A3B-Instruct",
            True,
            "<|im_start|>assistant",
            "<|im_end|>\n",
            id="qwen3-next",
        ),
    ],
)
@requires_hf_token()
def test_template_detection_gated(
    model_name, trust_remote_code, expected_response, expected_eot
):
    tokenizer = _load_tokenizer(model_name, trust_remote_code)
    response_template, end_of_turn_template = resolve_collator_templates(tokenizer)
    assert _normalize(response_template) == _normalize(expected_response)
    assert _normalize(end_of_turn_template) == _normalize(expected_eot)
    assert response_template.strip()
    assert end_of_turn_template.strip()
    assert "<think>" not in response_template


# -- Models requiring newer transformers --------------------------------------
# These tokenizers need transformers >= 5.3.
# Tests skip gracefully if the tokenizer cannot be loaded.


@pytest.mark.parametrize(
    "model_name,trust_remote_code,expected_response,expected_eot",
    [
        pytest.param(
            "google/gemma-4-E2B-it",
            True,
            "<|turn>model",
            "<turn|>\n",
            id="gemma4",
        ),
        pytest.param(
            "mistralai/Mistral-Small-4-119B-2603",
            True,
            "[/INST]",
            "</s>",
            id="mistral-small-4",
        ),
    ],
)
@requires_hf_token()
def test_template_detection_newer_transformers(
    model_name, trust_remote_code, expected_response, expected_eot
):
    try:
        tokenizer = _load_tokenizer(model_name, trust_remote_code)
    except (AttributeError, ValueError, KeyError) as e:
        pytest.skip(
            f"Tokenizer for {model_name} not loadable with "
            f"transformers {transformers.__version__}: {e}"
        )
    response_template, end_of_turn_template = resolve_collator_templates(tokenizer)
    assert _normalize(response_template) == _normalize(expected_response)
    assert _normalize(end_of_turn_template) == _normalize(expected_eot)
    assert response_template.strip()
    assert end_of_turn_template.strip()
    assert "<think>" not in response_template


# -- Tool results nested inside assistant turns --------------------------------
# gemma-4 and GLM-4.5 render tool results inside the model turn, so span masking
# trains on environment output unless the tool-result bracket is subtracted.


def _template_inputs(conversation: Conversation):
    """Messages/tools shaped for templates that require mapping arguments."""
    return create_chat_template_inputs(
        conversation, tool_arguments_format="mapping", content_format="empty"
    )


def _build_collator(tokenizer, model_name: str):
    """The collator the trainer would build for this model, bracket and all."""
    config = TrainingConfig(
        data=DataParams(
            train=DatasetSplitParams(
                collator_name="text_completions_only_with_padding",
                train_target=TrainTarget.ALL_ASSISTANT_TURNS,
                datasets=[DatasetParams(dataset_name="dummy", split="train")],
            )
        ),
        # The bracket is keyed on the model's config.model_type, so model_name has to
        # be the real checkpoint rather than a stand-in.
        model=ModelParams(
            model_name=model_name,
            trust_remote_code=True,
            model_max_length=8192,
        ),
    )
    collator = build_collator_from_config(config, tokenizer=tokenizer)
    assert collator is not None
    return collator


def _encode_conversation(tokenizer, conversation: Conversation) -> list[int]:
    messages, tools = _template_inputs(conversation)
    encoded = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        return_dict=True,
        add_generation_prompt=False,
    )
    return list(encoded["input_ids"])


def _labels(collator, input_ids: list[int]) -> list[int]:
    batch = collator([{"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}])
    return batch["labels"][0].tolist()


def _masked_bracket_regions_are_complete(collator, input_ids, labels) -> bool:
    """Every token from a tool-result opener through its closer must be masked.

    The payload sentinels only cover the tool result's own text; this also covers the
    bracket tokens and whatever the template renders between them. Returns True when
    the model uses no bracket.
    """
    opener = collator._default_collator.tool_response_token_ids
    closer = collator._default_collator.end_of_tool_response_token_ids
    if not opener or not closer:
        return True

    # Slicing past the end yields a short list, which never equals the marker, so no
    # bounds arithmetic is needed here.
    opener_starts = [
        i for i in range(len(input_ids)) if input_ids[i : i + len(opener)] == opener
    ]
    closer_ends = [
        i + len(closer)
        for i in range(len(input_ids))
        if input_ids[i : i + len(closer)] == closer
    ]
    assert opener_starts, "bracket was resolved but never occurs in the conversation"

    for start in opener_starts:
        # The block ends at the first closer after this opener. An opener with no
        # closer runs to the end of the sequence, which is what the collator masks.
        end = next((e for e in closer_ends if e > start), len(input_ids))
        if any(label != LABEL_IGNORE_INDEX for label in labels[start:end]):
            return False
    return True


def _sentinel_labels(tokenizer, input_ids, labels, text: str) -> list[int]:
    """The labels the collator gave the tokens spelling `text`.

    Compares token ids rather than decoded text: a payload that is only partly masked
    still decodes to something, so a substring check would pass on it.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    for start in range(len(input_ids)):
        if input_ids[start : start + len(tokens)] == tokens:
            return labels[start : start + len(tokens)]
    raise AssertionError(f"sentinel {text!r} is not in the tokenized conversation")


def _assert_excluded_from_loss(tokenizer, input_ids, labels, text: str, model: str):
    """Every token spelling `text` must be masked."""
    sentinel = _sentinel_labels(tokenizer, input_ids, labels, text)
    in_loss = sum(label != LABEL_IGNORE_INDEX for label in sentinel)
    assert not in_loss, (
        f"{model} trains on {text!r}: {in_loss} of its {len(sentinel)} tokens are "
        "in the loss"
    )


def _assert_included_in_loss(tokenizer, input_ids, labels, text: str, model: str):
    """Every token spelling `text` must contribute to the loss."""
    sentinel = _sentinel_labels(tokenizer, input_ids, labels, text)
    masked = sum(label == LABEL_IGNORE_INDEX for label in sentinel)
    assert not masked, (
        f"{model} drops {text!r} from the loss: {masked} of its {len(sentinel)} "
        "tokens are masked"
    )


@pytest.mark.parametrize(
    "model_name,trust_remote_code",
    [
        pytest.param(
            "google/gemma-4-E2B-it",
            False,
            id="gemma-4-nested",
            marks=pytest.mark.skipif(
                not is_transformers_v5(),
                reason="gemma-4 tokenizers require transformers v5",
            ),
        ),
        pytest.param("zai-org/GLM-4.5", False, id="glm-4.5-nested"),
        pytest.param("Qwen/Qwen3-0.6B", False, id="qwen3-separate-turn"),
    ],
)
def test_tool_results_are_excluded_from_the_loss(model_name, trust_remote_code):
    tokenizer = _load_tokenizer(model_name, trust_remote_code)

    conversation = canonical_tool_conversation()
    collator = _build_collator(tokenizer, model_name)
    input_ids = _encode_conversation(tokenizer, conversation)
    labels = _labels(collator, input_ids)

    # Environment output must never be trained on.
    for tool_result in (TOOL_RESULT_1, TOOL_RESULT_2, TOOL_RESULT_3):
        _assert_excluded_from_loss(
            tokenizer, input_ids, labels, tool_result, model_name
        )

    # Prompt text is context, not a target.
    for prompt_text in (SYSTEM_TEXT, USER_TEXT_1, USER_TEXT_2):
        _assert_excluded_from_loss(
            tokenizer, input_ids, labels, prompt_text, model_name
        )

    # The model's own turns must survive, including the answer that gemma-4 renders
    # after the tool results inside the same turn.
    for assistant_text in (ASSISTANT_TEXT_1, ASSISTANT_TEXT_2):
        _assert_included_in_loss(
            tokenizer, input_ids, labels, assistant_text, model_name
        )

    # The sentinels above only cover the payload. This covers the bracket tokens and
    # anything the template puts between them.
    assert _masked_bracket_regions_are_complete(collator, input_ids, labels), (
        f"{model_name} leaves part of a bracketed tool result in the loss"
    )


def test_bracket_forced_onto_a_separate_turn_model_changes_nothing():
    """Qwen3 emits <tool_response> markers, but in a turn of their own.

    Assert that providing a bracket for template that does not nest change results.
    """
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = _load_tokenizer(model_name)

    conversation = canonical_tool_conversation()
    input_ids = _encode_conversation(tokenizer, conversation)

    response_template, end_of_turn_template = resolve_collator_templates(tokenizer)
    baseline = build_data_collator(
        "text_completions_only_with_padding",
        tokenizer=tokenizer,
        max_length=None,
        response_template=response_template,
        end_of_turn_template=end_of_turn_template,
        train_target="all_assistant_turns",
    )
    forced = build_data_collator(
        "text_completions_only_with_padding",
        tokenizer=tokenizer,
        max_length=None,
        response_template=response_template,
        end_of_turn_template=end_of_turn_template,
        train_target="all_assistant_turns",
        tool_response_template="<tool_response>",
        end_of_tool_response_template="</tool_response>",
    )

    # The markers really are present, so this is not a vacuous comparison.
    opener = forced._default_collator.tool_response_token_ids
    assert any(
        input_ids[i : i + len(opener)] == opener for i in range(len(input_ids))
    ), "expected <tool_response> to appear in the rendered conversation"

    assert _labels(forced, input_ids) == _labels(baseline, input_ids)


#
# Restoring a `type` that `to_dict()` left out.
#


def test_deepseek_renders_a_conversation_whose_tool_calls_omit_type():
    """DeepSeek reads `type` directly off each tool call and tool definition.

    `Conversation.to_dict()` dumps with `exclude_unset=True`, so a conversation
    built in code drops the key. Without the adapter restoring it, this model
    raises `UndefinedError` and the conversation cannot be rendered at all.
    """
    tokenizer = _load_tokenizer("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    conversation = canonical_tool_conversation()

    # Not a vacuous test: the key really is absent on the wire format.
    raw = conversation.to_dict()
    assert "type" not in raw["messages"][FIRST_TOOL_CALL_INDEX]["tool_calls"][0]
    assert "type" not in raw["tools"][0]

    messages, tools = create_chat_template_inputs(
        conversation, tool_arguments_format="string", content_format="null"
    )
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        tools=tools,  # type: ignore[arg-type]
    )

    assert ARGUMENT_SENTINEL in rendered


#
# Tool-format detection against the shipped templates.
#


@pytest.mark.parametrize(
    "model_name,trust_remote_code,expected_format",
    [
        pytest.param(
            "google/gemma-4-E2B-it",
            False,
            ("mapping", "empty"),
            id="gemma-4-raises-on-string-arguments",
            marks=pytest.mark.skipif(
                not is_transformers_v5(),
                reason="gemma-4 tokenizers require transformers v5",
            ),
        ),
        pytest.param(
            "zai-org/GLM-4.5",
            False,
            ("mapping", "empty"),
            id="glm-4.5-iterates-arguments-items",
        ),
        pytest.param(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            False,
            ("string", "null"),
            id="deepseek-r1-concatenates-arguments",
        ),
        pytest.param(
            "Qwen/Qwen3.5-0.8B",
            True,
            ("mapping", "empty"),
            id="qwen3.5-rejects-string-arguments",
        ),
        pytest.param(
            "Qwen/Qwen3.6-35B-A3B",
            True,
            ("mapping", "empty"),
            id="qwen3.6-rejects-string-arguments",
        ),
        pytest.param(
            "MiniMaxAI/MiniMax-M2.5",
            True,
            ("mapping", "empty"),
            id="minimax-m2.5-rejects-string-arguments",
        ),
        pytest.param(
            "openai/gpt-oss-20b",
            True,
            ("mapping", "empty"),
            id="gpt-oss-requires-non-null-content",
            marks=requires_hf_token(),
        ),
    ],
)
def test_detected_tool_format_matches_the_shipped_template(
    model_name, trust_remote_code, expected_format
):
    tokenizer = _load_tokenizer(model_name, trust_remote_code)

    assert tuple(detect_chat_template_tool_format(tokenizer)) == expected_format


@pytest.mark.parametrize(
    "model_name,arguments_form,content_form,expected_verdict",
    [
        pytest.param(
            "google/gemma-4-E2B-it",
            "string",
            "empty",
            None,
            id="gemma-4-raises-rather-than-accepting-a-string",
            marks=pytest.mark.skipif(
                not is_transformers_v5(),
                reason="gemma-4 tokenizers require transformers v5",
            ),
        ),
        pytest.param(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "mapping",
            "null",
            None,
            id="deepseek-r1-raises-rather-than-accepting-a-mapping",
        ),
        pytest.param(
            # DeepSeek renders the call only when `content` is None, so "" silently
            # drops it.
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "string",
            "empty",
            False,
            id="deepseek-r1-drops-the-tool-call-for-empty-content",
        ),
        pytest.param(
            "HuggingFaceTB/SmolLM3-3B",
            "mapping",
            "empty",
            False,
            id="smollm3-drops-the-tool-call-without-raising",
        ),
        pytest.param(
            "Qwen/Qwen2.5-0.5B-Instruct",
            "string",
            "empty",
            False,
            id="qwen2.5-double-encodes-without-raising",
        ),
        pytest.param(
            "Qwen/Qwen3.5-0.8B",
            "mapping",
            "empty",
            True,
            id="qwen3.5-renders-the-sentinel-verbatim",
        ),
    ],
)
def test_probe_verdict_for_shipped_template(
    model_name, arguments_form, content_form, expected_verdict
):
    tokenizer = _load_tokenizer(model_name)

    verdict = _probe_renders_sentinel(
        tokenizer, arguments_form=arguments_form, content_form=content_form
    )

    assert verdict is expected_verdict


def test_double_encoding_is_why_qwen2_5_rejects_string_arguments():
    """Qwen2.5 pipes `arguments` through `tojson` with no type check.

    Given the JSON string Oumi stores, it encodes the value a second time and
    the model would train on `"{\\"city\\": \\"...\\"}"` where an object belongs.
    Nothing raises and the sentinel survives verbatim, so the escape check is
    the only thing standing between this and silently corrupted data.
    """
    tokenizer = _load_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")
    messages, tools = create_chat_template_inputs(
        canonical_tool_conversation(),
        tool_arguments_format="string",  # type: ignore[arg-type]
        content_format="empty",  # type: ignore[arg-type]
    )

    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        tools=tools,  # type: ignore[arg-type]
    )

    # Not a vacuous rejection: the value is present, just escaped.
    assert ARGUMENT_SENTINEL in rendered
    assert f'\\"{ARGUMENT_SENTINEL}' in rendered
    assert (
        _probe_renders_sentinel(
            tokenizer, arguments_form="string", content_form="empty"
        )
        is False
    )


def test_template_with_no_working_form_falls_back_and_warns():
    """SmolLM3 has no branch that renders an assistant message's `tool_calls`.

    Every form is rejected, so detection cannot do better than warn and return
    the preferred default.
    """
    # A fresh instance, so the cached detection for this model does not hide the
    # warning this test is asserting on.
    tokenizer = transformers.AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")

    with patch("oumi.core.tokenizers.utils.logger") as mock_logger:
        resolved = detect_chat_template_tool_format(tokenizer)

    assert tuple(resolved) == ("mapping", "empty")
    mock_logger.warning.assert_called_once()


def test_glm_does_not_render_a_literal_none_for_null_content():
    """GLM-4.5 stringifies `content: None` into the text it trains on.

    Its template pipes content through `visible_text(m.content)` unconditionally,
    so `None` becomes the literal word. Both content forms render and the
    argument sentinel survives either way, which is why the candidate order --
    not the probe's accept/reject -- is what keeps this out of the data.
    """
    tokenizer = _load_tokenizer("zai-org/GLM-4.5")
    resolved = detect_chat_template_tool_format(tokenizer)

    messages, tools = _template_inputs(canonical_tool_conversation())
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        tools=tools,  # type: ignore[arg-type]
    )

    assert resolved.content == "empty"
    assert "\nNone\n" not in rendered
