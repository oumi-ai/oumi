import re
from unittest.mock import MagicMock

import pytest

import oumi.core.constants as constants
from oumi.builders import build_tokenizer
from oumi.builders.collators import (
    _NESTING_MODEL_TYPES,
    _TOOL_RESPONSE_BRACKETS,
    _known_tool_response_markers,
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
    TrainingParams,
    TrainTarget,
)
from tests.markers import requires_hf_token


def test_build_data_collator_empty_name(mock_tokenizer):
    with pytest.raises(ValueError, match="Empty data collator name"):
        build_data_collator("", mock_tokenizer, max_length=None)

    with pytest.raises(ValueError, match="Empty data collator name"):
        build_data_collator(
            "",
            mock_tokenizer,
            max_length=None,
            label_ignore_index=None,
        )

    with pytest.raises(ValueError, match="Empty data collator name"):
        build_data_collator(
            collator_name="",
            tokenizer=mock_tokenizer,
            max_length=1024,
            label_ignore_index=constants.LABEL_IGNORE_INDEX,
        )


def test_build_data_collator_unknown_name(mock_tokenizer):
    with pytest.raises(
        ValueError, match="Unknown data collator name: 'non_existent_collator00'"
    ):
        build_data_collator("non_existent_collator00", mock_tokenizer, max_length=None)

    with pytest.raises(
        ValueError, match="Unknown data collator name: 'non_existent_collator01'"
    ):
        build_data_collator(
            "non_existent_collator01",
            mock_tokenizer,
            max_length=None,
            label_ignore_index=None,
        )

    with pytest.raises(
        ValueError, match="Unknown data collator name: 'non_existent_collator02'"
    ):
        build_data_collator(
            collator_name="non_existent_collator02",
            tokenizer=mock_tokenizer,
            max_length=1024,
            label_ignore_index=None,
        )
    with pytest.raises(
        ValueError, match="Unknown data collator name: 'non_existent_collator02'"
    ):
        build_data_collator(
            collator_name="non_existent_collator02",
            tokenizer=mock_tokenizer,
            max_length=1024,
            label_ignore_index=constants.LABEL_IGNORE_INDEX,
        )


def test_build_data_collator_text_with_padding(mock_tokenizer):
    collator = build_data_collator("text_with_padding", mock_tokenizer, max_length=256)
    assert collator is not None
    assert callable(collator)

    # TODO add tests to exercise the collator


def test_build_data_collator_vision_language_with_padding(mock_tokenizer):
    collator = build_data_collator(
        "vision_language_with_padding",
        mock_tokenizer,
        max_length=64,
        label_ignore_index=None,
    )
    assert collator is not None
    assert callable(collator)

    # TODO add tests to exercise the collator


def test_build_data_collator_vision_language_sft(mock_tokenizer):
    with pytest.raises(ValueError, match=re.escape("Empty processor_name")):
        collator = build_data_collator(
            "vision_language_sft",
            mock_tokenizer,
            max_length=64,
            label_ignore_index=None,
        )

    def _convert_tokens_to_ids(token: str) -> int:
        if token == "<image>":
            return 32000
        return 101

    mock_tokenizer.convert_tokens_to_ids = MagicMock(side_effect=_convert_tokens_to_ids)

    collator = build_data_collator(
        "vision_language_sft",
        mock_tokenizer,
        max_length=1024,
        label_ignore_index=None,
        processor_name="llava-hf/llava-1.5-7b-hf",
    )
    assert collator is not None
    assert callable(collator)


@pytest.mark.parametrize("label_ignore_index", [None, -100])
def test_build_collator_from_config_with_collator(
    label_ignore_index: int | None, mock_tokenizer
):
    training_config = TrainingConfig(
        data=DataParams(
            train=DatasetSplitParams(
                collator_name="text_with_padding",
                datasets=[DatasetParams(dataset_name="dummy", split="train")],
            )
        ),
        model=ModelParams(
            model_name="MlpEncoder",
            tokenizer_name="openai-community/gpt2",
            model_max_length=64,
        ),
        training=TrainingParams(
            label_ignore_index=label_ignore_index,
        ),
    )

    collator = build_collator_from_config(training_config, tokenizer=mock_tokenizer)
    assert collator is not None
    assert callable(collator)


def test_build_collator_from_config_no_collator(mock_tokenizer):
    training_config = TrainingConfig(
        data=DataParams(
            train=DatasetSplitParams(
                collator_name=None,
                datasets=[DatasetParams(dataset_name="dummy", split="train")],
            )
        ),
        model=ModelParams(
            model_name="MlpEncoder",
            tokenizer_name="openai-community/gpt2",
            model_max_length=64,
        ),
    )

    collator = build_collator_from_config(training_config, tokenizer=mock_tokenizer)
    assert collator is None


def test_build_collator_from_config_no_collator_no_tokenizer():
    training_config = TrainingConfig(
        data=DataParams(
            train=DatasetSplitParams(
                collator_name=None,
                datasets=[DatasetParams(dataset_name="dummy", split="train")],
            )
        ),
        model=ModelParams(
            model_name="MlpEncoder",
            tokenizer_name="openai-community/gpt2",
            model_max_length=64,
        ),
    )

    collator = build_collator_from_config(training_config, tokenizer=None)
    assert collator is None


def test_build_collator_from_config_with_collator_no_tokenizer(mock_tokenizer):
    training_config = TrainingConfig(
        data=DataParams(
            train=DatasetSplitParams(
                collator_name="text_with_padding",
                datasets=[DatasetParams(dataset_name="dummy", split="train")],
            )
        ),
        model=ModelParams(
            model_name="CnnClassifier",
            tokenizer_name="openai-community/gpt2",
            model_max_length=64,
        ),
    )

    with pytest.raises(
        ValueError, match="Tokenizer must be provided if collator is specified"
    ):
        build_collator_from_config(training_config, tokenizer=None)


def test_build_collator_from_config_with_collator_kwargs(mock_tokenizer):
    training_config = TrainingConfig(
        data=DataParams(
            train=DatasetSplitParams(
                collator_name="text_with_padding",
                collator_kwargs={"max_variable_sized_dims": 10},
                datasets=[DatasetParams(dataset_name="dummy", split="train")],
            )
        ),
        model=ModelParams(
            model_name="MlpEncoder",
            tokenizer_name="openai-community/gpt2",
            model_max_length=64,
        ),
    )

    collator = build_collator_from_config(training_config, tokenizer=mock_tokenizer)
    assert collator is not None
    assert callable(collator)
    # Verify that the collator has the expected max_variable_sized_dims
    assert collator._max_variable_sized_dims == 10


def test_build_collator_from_config_collator_kwargs_override(mock_tokenizer):
    training_config = TrainingConfig(
        data=DataParams(
            train=DatasetSplitParams(
                collator_name="vision_language_with_padding",
                collator_kwargs={"allow_multi_image_inputs": False},
                datasets=[DatasetParams(dataset_name="dummy", split="train")],
            )
        ),
        model=ModelParams(
            model_name="llava-hf/llava-1.5-7b-hf",
            tokenizer_name="llava-hf/llava-1.5-7b-hf",
            model_max_length=64,
        ),
    )

    collator = build_collator_from_config(training_config, tokenizer=mock_tokenizer)
    assert collator is not None
    assert callable(collator)
    # Verify that the config kwargs override the model-determined kwargs
    assert collator._allow_multi_image_inputs is False


# ---------------------------------------------------------------------------
# Mock tokenizer factories for resolve_collator_templates error paths
# (Happy-path coverage is in integration tests with real tokenizers.)
# ---------------------------------------------------------------------------


def _unknown_tokenizer():
    """Mock tokenizer with no chat template."""
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.model_max_length = 2048
    tok.apply_chat_template = MagicMock(
        side_effect=Exception("No chat template configured")
    )
    return tok


def _non_string_template_tokenizer():
    """Mock tokenizer whose apply_chat_template returns a list instead of str."""
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.model_max_length = 2048
    tok.apply_chat_template = MagicMock(return_value=[101, 102, 103])
    return tok


def _no_sentinels_tokenizer():
    """Mock tokenizer that renders a template but drops message content."""
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.model_max_length = 2048
    tok.apply_chat_template = MagicMock(
        return_value=(
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\nhi<|im_end|>\n"
        )
    )
    return tok


def _think_only_tokenizer():
    """Mock where the assistant header is only a <think> block (no role prefix)."""
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.model_max_length = 2048

    def _apply(messages, **kw):
        last_asst_idx = max(
            i for i, m in enumerate(messages) if m["role"] == "assistant"
        )
        parts = []
        for i, m in enumerate(messages):
            if m["role"] == "assistant" and i == last_asst_idx:
                parts.append(f"<think>{m['content']}<|im_end|>\n")
            else:
                parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        return "".join(parts)

    tok.apply_chat_template = MagicMock(side_effect=_apply)

    _encode_map = {
        "<|im_end|>\n": [101, 10],
        "<|im_end|>\n<|im_start|>user\n": [101, 10, 100, 20],
        "<|im_end|>\n<think>": [101, 10, 600],
    }
    _decode_map = {
        (101, 10): "<|im_end|>\n",
        (600,): "<think>",
    }
    tok.encode = MagicMock(side_effect=lambda text, **kw: _encode_map[text])
    tok.decode = MagicMock(side_effect=lambda ids, **kw: _decode_map[tuple(ids)])
    return tok


def _empty_response_template_tokenizer():
    """Mock where header_text equals the EOT, so response_template is empty."""
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.model_max_length = 2048

    def _apply(messages, **kw):
        return "<|e|>".join(m["content"] for m in messages) + "<|e|>"

    tok.apply_chat_template = MagicMock(side_effect=_apply)

    _encode_map = {
        "<|e|>": [200],
        "<|e|><<__U__>>": [200, 300],
        "<|e|><<__A__>>": [200, 400],
    }
    _decode_map = {
        (200,): "<|e|>",
        (): " ",
    }
    tok.encode = MagicMock(side_effect=lambda text, **kw: _encode_map[text])
    tok.decode = MagicMock(side_effect=lambda ids, **kw: _decode_map[tuple(ids)])
    return tok


def _empty_eot_template_tokenizer():
    """Mock where between/after texts are empty, producing whitespace-only EOT."""
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.model_max_length = 2048

    def _apply(messages, **kw):
        parts = []
        for m in messages:
            prefix = "[A]" if m["role"] == "assistant" else ""
            parts.append(f"{prefix}{m['content']}")
        return "".join(parts)

    tok.apply_chat_template = MagicMock(side_effect=_apply)

    _encode_map = {
        "": [],
        "[A]": [500],
    }
    _decode_map = {
        (): " ",
        (500,): "[A]",
    }
    tok.encode = MagicMock(side_effect=lambda text, **kw: _encode_map[text])
    tok.decode = MagicMock(side_effect=lambda ids, **kw: _decode_map[tuple(ids)])
    return tok


@pytest.mark.parametrize(
    "make_tok,match",
    [
        (_unknown_tokenizer, "no chat template"),
        (_non_string_template_tokenizer, "non-string type"),
        (_no_sentinels_tokenizer, "Could not locate assistant turn boundaries"),
        (_think_only_tokenizer, "only a <think> block"),
        (_empty_response_template_tokenizer, "response_template is empty"),
        (_empty_eot_template_tokenizer, "end_of_turn_template is empty"),
    ],
)
def test_resolve_templates_error(make_tok, match):
    with pytest.raises(ValueError, match=match):
        resolve_collator_templates(make_tok())


# ---------------------------------------------------------------------------
# build_collator_from_config with train_target
# ---------------------------------------------------------------------------


def _completions_config(
    train_target: TrainTarget | None = None,
    collator_kwargs: dict | None = None,
    model_name: str = "MlpEncoder",
) -> TrainingConfig:
    """Config for the completions collator; model_name drives the bracket lookup."""
    return TrainingConfig(
        data=DataParams(
            train=DatasetSplitParams(
                collator_name="text_completions_only_with_padding",
                train_target=train_target,
                collator_kwargs=collator_kwargs or {},
                datasets=[DatasetParams(dataset_name="dummy", split="train")],
            )
        ),
        model=ModelParams(
            model_name=model_name,
            tokenizer_name="openai-community/gpt2",
            trust_remote_code=True,
            model_max_length=512,
        ),
    )


def test_build_data_collator_text_completions_with_tool_kwargs(mock_tokenizer):
    collator = build_data_collator(
        "text_completions_only_with_padding",
        mock_tokenizer,
        max_length=512,
        label_ignore_index=-200,
        response_template="<|assistant|>",
        end_of_turn_template="<|end|>",
        train_target="all_assistant_turns",
    )
    assert collator is not None
    assert callable(collator)
    assert collator._default_collator.ignore_index == -200


def test_train_target_on_wrong_collator():
    with pytest.raises(ValueError, match="train_target.*requires"):
        DatasetSplitParams(
            collator_name="text_with_padding",
            train_target=TrainTarget.ALL_ASSISTANT_TURNS,
            datasets=[DatasetParams(dataset_name="dummy", split="train")],
        )


def test_bare_collator_name_raises_without_templates(mock_tokenizer):
    config = _completions_config()
    with pytest.raises(ValueError, match="response_template"):
        build_collator_from_config(config, tokenizer=mock_tokenizer)


# ---------------------------------------------------------------------------
# Legacy / old-recipe backward compatibility
# ---------------------------------------------------------------------------


def test_legacy_instruction_template_backward_compat(mock_tokenizer):
    config = _completions_config(
        collator_kwargs={
            "response_template": "<|assistant|>",
            "instruction_template": "<|user|>",
        },
    )
    with pytest.warns(
        DeprecationWarning, match="Instruction-based masking is deprecated"
    ):
        collator = build_collator_from_config(config, tokenizer=mock_tokenizer)
    assert collator is not None
    inner = collator._default_collator
    assert inner.response_template == "<|assistant|>"
    assert inner.instruction_template == "<|user|>"
    assert inner.train_target == "_legacy_instruction_response"


def test_old_recipe_response_only_sets_final(mock_tokenizer):
    config = _completions_config(
        collator_kwargs={"response_template": "<|assistant|>"},
    )
    collator = build_collator_from_config(config, tokenizer=mock_tokenizer)
    assert collator is not None
    assert collator._default_collator.train_target == "final_assistant_turn"


def test_old_recipe_eot_sets_all_assistant(mock_tokenizer):
    config = _completions_config(
        collator_kwargs={
            "response_template": "<|assistant|>",
            "end_of_turn_template": "<|end|>",
        },
    )
    collator = build_collator_from_config(config, tokenizer=mock_tokenizer)
    assert collator is not None
    assert collator._default_collator.train_target == "all_assistant_turns"


# ---------------------------------------------------------------------------
# Tool-result bracket for architectures that nest tool results in assistant turns
# ---------------------------------------------------------------------------

# Verified against transformers 5.7.0. Keep in sync with _TOOL_RESPONSE_BRACKETS.
_GEMMA4_BRACKET = ("<|tool_response>", "<tool_response|>")
_GLM4_BRACKET = ("<tool_response>", "</tool_response>")

_SPAN_KWARGS = {
    "response_template": "<|assistant|>",
    "end_of_turn_template": "<|end|>",
}


@pytest.fixture(scope="module")
def real_tokenizer():
    """A real tokenizer, so bracket strings encode to comparable token IDs."""
    return build_tokenizer(
        ModelParams(
            model_name="MlpEncoder",
            torch_dtype_str="float16",
            trust_remote_code=False,
            tokenizer_name="openai-community/gpt2",
            tokenizer_pad_token="<|endoftext|>",
        )
    )


#
# _known_tool_response_markers: pure lookup on model_type
#


@pytest.mark.parametrize(
    "model_type,bracket",
    [
        pytest.param("gemma4", _GEMMA4_BRACKET, id="gemma4"),
        pytest.param("gemma4_unified", _GEMMA4_BRACKET, id="gemma4_unified-12b"),
        pytest.param("glm4_moe", _GLM4_BRACKET, id="glm4_moe"),
        pytest.param("glm4v_moe", _GLM4_BRACKET, id="glm4v_moe"),
    ],
)
def test_nesting_architectures_map_to_their_bracket(model_type, bracket):
    assert _known_tool_response_markers(model_type) == bracket


@pytest.mark.parametrize(
    "model_type",
    [
        pytest.param("qwen3", id="qwen3"),
        pytest.param("llama", id="llama"),
        pytest.param("gemma3", id="gemma3-different-generation"),
        pytest.param("chatglm", id="chatglm-older-glm"),
        # GLM-4.6V-Flash reports glm4v, but so does GLM-4.1V, which does not nest.
        # Leaving out glm4v is deliberate.
        pytest.param("glm4v", id="glm4v-ambiguous"),
        pytest.param(None, id="model-type-unknown"),
        pytest.param("", id="model-type-empty"),
    ],
)
def test_non_nesting_architectures_map_to_no_bracket(model_type):
    assert _known_tool_response_markers(model_type) is None


def test_registry_families_all_have_brackets():
    assert set(_NESTING_MODEL_TYPES.values()) <= set(_TOOL_RESPONSE_BRACKETS)


#
# End-to-end through build_collator_from_config
#


@pytest.mark.parametrize(
    "model_name,bracket",
    [
        pytest.param("google/gemma-4-E2B-it", _GEMMA4_BRACKET, id="gemma-4"),
        pytest.param("zai-org/GLM-4.5", _GLM4_BRACKET, id="glm-4.5"),
    ],
)
def test_known_architecture_resolves_bracket(model_name, bracket, real_tokenizer):
    collator = build_collator_from_config(
        _completions_config(collator_kwargs=dict(_SPAN_KWARGS), model_name=model_name),
        tokenizer=real_tokenizer,
    )
    assert collator is not None
    inner = collator._default_collator

    opener, closer = bracket
    assert inner.tool_response_token_ids == real_tokenizer.encode(
        opener, add_special_tokens=False
    )
    assert inner.end_of_tool_response_token_ids == real_tokenizer.encode(
        closer, add_special_tokens=False
    )


@pytest.mark.parametrize(
    "model_name",
    [
        pytest.param("Qwen/Qwen3-0.6B", id="qwen3"),
        # Gated repo: resolving the architecture needs an authenticated config fetch.
        pytest.param("google/gemma-3-4b-it", id="gemma-3", marks=requires_hf_token()),
        pytest.param("MlpEncoder", id="custom-oumi-model"),
    ],
)
def test_unknown_architecture_resolves_no_bracket(model_name, real_tokenizer):
    """Unlisted architectures are untouched — the fallback path must stay inert."""
    collator = build_collator_from_config(
        _completions_config(collator_kwargs=dict(_SPAN_KWARGS), model_name=model_name),
        tokenizer=real_tokenizer,
    )
    assert collator is not None
    assert collator._default_collator.tool_response_token_ids is None


def test_supplied_bracket_overrides_known_architecture(real_tokenizer):
    """A hand-configured bracket wins over the built-in table."""
    config = _completions_config(
        model_name="google/gemma-4-E2B-it",
        collator_kwargs={
            **_SPAN_KWARGS,
            "tool_response_template": "<custom_open>",
            "end_of_tool_response_template": "<custom_close>",
        },
    )

    collator = build_collator_from_config(config, tokenizer=real_tokenizer)
    assert collator is not None
    assert collator._default_collator.tool_response_token_ids == real_tokenizer.encode(
        "<custom_open>", add_special_tokens=False
    )


def test_non_span_train_target_resolves_no_bracket(real_tokenizer):
    """Only span masking can unmask a tool result, so legacy mode needs no bracket."""
    config = _completions_config(
        model_name="google/gemma-4-E2B-it",
        collator_kwargs={
            "response_template": "<|assistant|>",
            "instruction_template": "<|user|>",
        },
    )

    with pytest.deprecated_call():
        collator = build_collator_from_config(config, tokenizer=real_tokenizer)

    assert collator is not None
    assert collator._default_collator.train_target == "_legacy_instruction_response"
    assert collator._default_collator.tool_response_token_ids is None


def test_legacy_train_target_still_accepts_a_hand_supplied_bracket(real_tokenizer):
    """A hand-supplied bracket reaches the legacy collator, which never applies it.

    ``_resolve_tool_bracket`` returns early for a user-supplied pair, so the
    "only span masking can unmask a tool result" guard below it never runs. The
    markers are tokenized and stored; ``torch_call``'s legacy branch ignores them.
    Supplying one marker raises, supplying both is silently inert.
    """
    config = _completions_config(
        model_name="google/gemma-4-E2B-it",
        collator_kwargs={
            "response_template": "<|assistant|>",
            "instruction_template": "<|user|>",
            "tool_response_template": "<custom_open>",
            "end_of_tool_response_template": "<custom_close>",
        },
    )

    with pytest.deprecated_call():
        collator = build_collator_from_config(config, tokenizer=real_tokenizer)

    assert collator is not None
    inner = collator._default_collator
    assert inner.train_target == "_legacy_instruction_response"
    assert inner.tool_response_token_ids == real_tokenizer.encode(
        "<custom_open>", add_special_tokens=False
    )
    assert inner.end_of_tool_response_token_ids == real_tokenizer.encode(
        "<custom_close>", add_special_tokens=False
    )


@pytest.mark.parametrize(
    "supplied,missing",
    [
        ("tool_response_template", "end_of_tool_response_template"),
        ("end_of_tool_response_template", "tool_response_template"),
    ],
)
def test_build_collator_half_supplied_tool_bracket_raises(
    supplied, missing, mock_tokenizer
):
    """One marker alone masks nothing, so it must be reported rather than ignored."""
    config = _completions_config(
        model_name="Qwen/Qwen3-0.6B",
        collator_kwargs={**_SPAN_KWARGS, supplied: "<|tres|>"},
    )

    with pytest.raises(ValueError, match=f"without '{missing}'"):
        build_collator_from_config(config, tokenizer=mock_tokenizer)
