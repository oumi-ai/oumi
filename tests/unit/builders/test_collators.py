import re
from unittest.mock import MagicMock

import pytest

import oumi.core.constants as constants
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
    TrainingParams,
    TrainTarget,
)


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
) -> TrainingConfig:
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
            model_name="MlpEncoder",
            tokenizer_name="openai-community/gpt2",
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
