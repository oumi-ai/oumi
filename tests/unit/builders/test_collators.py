import re
from unittest.mock import MagicMock

import pytest

import oumi.core.constants as constants
from oumi.builders.collators import build_collator_from_config, build_data_collator
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
# TrainTarget / builder auto-detection tests
# ---------------------------------------------------------------------------


def _chatml_tokenizer():
    """Mock tokenizer that renders ChatML format."""
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.model_max_length = 2048

    def _apply(messages, **kw):
        out = "".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
        )
        if kw.get("add_generation_prompt"):
            out += "<|im_start|>assistant\n"
        return out

    tok.apply_chat_template = MagicMock(side_effect=_apply)

    # The production code encodes/decodes three substrings from the
    # rendered template.  Map each to stable token IDs so the
    # common-prefix logic works.
    _encode_map = {
        "<|im_end|>\n": [101, 10],
        "<|im_end|>\n<|im_start|>user\n": [101, 10, 100, 20],
        "<|im_end|>\n<|im_start|>assistant\n": [101, 10, 100, 30],
        "<|im_start|>assistant\n": [100, 30],
    }
    _decode_map = {
        (101, 10): "<|im_end|>\n",
        (100, 30): "<|im_start|>assistant\n",
    }
    tok.encode = MagicMock(side_effect=lambda text, **kw: _encode_map[text])
    tok.decode = MagicMock(side_effect=lambda ids, **kw: _decode_map[tuple(ids)])
    return tok


def _llama3_tokenizer():
    """Mock tokenizer that renders Llama-3 format."""
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.model_max_length = 2048

    def _apply(messages, **kw):
        parts = ["<|begin_of_text|>"]
        for m in messages:
            parts.append(
                f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n"
                f"{m['content']}<|eot_id|>"
            )
        if kw.get("add_generation_prompt"):
            parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return "".join(parts)

    tok.apply_chat_template = MagicMock(side_effect=_apply)

    _encode_map = {
        "<|eot_id|>": [203],
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n": [
            203,
            201,
            20,
            202,
            10,
        ],
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n": [
            203,
            201,
            30,
            202,
            10,
        ],
        "<|start_header_id|>assistant<|end_header_id|>\n\n": [201, 30, 202, 10],
    }
    _decode_map = {
        (203,): "<|eot_id|>",
        (201, 30, 202, 10): "<|start_header_id|>assistant<|end_header_id|>\n\n",
    }
    tok.encode = MagicMock(side_effect=lambda text, **kw: _encode_map[text])
    tok.decode = MagicMock(side_effect=lambda ids, **kw: _decode_map[tuple(ids)])
    return tok


def _unknown_tokenizer():
    """Mock tokenizer with no chat template."""
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.model_max_length = 2048
    tok.apply_chat_template = MagicMock(
        side_effect=Exception("No chat template configured")
    )
    return tok


def test_build_data_collator_text_completions_with_tool_kwargs(mock_tokenizer):
    """Build completions collator with end_of_turn_template + custom ignore index."""
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
    inner = collator._default_collator
    assert inner.ignore_index == -200


def test_train_target_all_assistant_turns():
    """ChatML auto-detection with ALL_ASSISTANT_TURNS train target."""
    tok = _chatml_tokenizer()
    config = TrainingConfig(
        data=DataParams(
            train=DatasetSplitParams(
                collator_name="text_completions_only_with_padding",
                train_target=TrainTarget.ALL_ASSISTANT_TURNS,
                datasets=[DatasetParams(dataset_name="dummy", split="train")],
            )
        ),
        model=ModelParams(
            model_name="MlpEncoder",
            tokenizer_name="openai-community/gpt2",
            model_max_length=512,
        ),
    )
    collator = build_collator_from_config(config, tokenizer=tok)
    assert collator is not None
    inner = collator._default_collator
    assert inner.response_template == "<|im_start|>assistant\n"


def test_train_target_final_assistant_turn():
    """ChatML auto-detection with FINAL_ASSISTANT_TURN train target."""
    tok = _chatml_tokenizer()
    config = TrainingConfig(
        data=DataParams(
            train=DatasetSplitParams(
                collator_name="text_completions_only_with_padding",
                train_target=TrainTarget.FINAL_ASSISTANT_TURN,
                datasets=[DatasetParams(dataset_name="dummy", split="train")],
            )
        ),
        model=ModelParams(
            model_name="MlpEncoder",
            tokenizer_name="openai-community/gpt2",
            model_max_length=512,
        ),
    )
    collator = build_collator_from_config(config, tokenizer=tok)
    assert collator is not None
    inner = collator._default_collator
    assert inner.response_template == "<|im_start|>assistant\n"


def test_train_target_llama3():
    """Llama-3 auto-detection with ALL_ASSISTANT_TURNS train target."""
    tok = _llama3_tokenizer()
    config = TrainingConfig(
        data=DataParams(
            train=DatasetSplitParams(
                collator_name="text_completions_only_with_padding",
                train_target=TrainTarget.ALL_ASSISTANT_TURNS,
                datasets=[DatasetParams(dataset_name="dummy", split="train")],
            )
        ),
        model=ModelParams(
            model_name="MlpEncoder",
            tokenizer_name="openai-community/gpt2",
            model_max_length=512,
        ),
    )
    collator = build_collator_from_config(config, tokenizer=tok)
    assert collator is not None
    inner = collator._default_collator
    assert (
        inner.response_template == "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def test_train_target_unknown_tokenizer():
    """Error when tokenizer vocab does not match any known chat format."""
    tok = _unknown_tokenizer()
    config = TrainingConfig(
        data=DataParams(
            train=DatasetSplitParams(
                collator_name="text_completions_only_with_padding",
                train_target=TrainTarget.ALL_ASSISTANT_TURNS,
                datasets=[DatasetParams(dataset_name="dummy", split="train")],
            )
        ),
        model=ModelParams(
            model_name="MlpEncoder",
            tokenizer_name="openai-community/gpt2",
            model_max_length=512,
        ),
    )
    with pytest.raises(ValueError, match="no chat template"):
        build_collator_from_config(config, tokenizer=tok)


def test_train_target_with_collator_kwargs_override():
    """collator_kwargs overrides auto-resolved templates when train_target is set."""
    tok = _chatml_tokenizer()
    config = TrainingConfig(
        data=DataParams(
            train=DatasetSplitParams(
                collator_name="text_completions_only_with_padding",
                train_target=TrainTarget.ALL_ASSISTANT_TURNS,
                collator_kwargs={"response_template": "<|im_end|>\n"},
                datasets=[DatasetParams(dataset_name="dummy", split="train")],
            )
        ),
        model=ModelParams(
            model_name="MlpEncoder",
            tokenizer_name="openai-community/gpt2",
            model_max_length=512,
        ),
    )
    collator = build_collator_from_config(config, tokenizer=tok)
    assert collator is not None
    inner = collator._default_collator
    # Auto-resolved would be "<|im_start|>assistant\n"; user override wins
    assert inner.response_template == "<|im_end|>\n"


def test_train_target_on_wrong_collator():
    """train_target is only valid for text_completions_only_with_padding."""
    with pytest.raises(ValueError, match="train_target.*requires"):
        DatasetSplitParams(
            collator_name="text_with_padding",
            train_target=TrainTarget.ALL_ASSISTANT_TURNS,
            datasets=[DatasetParams(dataset_name="dummy", split="train")],
        )


def test_legacy_instruction_template_backward_compat(mock_tokenizer):
    """Legacy path: instruction_template + response_template → _legacy + warning."""
    config = TrainingConfig(
        data=DataParams(
            train=DatasetSplitParams(
                collator_name="text_completions_only_with_padding",
                collator_kwargs={
                    "response_template": "<|assistant|>",
                    "instruction_template": "<|user|>",
                },
                datasets=[DatasetParams(dataset_name="dummy", split="train")],
            )
        ),
        model=ModelParams(
            model_name="MlpEncoder",
            tokenizer_name="openai-community/gpt2",
            model_max_length=512,
        ),
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


def test_bare_collator_name_raises_without_templates(mock_tokenizer):
    """Bare collator_name without kwargs or train_target raises an error."""
    config = TrainingConfig(
        data=DataParams(
            train=DatasetSplitParams(
                collator_name="text_completions_only_with_padding",
                datasets=[DatasetParams(dataset_name="dummy", split="train")],
            )
        ),
        model=ModelParams(
            model_name="MlpEncoder",
            tokenizer_name="openai-community/gpt2",
            model_max_length=512,
        ),
    )
    with pytest.raises(ValueError, match="response_template"):
        build_collator_from_config(config, tokenizer=mock_tokenizer)


def test_old_recipe_response_only_sets_final(mock_tokenizer):
    """Old recipe: response_template only → final_assistant_turn."""
    config = TrainingConfig(
        data=DataParams(
            train=DatasetSplitParams(
                collator_name="text_completions_only_with_padding",
                collator_kwargs={
                    "response_template": "<|assistant|>",
                },
                datasets=[DatasetParams(dataset_name="dummy", split="train")],
            )
        ),
        model=ModelParams(
            model_name="MlpEncoder",
            tokenizer_name="openai-community/gpt2",
            model_max_length=512,
        ),
    )
    collator = build_collator_from_config(config, tokenizer=mock_tokenizer)
    assert collator is not None
    assert collator._default_collator.train_target == "final_assistant_turn"


def test_old_recipe_eot_sets_all_assistant(mock_tokenizer):
    """Old recipe: response_template + end_of_turn_template → all_assistant_turns."""
    config = TrainingConfig(
        data=DataParams(
            train=DatasetSplitParams(
                collator_name="text_completions_only_with_padding",
                collator_kwargs={
                    "response_template": "<|assistant|>",
                    "end_of_turn_template": "<|end|>",
                },
                datasets=[DatasetParams(dataset_name="dummy", split="train")],
            )
        ),
        model=ModelParams(
            model_name="MlpEncoder",
            tokenizer_name="openai-community/gpt2",
            model_max_length=512,
        ),
    )
    collator = build_collator_from_config(config, tokenizer=mock_tokenizer)
    assert collator is not None
    assert collator._default_collator.train_target == "all_assistant_turns"
