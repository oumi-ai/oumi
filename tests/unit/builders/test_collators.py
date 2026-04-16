import re
from unittest.mock import MagicMock

import pytest

import oumi.core.constants as constants
from oumi.builders.collators import build_collator_from_config, build_data_collator
from oumi.core.configs import (
    DataParams,
    DatasetParams,
    DatasetSplitParams,
    MaskingMethod,
    ModelParams,
    TrainingConfig,
    TrainingParams,
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
# MaskingMethod / builder auto-detection tests
# ---------------------------------------------------------------------------


def _chatml_tokenizer():
    """Return a mock tokenizer whose vocab contains ChatML special tokens."""
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.model_max_length = 2048
    tok.get_vocab.return_value = {
        "<|im_start|>": 100,
        "<|im_end|>": 101,
    }
    return tok


def _llama3_tokenizer():
    """Return a mock tokenizer whose vocab contains Llama-3 special tokens."""
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.model_max_length = 2048
    tok.get_vocab.return_value = {
        "<|start_header_id|>": 200,
        "<|end_header_id|>": 201,
        "<|eot_id|>": 202,
    }
    return tok


def _unknown_tokenizer():
    """Return a mock tokenizer with no recognisable chat tokens."""
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.model_max_length = 2048
    tok.get_vocab.return_value = {"hello": 1, "world": 2}
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
        masking_method="assistant_turn",
    )
    assert collator is not None
    assert callable(collator)
    inner = collator._default_collator
    assert inner.ignore_index == -200


def test_masking_method_assistant_turn():
    """ChatML auto-detection with ASSISTANT_TURN masking method."""
    tok = _chatml_tokenizer()
    config = TrainingConfig(
        data=DataParams(
            train=DatasetSplitParams(
                collator_name="text_completions_only_with_padding",
                masking_method=MaskingMethod.ASSISTANT_TURN,
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


def test_masking_method_final_assistant_turn():
    """ChatML auto-detection with FINAL_ASSISTANT_TURN masking method."""
    tok = _chatml_tokenizer()
    config = TrainingConfig(
        data=DataParams(
            train=DatasetSplitParams(
                collator_name="text_completions_only_with_padding",
                masking_method=MaskingMethod.FINAL_ASSISTANT_TURN,
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


def test_masking_method_llama3():
    """Llama-3 auto-detection with ASSISTANT_TURN masking method."""
    tok = _llama3_tokenizer()
    config = TrainingConfig(
        data=DataParams(
            train=DatasetSplitParams(
                collator_name="text_completions_only_with_padding",
                masking_method=MaskingMethod.ASSISTANT_TURN,
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


def test_masking_method_unknown_tokenizer():
    """Error when tokenizer vocab does not match any known chat format."""
    tok = _unknown_tokenizer()
    config = TrainingConfig(
        data=DataParams(
            train=DatasetSplitParams(
                collator_name="text_completions_only_with_padding",
                masking_method=MaskingMethod.ASSISTANT_TURN,
                datasets=[DatasetParams(dataset_name="dummy", split="train")],
            )
        ),
        model=ModelParams(
            model_name="MlpEncoder",
            tokenizer_name="openai-community/gpt2",
            model_max_length=512,
        ),
    )
    with pytest.raises(ValueError, match="Cannot auto-detect chat template format"):
        build_collator_from_config(config, tokenizer=tok)


def test_masking_method_and_collator_kwargs_exclusive():
    """masking_method and collator_kwargs are mutually exclusive."""
    with pytest.raises(ValueError, match="Cannot specify both"):
        DatasetSplitParams(
            collator_name="text_completions_only_with_padding",
            masking_method=MaskingMethod.ASSISTANT_TURN,
            collator_kwargs={"response_template": "<|assistant|>"},
            datasets=[DatasetParams(dataset_name="dummy", split="train")],
        )


def test_masking_method_on_wrong_collator():
    """masking_method is only valid for text_completions_only_with_padding."""
    tok = _chatml_tokenizer()
    config = TrainingConfig(
        data=DataParams(
            train=DatasetSplitParams(
                collator_name="text_with_padding",
                masking_method=MaskingMethod.ASSISTANT_TURN,
                datasets=[DatasetParams(dataset_name="dummy", split="train")],
            )
        ),
        model=ModelParams(
            model_name="MlpEncoder",
            tokenizer_name="openai-community/gpt2",
            model_max_length=512,
        ),
    )
    with pytest.raises(ValueError, match="only supported with"):
        build_collator_from_config(config, tokenizer=tok)


def test_no_masking_method_backward_compat(mock_tokenizer):
    """collator_kwargs still work when masking_method is not set."""
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
    collator = build_collator_from_config(config, tokenizer=mock_tokenizer)
    assert collator is not None
    inner = collator._default_collator
    assert inner.response_template == "<|assistant|>"
    assert inner.instruction_template == "<|user|>"


def test_bare_collator_name_raises_without_templates(mock_tokenizer):
    """Bare collator_name without kwargs or masking_method raises an error."""
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
    with pytest.raises(ValueError, match="requires a `response_template`"):
        build_collator_from_config(config, tokenizer=mock_tokenizer)


def test_legacy_collator_kwargs_with_instruction_template(mock_tokenizer):
    """Legacy path: collator_kwargs with instruction_template still works."""
    config = TrainingConfig(
        data=DataParams(
            train=DatasetSplitParams(
                collator_name="text_completions_only_with_padding",
                collator_kwargs={
                    "response_template": "<|start_header_id|>assistant"
                    "<|end_header_id|>\n\n",
                    "instruction_template": "<|start_header_id|>user"
                    "<|end_header_id|>\n\n",
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
    inner = collator._default_collator
    assert (
        inner.response_template == "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    assert inner.instruction_template == "<|start_header_id|>user<|end_header_id|>\n\n"
