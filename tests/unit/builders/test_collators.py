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


def test_build_data_collator_text_completions_with_tool_kwargs(mock_tokenizer):
    collator_name = "text_completions_only_with_padding"
    resp = "<|im_start|>" + "assistant\n"
    eot = "<|im_end|>"

    # Basic build with end_of_turn_template
    collator = build_data_collator(
        collator_name,
        mock_tokenizer,
        max_length=None,
        response_template=resp,
        end_of_turn_template=eot,
    )
    assert collator is not None
    assert callable(collator)

    # Default label_ignore_index is forwarded
    assert collator._default_collator.ignore_index == constants.LABEL_IGNORE_INDEX

    # Custom label_ignore_index is forwarded
    collator_custom = build_data_collator(
        collator_name,
        mock_tokenizer,
        max_length=None,
        label_ignore_index=-200,
        response_template=resp,
        end_of_turn_template=eot,
    )
    assert collator_custom._default_collator.ignore_index == -200

    # With masking_method=assistant_turn_no_tools
    collator_tc = build_data_collator(
        collator_name,
        mock_tokenizer,
        max_length=None,
        response_template=resp,
        masking_method="assistant_turn_no_tools",
        end_of_turn_template=eot,
        tool_call_start_template="<tool_call>",
    )
    assert collator_tc is not None
    assert callable(collator_tc)


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
# masking_method tests
# ---------------------------------------------------------------------------


def test_masking_method_assistant_turn(mock_tokenizer):
    mock_tokenizer.get_vocab = MagicMock(
        return_value={"<|im_start|>": 1, "<|im_end|>": 2}
    )
    config = TrainingConfig(
        data=DataParams(
            train=DatasetSplitParams(
                collator_name="text_completions_only_with_padding",
                masking_method=MaskingMethod.ASSISTANT_TURN,
                datasets=[DatasetParams(dataset_name="dummy", split="train")],
            )
        ),
        model=ModelParams(
            model_name="Qwen/Qwen2.5-1.5B",
            model_max_length=64,
            trust_remote_code=True,
        ),
    )
    collator = build_collator_from_config(config, tokenizer=mock_tokenizer)
    assert collator is not None
    inner = collator._default_collator
    assert inner.response_template == "<|im_start|>assistant\n"
    assert inner.end_of_turn_template == "<|im_end|>"
    assert inner.masking_method == "assistant_turn"
    assert inner.mask_tool_calls is False


def test_masking_method_no_tools(mock_tokenizer):
    mock_tokenizer.get_vocab = MagicMock(
        return_value={"<|im_start|>": 1, "<|im_end|>": 2}
    )
    config = TrainingConfig(
        data=DataParams(
            train=DatasetSplitParams(
                collator_name="text_completions_only_with_padding",
                masking_method=MaskingMethod.ASSISTANT_TURN_NO_TOOLS,
                datasets=[DatasetParams(dataset_name="dummy", split="train")],
            )
        ),
        model=ModelParams(
            model_name="Qwen/Qwen2.5-1.5B",
            model_max_length=64,
            trust_remote_code=True,
        ),
    )
    collator = build_collator_from_config(config, tokenizer=mock_tokenizer)
    assert collator is not None
    inner = collator._default_collator
    assert inner.mask_tool_calls is True
    assert inner.tool_call_start_token_ids is not None


def test_masking_method_final_assistant_turn(mock_tokenizer):
    mock_tokenizer.get_vocab = MagicMock(
        return_value={"<|im_start|>": 1, "<|im_end|>": 2}
    )
    config = TrainingConfig(
        data=DataParams(
            train=DatasetSplitParams(
                collator_name="text_completions_only_with_padding",
                masking_method=MaskingMethod.FINAL_ASSISTANT_TURN,
                datasets=[DatasetParams(dataset_name="dummy", split="train")],
            )
        ),
        model=ModelParams(
            model_name="Qwen/Qwen2.5-1.5B",
            model_max_length=64,
            trust_remote_code=True,
        ),
    )
    collator = build_collator_from_config(config, tokenizer=mock_tokenizer)
    assert collator is not None
    inner = collator._default_collator
    assert inner.masking_method == "final_assistant_turn"
    assert inner.mask_tool_calls is False


def test_masking_method_llama3(mock_tokenizer):
    mock_tokenizer.get_vocab = MagicMock(
        return_value={"<|start_header_id|>": 1, "<|eot_id|>": 2}
    )
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
            model_max_length=64,
        ),
    )
    collator = build_collator_from_config(config, tokenizer=mock_tokenizer)
    assert collator is not None
    inner = collator._default_collator
    assert inner.end_of_turn_template == "<|eot_id|>"


def test_masking_method_unknown_tokenizer(mock_tokenizer):
    mock_tokenizer.get_vocab = MagicMock(return_value={"hello": 1})
    config = TrainingConfig(
        data=DataParams(
            train=DatasetSplitParams(
                collator_name="text_completions_only_with_padding",
                masking_method=MaskingMethod.ASSISTANT_TURN,
                datasets=[DatasetParams(dataset_name="dummy", split="train")],
            )
        ),
        model=ModelParams(model_name="MlpEncoder", model_max_length=64),
    )
    with pytest.raises(ValueError, match="Cannot detect collator templates"):
        build_collator_from_config(config, tokenizer=mock_tokenizer)


def test_masking_method_and_collator_kwargs_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        DatasetSplitParams(
            masking_method=MaskingMethod.ASSISTANT_TURN,
            collator_kwargs={"response_template": "foo"},
            datasets=[DatasetParams(dataset_name="dummy", split="train")],
        )


def test_masking_method_on_wrong_collator(mock_tokenizer):
    mock_tokenizer.get_vocab = MagicMock(
        return_value={"<|im_start|>": 1, "<|im_end|>": 2}
    )
    config = TrainingConfig(
        data=DataParams(
            train=DatasetSplitParams(
                collator_name="text_with_padding",
                masking_method=MaskingMethod.ASSISTANT_TURN,
                datasets=[DatasetParams(dataset_name="dummy", split="train")],
            )
        ),
        model=ModelParams(model_name="MlpEncoder", model_max_length=64),
    )
    with pytest.raises(ValueError, match="only supported for"):
        build_collator_from_config(config, tokenizer=mock_tokenizer)


def test_no_masking_method_backward_compat(mock_tokenizer):
    config = TrainingConfig(
        data=DataParams(
            train=DatasetSplitParams(
                collator_name="text_completions_only_with_padding",
                collator_kwargs={
                    "response_template": "<|im_start|>assistant\n",
                    "end_of_turn_template": "<|im_end|>",
                },
                datasets=[DatasetParams(dataset_name="dummy", split="train")],
            )
        ),
        model=ModelParams(
            model_name="Qwen/Qwen2.5-1.5B",
            model_max_length=64,
            trust_remote_code=True,
        ),
    )
    collator = build_collator_from_config(config, tokenizer=mock_tokenizer)
    assert collator is not None


def test_masking_method_llama3_no_tools(mock_tokenizer):
    mock_tokenizer.get_vocab = MagicMock(
        return_value={
            "<|start_header_id|>": 1,
            "<|eot_id|>": 2,
            "<|python_tag|>": 3,
        }
    )
    config = TrainingConfig(
        data=DataParams(
            train=DatasetSplitParams(
                collator_name="text_completions_only_with_padding",
                masking_method=MaskingMethod.ASSISTANT_TURN_NO_TOOLS,
                datasets=[DatasetParams(dataset_name="dummy", split="train")],
            )
        ),
        model=ModelParams(
            model_name="MlpEncoder",
            model_max_length=64,
        ),
    )
    collator = build_collator_from_config(config, tokenizer=mock_tokenizer)
    assert collator is not None
    inner = collator._default_collator
    assert inner.mask_tool_calls is True
    assert inner.tool_call_start_token_ids is not None


def test_legacy_collator_kwargs_with_instruction_template(mock_tokenizer):
    """Old configs that pass instruction_template via collator_kwargs use
    the legacy instruction+response masking path."""
    config = TrainingConfig(
        data=DataParams(
            train=DatasetSplitParams(
                collator_name="text_completions_only_with_padding",
                collator_kwargs={
                    "response_template": (
                        "<|start_header_id|>assistant<|end_header_id|>\n\n"
                    ),
                    "instruction_template": (
                        "<|start_header_id|>user<|end_header_id|>\n\n"
                    ),
                },
                datasets=[DatasetParams(dataset_name="dummy", split="train")],
            )
        ),
        model=ModelParams(
            model_name="MlpEncoder",
            model_max_length=64,
        ),
    )
    collator = build_collator_from_config(config, tokenizer=mock_tokenizer)
    assert collator is not None
    assert collator is not None
    inner = collator._default_collator
    assert inner.instruction_template == (
        "<|start_header_id|>user<|end_header_id|>\n\n"
    )
    assert inner.response_template == (
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    # Verify it inferred the legacy instruction+response masking path
    assert inner.masking_method == "_legacy_instruction_response"
