import re
from typing import Optional
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
    label_ignore_index: Optional[int], mock_tokenizer
):
    training_config = TrainingConfig(
        data=DataParams(
            train=DatasetSplitParams(
                collator_name="text_with_padding",
                datasets=[DatasetParams(dataset_name="dummy", split="train")],
            )
        ),
        model=ModelParams(
            model_name="MlpEncoder", tokenizer_name="gpt2", model_max_length=64
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
            model_name="MlpEncoder", tokenizer_name="gpt2", model_max_length=64
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
            model_name="MlpEncoder", tokenizer_name="gpt2", model_max_length=64
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
            model_name="CnnClassifier", tokenizer_name="gpt2", model_max_length=64
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
            model_name="MlpEncoder", tokenizer_name="gpt2", model_max_length=64
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
