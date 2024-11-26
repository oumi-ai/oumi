from typing import NamedTuple, Optional

import datasets
import pytest

from oumi.builders import (
    build_dataset_mixture,
    build_tokenizer,
)
from oumi.core.configs import (
    DataParams,
    DatasetParams,
    DatasetSplit,
    DatasetSplitParams,
    ModelParams,
    TrainingConfig,
)
from oumi.core.datasets import VisionLanguageSftDataset
from oumi.core.registry import REGISTRY, RegistryType


def _get_all_sft_vision_dataset_names() -> list[str]:
    """List all SFT datasets in the registry."""
    datasets = []
    for key, value in REGISTRY._registry.items():
        if key.registry_type == RegistryType.DATASET and issubclass(
            value, VisionLanguageSftDataset
        ):
            datasets.append(key.name)
    return datasets


class LoadDatasetInfo(NamedTuple):
    dataset_name: str
    model_name: str
    max_rows: int = 32
    extra_dataset_features: Optional[list[str]] = None
    chat_template: str = "llama3-instruct"
    dataset_split: str = "train"
    collator_name: str = "vision_language_with_padding"
    trust_remote_code: bool = False


def _get_all_sft_vision_dataset_infos() -> list[LoadDatasetInfo]:
    names_set = set(_get_all_sft_vision_dataset_names())
    result = [
        LoadDatasetInfo(
            dataset_name="merve/vqav2-small",
            model_name="microsoft/Phi-3-vision-128k-instruct",
            dataset_split="validation",
            extra_dataset_features=["image_sizes"],
            trust_remote_code=True,
        )
    ]
    for idx, info in enumerate(result):
        assert info.dataset_name, f"Index: {idx}"
        assert info.dataset_name in names_set, f"Index: {idx}"
        assert info.model_name, f"Index: {idx}"
        assert info.chat_template, f"Index: {idx}"
        assert info.dataset_split, f"Index: {idx}"
        assert info.collator_name, f"Index: {idx}"

    return result


@pytest.mark.parametrize("info", _get_all_sft_vision_dataset_infos())
def test_build_dataset_mixture(info: LoadDatasetInfo):
    model_params = ModelParams(
        model_name=info.model_name,
        trust_remote_code=info.trust_remote_code,
        chat_template=info.chat_template,
    )
    tokenizer = build_tokenizer(model_params)
    train_split = DatasetSplitParams(
        collator_name=info.collator_name,
        target_col="text",
        datasets=[
            DatasetParams(
                dataset_name=info.dataset_name,
                split=info.dataset_split,
                shuffle=False,
                seed=42,
                trust_remote_code=info.trust_remote_code,
                dataset_kwargs={
                    "processor_name": info.model_name,
                    "limit": info.max_rows,
                    "return_tensors": True,
                },
            )
        ],
    )
    train_config = TrainingConfig(
        model=model_params, data=DataParams(train=train_split)
    )
    dataset = build_dataset_mixture(train_config, tokenizer, DatasetSplit.TRAIN)

    assert isinstance(dataset, datasets.Dataset)

    assert dataset.num_rows > 0
    assert dataset.num_rows <= info.max_rows

    assert "input_ids" in dataset.features
    assert "attention_mask" in dataset.features
    assert "pixel_values" in dataset.features
    assert "labels" in dataset.features

    if info.extra_dataset_features is not None and len(info.extra_dataset_features) > 0:
        for extra_feature in info.extra_dataset_features:
            assert extra_feature in dataset.features

    assert dataset[0] is not None
    assert dataset[dataset.num_rows - 1] is not None
