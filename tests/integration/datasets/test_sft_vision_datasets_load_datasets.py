import os
from typing import Final, NamedTuple, Optional

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

_DEFALT_DATASET_SPLIT: Final[str] = "test"
_DEFAULT_MODEL_NAME: Final[str] = "Qwen/Qwen2-VL-2B-Instruc"
_DEFAULT_CHAT_TEMPLATE: Final[str] = "llava"


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
    expected_rows: Optional[int] = 32
    extra_dataset_features: Optional[list[str]] = None
    chat_template: str = _DEFAULT_CHAT_TEMPLATE
    dataset_split: str = _DEFALT_DATASET_SPLIT
    collator_name: str = "vision_language_with_padding"
    trust_remote_code: bool = False


def _get_all_sft_vision_dataset_infos() -> list[LoadDatasetInfo]:
    # Special case datasets that should be excluded from default testing.
    _EXCLUDED_DATASETS = set({"coco_captions", "vision_language_jsonl", "vl_sft"})

    all_dataset_names = set(_get_all_sft_vision_dataset_names())
    result = [
        LoadDatasetInfo(
            dataset_name="merve/vqav2-small",
            model_name=_DEFAULT_MODEL_NAME,
            dataset_split="validation",
            chat_template=_DEFAULT_CHAT_TEMPLATE,
            trust_remote_code=True,
            max_rows=64,
            expected_rows=64,
        ),
        LoadDatasetInfo(
            dataset_name="merve/vqav2-small",
            model_name="microsoft/Phi-3-vision-128k-instruct",
            dataset_split="validation",
            extra_dataset_features=["image_sizes"],
            chat_template="phi3-instruct",
            trust_remote_code=True,
            max_rows=32,
            expected_rows=32,
        ),
    ]

    manually_configured_dataset_names = set({info.dataset_name for info in result})
    for dataset_name in all_dataset_names:
        if (dataset_name in manually_configured_dataset_names) or (
            dataset_name in _EXCLUDED_DATASETS
        ):
            continue
        result.append(
            LoadDatasetInfo(
                dataset_name=dataset_name,
                model_name=_DEFAULT_MODEL_NAME,
                dataset_split=_DEFALT_DATASET_SPLIT,
                chat_template=_DEFAULT_CHAT_TEMPLATE,
                trust_remote_code=True,
            )
        )

    assert len(result) > 1
    for idx, info in enumerate(result):
        assert info.dataset_name, f"Index: {idx}"
        assert info.dataset_name in all_dataset_names, f"Index: {idx}"
        assert info.model_name, f"Index: {idx}"
        assert info.chat_template, f"Index: {idx}"
        assert info.dataset_split, f"Index: {idx}"
        assert info.collator_name, f"Index: {idx}"

    return result


@pytest.skip(
    "This test is very time consuming, and should be run manually.",
    allow_module_level=True,
)
@pytest.mark.parametrize("info", _get_all_sft_vision_dataset_infos())
def test_build_dataset_mixture(info: LoadDatasetInfo):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    debug_tag = f"Test: {info}"
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

    assert dataset.num_rows > 0, debug_tag
    assert dataset.num_rows <= info.max_rows, debug_tag

    if info.expected_rows is not None:
        assert dataset.num_rows == info.expected_rows, debug_tag

    assert "input_ids" in dataset.features, debug_tag
    assert "attention_mask" in dataset.features, debug_tag
    assert "pixel_values" in dataset.features, debug_tag
    assert "labels" in dataset.features, debug_tag

    if info.extra_dataset_features is not None and len(info.extra_dataset_features) > 0:
        for extra_feature in info.extra_dataset_features:
            assert extra_feature in dataset.features, debug_tag

    assert dataset[0] is not None, debug_tag
    assert dataset[dataset.num_rows - 1] is not None, debug_tag


if __name__ == "__main__":
    datasets.disable_caching()
    info = LoadDatasetInfo(
        dataset_name="merve/vqav2-small",
        model_name=_DEFAULT_MODEL_NAME,
        dataset_split="validation",
        chat_template=_DEFAULT_CHAT_TEMPLATE,
        trust_remote_code=True,
        max_rows=513,
    )
    test_build_dataset_mixture(info)
