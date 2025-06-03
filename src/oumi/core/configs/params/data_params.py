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

import math
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Literal, Optional, Union

from omegaconf import MISSING

from oumi.core.configs.params.base_params import BaseParams


# Training Params
#
#
# Dataset Splits
#
class DatasetSplit(Enum):
    """Enum representing the split for a dataset."""

    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"


class MixtureStrategy(str, Enum):
    """Enum representing the supported mixture strategies for datasets."""

    FIRST_EXHAUSTED = "first_exhausted"
    ALL_EXHAUSTED = "all_exhausted"

    def get_literal_value(self) -> Literal["first_exhausted", "all_exhausted"]:
        """Returns a literal value of the enum."""
        if self.value == MixtureStrategy.FIRST_EXHAUSTED:
            return "first_exhausted"
        elif self.value == MixtureStrategy.ALL_EXHAUSTED:
            return "all_exhausted"
        else:
            raise ValueError("Unsupported value for MixtureStrategy")


@dataclass
class DatasetParams(BaseParams):
    dataset_name: str = MISSING
    """The name of the dataset to load. Required.

    This field is used to retrieve the appropriate class from the dataset registry
    that can be used to instantiate and preprocess the data.

    If `dataset_path` is not specified, then the raw data will be automatically
    downloaded from the huggingface hub or oumi registry. Otherwise, the dataset will
    be loaded from the specified `dataset_path`.
    """

    dataset_path: Optional[str] = None
    """The path to the dataset to load.

    This can be used to load a dataset of type `dataset_name` from a custom path.

    If `dataset_path` is not specified, then the raw data will be automatically
    downloaded from the huggingface hub or oumi registry.
    """

    subset: Optional[str] = None
    """The subset of the dataset to load.

    This is usually a subfolder within the dataset root.
    """

    split: str = "train"
    """The split of the dataset to load.

    This is typically one of "train", "test", or "validation". Defaults to "train".
    """

    dataset_kwargs: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments to pass to the dataset constructor.

    These arguments will be passed directly to the dataset constructor.
    """

    sample_count: Optional[int] = None
    """The number of examples to sample from the dataset.

    Must be non-negative. If `sample_count` is larger than the size of the dataset, then
    the required additional examples are sampled by looping over the original dataset.
    """

    mixture_proportion: Optional[float] = None
    """The proportion of examples from this dataset relative to other datasets
        in the mixture.

    If specified, all datasets must supply this value.
    Must be a float in the range [0, 1.0]. The `mixture_proportion` for all input
    datasets must sum to 1.

    Examples are sampled after the dataset has been sampled using `sample_count`
    if specified.
    """

    shuffle: bool = False
    """Whether to shuffle the dataset before any sampling occurs."""

    seed: Optional[int] = None
    """The random seed used for shuffling the dataset before sampling.

    If set to `None`, shuffling will be non-deterministic.
    """

    shuffle_buffer_size: int = 1000
    """The size of the shuffle buffer used for shuffling the dataset before sampling."""

    trust_remote_code: bool = False
    """Whether to trust remote code when loading the dataset."""

    transform_num_workers: Optional[Union[str, int]] = None
    """Number of subprocesses to use for dataset post-processing (`ds.transform()`).

    Multiprocessing is disabled by default (`None`).

    You can also use the special value "auto" to let oumi automatically
    select the number of subprocesses.

    Using multiple processes can speed-up processing
    e.g., for large or multi-modal datasets.

    The parameter is only supported for Map (non-iterable) datasets.
    """

    def __post_init__(self):
        """Verifies params."""
        if self.sample_count is not None:
            if self.sample_count < 0:
                raise ValueError("`sample_count` must be greater than 0.")
        if self.mixture_proportion is not None:
            if self.mixture_proportion < 0:
                raise ValueError("`mixture_proportion` must be greater than 0.")
            if self.mixture_proportion > 1:
                raise ValueError("`mixture_proportion` must not be greater than 1.0 .")

        if self.transform_num_workers is not None:
            if isinstance(self.transform_num_workers, str):
                if not (self.transform_num_workers == "auto"):
                    raise ValueError(
                        "Unknown value of transform_num_workers: "
                        f"{self.transform_num_workers}. Must be 'auto' if string."
                    )
            elif (not isinstance(self.transform_num_workers, int)) or (
                self.transform_num_workers <= 0
            ):
                raise ValueError(
                    "Non-positive value of transform_num_workers: "
                    f"{self.transform_num_workers}."
                )

        if len(self.dataset_kwargs) > 0:
            conflicting_keys = {f.name for f in fields(self)}.intersection(
                self.dataset_kwargs.keys()
            )
            if len(conflicting_keys) > 0:
                raise ValueError(
                    "dataset_kwargs attempts to override the following "
                    f"reserved fields: {conflicting_keys}. "
                    "Use properties of DatasetParams instead."
                )


@dataclass
class DatasetSplitParams(BaseParams):
    datasets: list[DatasetParams] = field(default_factory=list)
    """The datasets in this split."""

    collator_name: Optional[str] = None
    """Name of Oumi data collator.

    Data collator controls how to form a mini-batch from individual dataset elements.

    Valid options are:

        - "text_with_padding": Dynamically pads the inputs received to
            the longest length.
        - "vision_language_with_padding": Uses VisionLanguageCollator
            for image+text multi-modal data.

    If None, then a default collator will be assigned.
    """

    collator_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments to pass to the collator constructor.

    These arguments will be passed directly to the collator constructor
    and can be used to customize collator behavior beyond the default parameters.
    """

    pack: bool = False
    """Whether to pack the text into constant-length chunks.

    Each chunk will be the size of the model's max input length.
    This will stream the dataset, and tokenize on the fly
    if the dataset isn't already tokenized (i.e. has an `input_ids` column).
    """

    stream: bool = False
    """Whether to stream the dataset."""

    target_col: Optional[str] = None
    """The dataset column name containing the input for training/testing/validation.

    Deprecated:
        This parameter is deprecated and will be removed in the future.
    """

    mixture_strategy: str = field(
        default=MixtureStrategy.FIRST_EXHAUSTED.value,
        metadata={
            "help": "The mixture strategy to use when multiple datasets are "
            f"provided. `{MixtureStrategy.FIRST_EXHAUSTED.value}` will sample from all "
            "datasets until exactly one dataset is completely represented in the "
            f"mixture. `{MixtureStrategy.ALL_EXHAUSTED.value}` will sample from all "
            "datasets until every dataset is completely represented in the "
            f"mixture. Note that `{MixtureStrategy.ALL_EXHAUSTED.value}` may result in "
            "significant oversampling. Defaults to "
            f"`{MixtureStrategy.FIRST_EXHAUSTED.value}`."
        },
    )
    """The strategy for mixing multiple datasets.

    When multiple datasets are provided, this parameter determines how they are
    combined. Two strategies are available:

    1. FIRST_EXHAUSTED: Samples from all datasets until one is fully represented
       in the mixture. This is the default strategy.
    2. ALL_EXHAUSTED: Samples from all datasets until each one is fully represented
       in the mixture. This may lead to significant oversampling.
    """

    seed: Optional[int] = None
    """The random seed used for mixing this dataset split, if specified.

    If set to `None` mixing will be non-deterministic.
    """

    use_async_dataset: bool = False
    """Whether to use the PretrainingAsyncTextDataset instead of ConstantLengthDataset.

    Deprecated:
        This parameter is deprecated and will be removed in the future.
    """

    use_torchdata: Optional[bool] = None
    """Whether to use the `torchdata` library for dataset loading and processing.

    If set to `None`, this setting may be auto-inferred.
    """

    def __post_init__(self):
        """Verifies params."""
        if any([dataset.mixture_proportion is not None for dataset in self.datasets]):
            if not all(
                [dataset.mixture_proportion is not None for dataset in self.datasets]
            ):
                raise ValueError(
                    "If `mixture_proportion` is specified it must be "
                    " specified for all datasets"
                )
            mix_sum = sum(
                filter(None, [dataset.mixture_proportion for dataset in self.datasets])
            )
            if not self._is_sum_normalized(mix_sum):
                raise ValueError(
                    "The sum of `mixture_proportion` must be 1.0. "
                    f"The current sum is {mix_sum} ."
                )
        if (
            self.mixture_strategy != MixtureStrategy.ALL_EXHAUSTED
            and self.mixture_strategy != MixtureStrategy.FIRST_EXHAUSTED
        ):
            raise ValueError(
                "`mixture_strategy` must be one of "
                f'["{MixtureStrategy.FIRST_EXHAUSTED.value}", '
                f'"{MixtureStrategy.ALL_EXHAUSTED.value}"].'
            )

    def _is_sum_normalized(self, mix_sum) -> bool:
        # Note: the underlying interleave implementation requires
        # the mixture proportions to sum to 1.0.
        return math.isclose(mix_sum, 1.0)


@dataclass
class DataParams(BaseParams):
    train: DatasetSplitParams = field(default_factory=DatasetSplitParams)
    """The input datasets used for training."""

    test: DatasetSplitParams = field(default_factory=DatasetSplitParams)
    """The input datasets used for testing. This field is currently unused."""

    validation: DatasetSplitParams = field(default_factory=DatasetSplitParams)
    """The input datasets used for validation."""

    def get_split(self, split: DatasetSplit) -> DatasetSplitParams:
        """A public getting for individual dataset splits."""
        if split == DatasetSplit.TRAIN:
            return self.train
        elif split == DatasetSplit.TEST:
            return self.test
        elif split == DatasetSplit.VALIDATION:
            return self.validation
        else:
            raise ValueError(f"Received invalid split: {split}.")

    def __finalize_and_validate__(self):
        """Verifies params."""
        if len(self.train.datasets) == 0:
            raise ValueError("At least one training dataset is required.")

        all_collators = set()
        if self.train.collator_name:
            all_collators.add(self.train.collator_name)
        if self.validation.collator_name:
            all_collators.add(self.validation.collator_name)
        if self.test.collator_name:
            all_collators.add(self.test.collator_name)
        if len(all_collators) >= 2:
            raise ValueError(
                f"Different data collators are not supported yet: {all_collators}"
            )
        elif len(all_collators) == 1 and not self.train.collator_name:
            raise ValueError(
                "Data collator must be also specified "
                f"on the `train` split: {all_collators}"
            )
