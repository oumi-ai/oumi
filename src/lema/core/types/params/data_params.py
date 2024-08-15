import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from omegaconf import MISSING

from lema.core.types.params.base_params import BaseParams


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
    #: Parameters for `datasets.load_dataset()`
    dataset_name: str = MISSING
    #: The subset of the dataset to load, usually a subfolder within the dataset root.
    subset: Optional[str] = None
    #: The split of the dataset to load, usually "train", "test", or "validation".
    split: str = "train"
    #: Keyword arguments to pass to the dataset constructor.
    dataset_kwargs: Dict[str, Any] = field(default_factory=dict)

    #: The number of examples to sample from the dataset. Must be non-negative. If
    #: `sample_count` is larger than the size of the dataset then the required
    #: additional examples are sampled by looping over the original dataset.
    #: Defaults to None.
    sample_count: Optional[int] = None
    #: The proportion of examples from this dataset relative to other datasets in the
    #: mixture. If specified, all datasets must supply this value. Must be a float in
    #: the range [0, 1.0]. The `mixture_proportion` for all input datasets must sum to
    #: 1. Examples are sampled after the dataset has been sampled using `sample_count`
    #: if specified. Defaults to None.
    mixture_proportion: Optional[float] = None
    #: If specified, the dataset is shuffled before any sampling occurs.
    shuffle: bool = False
    #: The random seed used for shuffling the dataset before sampling, if specified.
    #: If set to `None` shuffling will be non-deterministic.
    seed: Optional[int] = None
    #: The size of the shuffle buffer used for shuffling the dataset before sampling.
    shuffle_buffer_size: int = 1000

    @staticmethod
    def _default_factory_preprocessing_kwargs() -> dict:
        """Creates default param values for the data preprocessing .map function.

        Returns:
        dict: contains the default set params.
        """
        defaults = dict()
        defaults["batched"] = False  # Note: same default as huggingface data loader.
        return defaults

    preprocessing_function_name: Optional[str] = None
    preprocessing_function_kwargs: Dict[str, Any] = field(
        default_factory=_default_factory_preprocessing_kwargs
    )

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


@dataclass
class DatasetSplitParams(BaseParams):
    #: The input datasets used for training. This will later be split into train, test,
    #: and validation.
    datasets: List[DatasetParams] = field(default_factory=list)
    #: Whether to pack the text into constant-length chunks,
    #: each the size of the model's max input length.
    #: This will stream the dataset, and tokenize on the fly
    #: if the dataset isn't already tokenized (i.e. has an `input_ids` column).
    #: Requires `stream` to be set to True.
    pack: bool = False
    stream: bool = False
    #: The dataset column name containing the input for training/testing/validation.
    #: Required for SFTTrainer. If specified, all datasets in this split must contain a
    #: column with this name.
    target_col: Optional[str] = None
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
    #: The random seed used for mixing this dataset split, if specified.
    #: If set to `None` mixing will be non-deterministic.
    seed: Optional[int] = None

    #: EXPERIMENTAL PARAMS -------------------------
    #: Whether to use the PretrainingAsyncTextDataset instead of ConstantLengthDataset.
    experimental_use_async_dataset: bool = False
    #: END EXPERIMENTAL PARAMS --------------------

    def __post_init__(self):
        """Verifies params."""
        if self.pack:
            # TODO: why is this check necessary?
            if not self.stream:
                raise ValueError("`stream` must be enabled if `pack` is enabled.")
            if not self.target_col:
                raise ValueError("`target_col` must be specified if `pack` is enabled.")
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
    #: The input datasets used for training.
    train: DatasetSplitParams = field(default_factory=DatasetSplitParams)

    #: The input datasets used for testing.
    test: DatasetSplitParams = field(default_factory=DatasetSplitParams)

    #: The input datasets used for validation.
    validation: DatasetSplitParams = field(default_factory=DatasetSplitParams)

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
