import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from omegaconf import MISSING

from lema.core.configs.params.base_params import BaseParams


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
    """The name of the dataset to load. Required."""

    subset: Optional[str] = None
    """The subset of the dataset to load.

    This is usually a subfolder within the dataset root.
    """

    split: str = "train"
    """The split of the dataset to load.

    This is typically one of "train", "test", or "validation". Defaults to "train".
    """

    dataset_kwargs: Dict[str, Any] = field(default_factory=dict)
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
    """[Deprecated] The name of the preprocessing function to apply to the dataset.

    If specified, this function will be applied to the dataset using the dataset's
    `map` method. The function should be defined in the preprocessing module.

    Warning:
        This is deprecated and will be removed in a future release.

        To customize dataset preprocessing, please see `Dataset.transform`.
    """

    preprocessing_function_kwargs: Dict[str, Any] = field(
        default_factory=_default_factory_preprocessing_kwargs
    )
    """[Deprecated] Keyword arguments to pass to the preprocessing function.

    These arguments will be passed directly to the preprocessing function when it
    is applied to the dataset using the `map` method.

    Warning:
        This is deprecated and will be removed in a future release.

        To customize dataset preprocessing, please see `Dataset.transform`.
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


@dataclass
class DatasetSplitParams(BaseParams):
    datasets: List[DatasetParams] = field(default_factory=list)
    """The input datasets used for training.

    This will later be split into train, test, and validation.
    """

    pack: bool = False
    """Whether to pack the text into constant-length chunks.

    Each chunk will be the size of the model's max input length.
    This will stream the dataset, and tokenize on the fly
    if the dataset isn't already tokenized (i.e. has an `input_ids` column).
    Requires `stream` to be set to True.
    """

    stream: bool = False
    """Whether to stream the dataset."""

    target_col: Optional[str] = None
    """The dataset column name containing the input for training/testing/validation.

    Required for SFTTrainer. If specified, all datasets in this split must contain a
    column with this name.
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

    # EXPERIMENTAL PARAMS -------------------------
    experimental_use_async_dataset: bool = False
    """Whether to use the PretrainingAsyncTextDataset instead of ConstantLengthDataset.

    Warning:
        This is an experimental feature and may change without notice.
    """
    # END EXPERIMENTAL PARAMS --------------------

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
    train: DatasetSplitParams = field(default_factory=DatasetSplitParams)
    """The input datasets used for training."""

    test: DatasetSplitParams = field(default_factory=DatasetSplitParams)
    """The input datasets used for testing."""

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
