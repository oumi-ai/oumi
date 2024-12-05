from dataclasses import dataclass, field
from enum import Enum
from typing import Any, NamedTuple, Optional

from oumi.core.configs.base_config import BaseConfig


class DatasetFirstDimAction(Enum):
    """Enum representing how to handle the first feature dimension in dataset."""

    DROP_ALWAYS = "drop_always"
    """The first dimension is commonly dummy (length: 1) and must be dropped.

    In effect, this operation is applied: `x = x[0, ...]`, which reduces
    `x`'s rank by 1 (e.g., 3D->2D), and discards the following elements: `x[1:, ...]`.
    """

    DROP_IF_DUMMY = "drop_if_dummy"
    """Drop the first dimension only if it's dummy (length: 1)."""

    KEEP = "keep"
    """Always preserve the first dimension."""


class DatasetInputFeatureSpec(NamedTuple):
    feature_name: str
    variable_shape: bool = False
    """Whether image features can be of variable shape."""
    first_dim_action: DatasetFirstDimAction = DatasetFirstDimAction.DROP_ALWAYS


@dataclass
class InternalVisualModelConfig(BaseConfig):
    variable_shape_image_features: bool = False
    """Whether image features can be of variable shape.

    In this case, the image features can be difficult to collate
    (`torch.stack()` requires compatible shapes) and some workaround
    is required: either require `batch_size=1`, or group examples
    so that each mini-batch only contains same-size features.
    """

    supports_multiple_images: bool = False
    """Whether the visual model supports multiple images."""

    label_ignore_index: Optional[int] = None
    """Special label value to be excluded from loss computation."""

    sanitize_nagative_labels: bool = False
    """Replace negative label values.

    Some VLM-s can use negative input_ids for image tokens,
    which can cause problems if used as label to compute loss.
    """

    processor_kwargs: dict[str, Any] = field(default_factory=dict)
    """Extra params to pass to processor constructor."""


@dataclass
class InternalModelConfig(BaseConfig):
    chat_template: str = "default"
    """Default chat template."""

    dataset_input_features: dict[str, DatasetInputFeatureSpec] = field(
        default_factory=dict
    )
    """Dataset input features specs."""

    visual_config: InternalVisualModelConfig = field(
        default_factory=InternalVisualModelConfig
    )
    """Configuration specific to visual models."""
