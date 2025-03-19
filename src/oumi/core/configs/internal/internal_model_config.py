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

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, NamedTuple, Optional

from oumi.core.configs.base_config import BaseConfig
from oumi.core.constants import LABEL_IGNORE_INDEX


class InternalPaddingSide(Enum):
    """Enum representing how to do padding for the model."""

    PAD_LEFT = "left"
    """Left padding."""

    PAD_RIGHT = "right"
    """Right padding."""


class InternalFeatureFirstDimAction(Enum):
    """Enum representing how to handle the first feature dimension in datasets."""

    DROP_ALWAYS = "drop_always"
    """The first dimension is commonly dummy (length: 1) and must be dropped.

    In effect, this operation is applied: `x = x[0, ...]`, which reduces
    `x`'s rank by 1 (e.g., 3D->2D), and discards the following elements: `x[1:, ...]`.
    """

    DROP_IF_DUMMY = "drop_if_dummy"
    """Drop the first dimension only if it's dummy (length: 1)."""

    KEEP = "keep"
    """Always preserve the first dimension."""


class InternalFeatureSpec(NamedTuple):
    name: str
    """Feature name."""

    required: bool = False
    """Whether the feature must be always present (vs optional)."""

    variable_shape: bool = False
    """Whether the feature can be of variable shape.

    For example, `input_ids` is normally of variable length.
    """

    first_dim_action: InternalFeatureFirstDimAction = (
        InternalFeatureFirstDimAction.DROP_ALWAYS
    )
    """Action to apply to the first feature dimension."""

    image_dependent: bool = False
    """Whether the feature depends on image data.

    For example, `pixel_values`, `cross_attention_mask`.
    """


@dataclass
class InternalVisualModelConfig(BaseConfig):
    main_image_feature: str = "pixel_values"
    """The key corresponding to the main image feature consumed by the model.

    E.g., raw pixels, transformed image patches, etc. resulting from data
    preprocessing and consumed by the underlying model."""

    variable_shape_image_features: bool = False
    """Whether image features can be of variable shape.

    In this case, the image features can be difficult to collate
    (`torch.stack()` requires compatible shapes) and some workaround
    is needed: either require `batch_size=1`, or group examples
    so that each mini-batch only contains same-sized features.
    """

    supports_multiple_images: bool = False
    """Whether the visual language model supports multiple images in one prompt."""


def _default_model_input_features_factory() -> dict[str, InternalFeatureSpec]:
    result_list: list[InternalFeatureSpec] = [
        InternalFeatureSpec(name="input_ids", required=True, variable_shape=True),
        InternalFeatureSpec(name="attention_mask", required=False, variable_shape=True),
        InternalFeatureSpec(name="labels", required=False, variable_shape=True),
    ]
    return {x.name: x for x in result_list}


@dataclass
class InternalModelConfig(BaseConfig):
    model_type: str = ""
    """Model type."""

    chat_template: str = "default"
    """Default chat template."""

    tokenizer_pad_token: Optional[str] = None
    """The default padding token used by the tokenizer.

    If specified in internal model type config and unspecified
    in `ModelParams.tokenizer_pad_token`, then this value will be used.
    """

    padding_side: Optional[InternalPaddingSide] = None
    """Padding side for the model."""

    model_input_features: dict[str, InternalFeatureSpec] = field(
        default_factory=_default_model_input_features_factory
    )
    """Model input features specs."""

    label_ignore_index: Optional[int] = LABEL_IGNORE_INDEX
    """Special label value to be excluded from loss computation."""

    sanitize_negative_labels: bool = False
    """Replace negative label values.

    Some VLM processors can generate negative `input_ids` for image tokens,
    which can cause problems if a negative integer is used as a label
    to compute loss e.g., cross-entropy loss may expect [0, num_classes) range.
    """

    processor_kwargs: dict[str, Any] = field(default_factory=dict)
    """Extra params to pass to processor constructor."""

    ignore_features: list[str] = field(default_factory=list)
    """Features from processing the input to ignore in the model's forward method."""

    visual_config: Optional[InternalVisualModelConfig] = None
    """Configuration specific to visual models."""
