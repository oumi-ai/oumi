from dataclasses import dataclass
from typing import Optional

from oumi.core.configs.params.base_params import BaseParams


@dataclass
class DataCollationParams(BaseParams):
    """Parameters controlling how to form a batch from individual dataset elements.

    For example, input elements can be of variable length, and data collation config
    defines how to pad and combine them into a mini-batch for training.
    """

    collator_name: Optional[str] = None
    """Name of Oumi data collator.

    Valid options are:
    - "text_with_padding": Uses DataCollatorWithPadding for text data.
    - "vision_language": Uses VisionLanguageCollator for image+text multi-modal data.

    If None, then a default collator will be assigned.
    """
