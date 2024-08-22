from dataclasses import dataclass
from typing import Optional

from lema.core.configs.base_config import BaseConfig


@dataclass
class GenerationConfig(BaseConfig):
    # TODO: Add more parameters to control text generation.
    max_new_tokens: int = 256
    batch_size: int = 2
    input_filepath: Optional[str] = None
    output_filepath: Optional[str] = None
