from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
from omegaconf import MISSING
from transformers.utils import is_flash_attn_2_available

from lema.core.types.exceptions import HardwareException
from lema.core.types.params.base_params import BaseParams
from lema.utils.logging import logger


@dataclass
class ModelParams(BaseParams):
    model_name: str = MISSING
    adapter_model: Optional[str] = None
    tokenizer_name: Optional[str] = None
    model_max_length: Optional[int] = None
    #: Whether to load the pretrained model's weights. Else, the model will be
    #: initialized from the pretrained config.
    load_pretrained_weights: bool = True
    trust_remote_code: bool = False
    torch_dtype_str: str = "float32"
    #: Whether to JIT compile the model. For training, do not set this param, and
    #: instead set `TrainingParams.compile`.
    compile: bool = False
    chat_template: Optional[str] = None
    attn_implementation: Optional[str] = None
    device_map: Optional[str] = "auto"
    model_kwargs: Dict[str, Any] = field(default_factory=dict)

    def torch_dtype(self):
        """Converts string dtype to torch.dtype."""
        if self.torch_dtype_str in ["f64", "float64", "double"]:
            return torch.float64
        elif self.torch_dtype_str in ["f32", "float32", "float"]:
            return torch.float32
        elif self.torch_dtype_str in ["bf16", "bfloat16"]:
            return torch.bfloat16
        elif self.torch_dtype_str in ["f16", "float16", "half"]:
            return torch.float16
        else:
            raise ValueError(f"Unsupported data type: {self.torch_dtype_str}")

    def to_lm_harness(self) -> Dict[str, Any]:
        """Converts LeMa's ModelParams to LM Harness model arguments."""
        model_args_dict = {
            "pretrained": self.model_name,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.attn_implementation:
            model_args_dict["attn_implementation"] = self.attn_implementation
        return model_args_dict

    def __post_init__(self):
        """Verifies params immediately after initialization."""
        # Check if flash-attention-2 is requested with half-precision
        if (self.attn_implementation == "flash_attention_2") and (
            self.torch_dtype() not in [torch.bfloat16, torch.float16]
        ):
            logger.warning(
                "Cannot use flash_attention_2 with a full-precision "
                f"({self.torch_dtype()}) model. Ignoring request for using "
                "flash_attention_2 by setting attn_implementation system's default."
            )
            self.attn_implementation = None

    def __validate__(self):
        """Validates final config params."""
        # Check if flash-attention-2 is requested and supported
        if (self.attn_implementation == "flash_attention_2") and (
            not is_flash_attn_2_available()
        ):
            raise HardwareException(
                "Flash attention 2 was requested but it is not "
                "supported. Confirm that your hardware is compatible and then "
                "consider installing it: pip install -U flash-attn --no-build-isolation"
            )

    @property
    def should_use_flash_attention_2(self) -> bool:
        """Checks if flash-attention-2 was requested.

        Note: Flash attention 2 paper https://arxiv.org/abs/2307.08691
        TODO add flash-attention-2 in optional dependencies if we want to
        use it frequently (.toml).
        """
        return self.attn_implementation == "flash_attention_2"
