from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from omegaconf import MISSING
from transformers.utils import is_flash_attn_2_available

from oumi.core.configs.params.base_params import BaseParams
from oumi.core.types.exceptions import HardwareException


@dataclass
class ModelParams(BaseParams):
    model_name: str = MISSING
    """The name or path of the model to use.

    This can be a model identifier from the Oumi registry, Hugging Face model hub,
    or a path to a local directory containing model files.
    """

    adapter_model: Optional[str] = None
    """The path to an adapter model to be applied on top of the base model.

    If provided, this adapter will be loaded and applied to the base model.
    """

    tokenizer_name: Optional[str] = None
    """The name or path of the tokenizer to use.

    If None, the tokenizer associated with `model_name` will be used.
    Specify this if you want to use a different tokenizer than the default
    for the model.
    """

    model_max_length: Optional[int] = None
    """The maximum sequence length the model can handle.

    If specified, this will override the default max length of the model's config.

    Note:
        Setting this to a larger value may increase memory usage but allow for
        processing longer inputs. Ensure your hardware can support the chosen
        length.
    """

    load_pretrained_weights: bool = True
    """Whether to load the pretrained model's weights.

    If True, the model will be initialized with pretrained weights.
    If False, the model will be initialized from the pretrained config without loading
    weights.
    """

    trust_remote_code: bool = False
    """Whether to allow loading remote code when loading the model.

    If True, this allows loading and executing code from the model's repository,
    which can be a security risk. Only set to True for models you trust.

    Defaults to False for safety.
    """

    torch_dtype_str: str = "float32"
    """The data type to use for the model's parameters as a string.

    Valid options are:
    - "float32" or "f32" for 32-bit floating point
    - "float16" or "f16" for 16-bit floating point
    - "bfloat16" or "bf16" for brain floating point
    - "float64" or "f64" for 64-bit floating point

    This string will be converted to the corresponding torch.dtype.
    Defaults to "float32" for full precision.
    """

    compile: bool = False
    """Whether to JIT compile the model.

    For training, do not set this param, and instead set `TrainingParams.compile`.
    """

    chat_template: Optional[str] = None
    """The chat template to use for formatting inputs.

    If provided, this template will be used to format multi-turn conversations
    for models that support chat-like interactions.

    Note:
        Different models may require specific chat templates. Consult the model's
        documentation for the appropriate template to use.
    """

    attn_implementation: Optional[str] = None
    """The attention implementation to use.

    Valid options include:
    - None: Use the default attention implementation (spda for torch>=2.1.1, else eager)
    - "sdpa": Use PyTorch's scaled dot-product attention
    - "flash_attention_2": Use Flash Attention 2 for potentially faster computation.
      Requires "flash-attn" package to be installed
    - "eager": Manual implementation of attention
    """

    device_map: Optional[str] = "auto"
    """Specifies how to distribute the model's layers across available devices.

    - "auto": Automatically distribute the model across available devices
    - None: Load the entire model on the default device

    Note:
        "auto" is generally recommended as it optimizes device usage,
        especially for large models that don't fit on a single GPU.
    """

    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments to pass to the model's constructor.

    This allows for passing any model-specific parameters that are not
    covered by other fields in ModelParams.

    Note:
        Use this for model-specific parameters or to enable experimental features.
    """

    enable_liger_kernel: bool = False
    """Whether to enable the Liger kernel for potential performance improvements.

    Liger is an optimized CUDA kernel that can accelerate certain operations.

    Tip:
        Enabling this may improve performance, but ensure compatibility with your
        model and hardware before use in production.
    """

    shard_for_eval: bool = False
    """Whether to shard the model for evaluation.

    This is needed for large models that do not fit on a single GPU.
    It is used as the value for the `parallelize` argument in LM Harness.
    """

    freeze_layers: List[str] = field(default_factory=list)
    """A list of layer names to freeze during training.

    These layers will have their parameters set to not require gradients,
    effectively preventing them from being updated during the training process.
    This is useful for fine-tuning specific parts of a model while keeping
    other parts fixed.
    """

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
        """Converts Oumi's ModelParams to LM Harness model arguments."""
        model_args_dict = {
            "pretrained": self.model_name,
            "trust_remote_code": self.trust_remote_code,
            "parallelize": self.shard_for_eval,
        }
        if self.adapter_model:
            model_args_dict["peft"] = self.adapter_model
        if self.attn_implementation:
            model_args_dict["attn_implementation"] = self.attn_implementation
        return model_args_dict

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
