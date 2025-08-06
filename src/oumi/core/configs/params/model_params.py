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

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

from omegaconf import MISSING
from transformers.utils import find_adapter_config_file, is_flash_attn_2_available

from oumi.core.configs.params.base_params import BaseParams
from oumi.core.types.exceptions import HardwareException
from oumi.utils.logging import logger
from oumi.utils.torch_utils import get_torch_dtype


@dataclass
class ModelParams(BaseParams):
    model_name: str = MISSING
    """The name or path of the model or LoRA adapter to use.

    This can be a model identifier from the Oumi registry, HuggingFace Hub,
    or a path to a local directory containing model files.

    The LoRA adapter can be specified here instead of in `adapter_model`. If so, this
    value is copied to `adapter_model`, and the appropriate base model is set here
    instead. The base model could either be in the same directory as the adapter, or
    specified in the adapter's config file.
    """

    adapter_model: Optional[str] = None
    """The path to an adapter model to be applied on top of the base model.

    If provided, this adapter will be loaded and applied to the base model. The
    adapter path could alternatively be specified in `model_name`.
    """

    tokenizer_name: Optional[str] = None
    """The name or path of the tokenizer to use.

    If None, the tokenizer associated with `model_name` will be used.
    Specify this if you want to use a different tokenizer than the default
    for the model.
    """

    tokenizer_pad_token: Optional[str] = None
    """The padding token used by the tokenizer.

    If this is set, it will override the default padding token of the tokenizer and the
    padding token optionally defined in the `tokenizer_kwargs`.
    """

    tokenizer_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments to pass into the tokenizer's constructor.

    This allows for passing any tokenizer-specific parameters that are not
    covered by other fields in ModelParams.
    """

    processor_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments to pass into the processor's constructor.

    Processors are used in Oumi for vision-language models to process image and
    text inputs. This field is optional and can be left empty for text-only models,
    or if not needed.

    These params override model-specific default values for these kwargs, if present.
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
    - "float32" or "f32" or "float" for 32-bit floating point
    - "float16" or "f16" or "half" for 16-bit floating point
    - "bfloat16" or "bf16" for brain floating point
    - "float64" or "f64" or "double" for 64-bit floating point

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

    model_kwargs: dict[str, Any] = field(default_factory=dict)
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

    freeze_layers: list[str] = field(default_factory=list)
    """A list of layer names to freeze during training.

    These layers will have their parameters set to not require gradients,
    effectively preventing them from being updated during the training process.
    This is useful for fine-tuning specific parts of a model while keeping
    other parts fixed.
    """

    model_revision: Optional[str] = None
    """The revision of the model to use.

    This is used to specify the version of the model to use.
    """

    def __post_init__(self):
        """Populate additional params."""
        self.torch_dtype = get_torch_dtype(self.torch_dtype_str)

        if len(self.processor_kwargs) > 0:
            conflicting_keys = {f.name for f in fields(self)}.intersection(
                self.processor_kwargs.keys()
            )
            if len(conflicting_keys) > 0:
                raise ValueError(
                    "processor_kwargs attempts to override the following "
                    f"reserved fields: {conflicting_keys}. "
                    "Use properties of ModelParams instead."
                )

        if "revision" in self.model_kwargs:
            logger.warning(
                "`revision` is deprecated. Use `model_revision` instead. "
                "This will be removed in a future version."
            )
            self.model_revision = self.model_kwargs.pop("revision")

    def __finalize_and_validate__(self):
        """Finalizes and validates final config params."""
        # If the user didn't specify a LoRA adapter, check to see if the dir/repo
        # specified by `model_name` contains an adapter, and set `adapter_name` if so.
        if self.adapter_model is None:
            # This is a HF utility function that tries to find `adapter_config.json`
            # given either a local dir or a HF Hub repo id. In the latter case, the repo
            # will be downloaded from HF Hub if it's not already cached.
            try:
                adapter_config_file = find_adapter_config_file(self.model_name)
            except OSError:
                logger.debug(
                    f"Model folder does not contain an adapter: {self.model_name}"
                )
                adapter_config_file = None
            # If this check fails, it means this is not a LoRA model.
            if adapter_config_file:
                # If `model_name` is a local dir, this should be the same.
                # If it's a HF Hub repo, this should be the path to the cached repo.
                adapter_dir = Path(adapter_config_file).parent
                self.adapter_model = self.model_name
                logger.info(
                    f"Found LoRA adapter at {adapter_dir}, "
                    "setting `adapter_model` to `model_name`."
                )
                # If `model_name` specifies a LoRA adapter dir without the base model
                # present, set it to the base model name found in the adapter config,
                # if present. Error otherwise.
                if len(list(adapter_dir.glob("config.json"))) == 0:
                    with open(adapter_config_file) as f:
                        adapter_config = json.load(f)
                    model_name = adapter_config.get("base_model_name_or_path")
                    if not model_name:
                        raise ValueError(
                            "`model_name` specifies an adapter model only,"
                            " but the base model could not be found!"
                        )
                    self.model_name = model_name
                    logger.info(
                        f"Setting `model_name` to {model_name} found in adapter config."
                    )

        # Check if flash-attention-2 is requested and supported
        if (self.attn_implementation == "flash_attention_2") and (
            not is_flash_attn_2_available()
        ):
            raise HardwareException(
                "Flash attention 2 was requested but it is not "
                "supported. Confirm that your hardware is compatible and then "
                "consider installing it: pip install -U flash-attn --no-build-isolation"
            )

        if self.model_max_length is not None and self.model_max_length <= 0:
            raise ValueError("model_max_length must be a positive integer or None.")
