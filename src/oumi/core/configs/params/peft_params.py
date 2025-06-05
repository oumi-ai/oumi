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
from typing import Literal, Optional

from peft.utils.peft_types import TaskType
from transformers import BitsAndBytesConfig

from oumi.core.configs.params.base_params import BaseParams


class PeftSaveMode(Enum):
    """Enum representing how to save the final model during PEFT training.

    While models saved with any of these options can be loaded by Oumi, those saved
    with `ADAPTER_ONLY` are not self-contained; the base model will be loaded
    separately from the local HF cache or downloaded from HF Hub if not in the cache.
    """

    ADAPTER_ONLY = "adapter_only"
    """Only save the model adapter.

    Note that when loading this saved model, the base model will be loaded separately
    from the local HF cache or downloaded from HF Hub.
    """

    ADAPTER_AND_BASE_MODEL = "adapter_and_base_model"
    """Save the base model in addition to the adapter.

    This is similar to `ADAPTER_ONLY`, but the base model's weights are also saved in
    the same directory as the adapter weights, making the output dir self-contained.
    """

    MERGED = "merged"
    """Merge the adapter and base model's weights and save as a single model.

    Note that the resulting model is a standard HF Transformers model, and is no longer
    a PEFT model. A copy of the adapter before merging is saved in the "adapter/"
    subdirectory.
    """


class LoraWeightInitialization(str, Enum):
    """Enum representing the supported weight initializations for LoRA adapters."""

    DEFAULT = "default"  # Use the model reference initialization from Microsoft.
    RANDOM = "random"  # Fully random initialization, discouraged.
    GAUSSIAN = "gaussian"
    EVA = "eva"
    PISA = "pissa"
    PISSA_NITER = "pissa_niter_[number of iters]"
    LOFTQ = "loftq"
    OLORA = "olora"

    def get_literal_value(
        self,
    ) -> Literal[
        "default",
        "random",
        "gaussian",
        "eva",
        "pissa",
        "pissa_niter_[number of iters]",
        "loftq",
        "olora",
    ]:
        """Returns a literal value of the enum."""
        if self.value not in {
            "default",
            "random",
            "gaussian",
            "eva",
            "pissa",
            "pissa_niter_[number of iters]",
            "loftq",
            "olora",
        }:
            raise ValueError(f"Invalid enum value: {self.value}")

        return self.value


@dataclass
class PeftParams(BaseParams):
    # Lora Params
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA R value."},
    )
    """The rank of the update matrices in LoRA.

    A higher value allows for more expressive adaptations but increases
    the number of trainable parameters.
    """

    lora_alpha: int = field(
        default=8,
        metadata={"help": "LoRA alpha."},
    )
    """The scaling factor for the LoRA update.

    This value is typically set equal to `lora_r` or `2*lora_r` for stable training.
    """

    lora_dropout: float = field(
        default=0.0,
        metadata={"help": "LoRA dropout."},
    )
    """The dropout probability applied to LoRA layers.

    This helps prevent overfitting in the adaptation layers.
    """

    lora_target_modules: Optional[list[str]] = field(
        default=None,
        metadata={"help": "LoRA target modules."},
    )
    """List of module names to apply LoRA to.

    If None, LoRA will be applied to all linear layers in the model.
    Specify module names to selectively apply LoRA to certain parts of the model.
    """

    lora_modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={"help": "Model layers to unfreeze and train."},
    )
    """List of module names to unfreeze and train alongside LoRA parameters.

    These modules will be fully fine-tuned, not adapted using LoRA.
    Use this to selectively train certain parts of the model in full precision.
    """

    lora_bias: str = field(
        default="none",
        metadata={
            "help": (
                "Bias type for Lora. Can be 'none', 'all' or 'lora_only'. "
                "If 'all' or 'lora_only', the corresponding biases will "
                "be updated during training. Be aware that this means that, "
                "even when disabling the adapters, the model will not "
                "produce the same output as the base model would have "
                "without adaptation."
                "NOTE: see: "
                "https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/config.py"
                "for more details."
            )
        },
    )
    """Bias type for LoRA.

    Can be 'none', 'all' or 'lora_only':
    - 'none': No biases are trained.
    - 'all': All biases in the model are trained.
    - 'lora_only': Only biases in LoRA layers are trained.

    If 'all' or 'lora_only', the corresponding biases will be updated during training.
    Note that this means even when disabling the adapters, the model will not produce
    the same output as the base model would have without adaptation.

    For more details, see:
    https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/config.py
    """

    lora_init_weights: LoraWeightInitialization = field(
        default=LoraWeightInitialization.DEFAULT,
        metadata={
            "help": "Weights initialization for LoRA adapters.",
        },
    )
    """
    Passing `LoraWeightInitialization.DEFAULT` will use the underlying reference
    implementation of the corresponding model from Microsoft.

    Other valid (LoraWeightInitialization) options include:
        - "random" which will use fully random initialization and is discouraged.
        - "gaussian" for Gaussian initialization.
        - "eva" for Explained Variance Adaptation (EVA) (https://arxiv.org/abs/2410.07170).
        - "loftq" for improved performance when LoRA is combined with with quantization (https://arxiv.org/abs/2310.08659).
        - "olora" for Orthonormal Low-Rank Adaptation of Large Language Models (OLoRA) (https://arxiv.org/html/2406.01775v1).
        - "pissa" for Principal Singular values and Singular vectors Adaptation (PiSSA) (https://arxiv.org/abs/2404.02948).

    For more information, see HF:
        https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/config.py
    """

    lora_task_type: TaskType = TaskType.CAUSAL_LM
    """The task type for LoRA adaptation.

    Defaults to CAUSAL_LM (Causal Language Modeling).
    """

    # Q-Lora Params
    q_lora: bool = field(default=False, metadata={"help": "Use model quantization."})
    """Whether to use quantization for LoRA (Q-LoRA).

    If True, enables quantization for more memory-efficient fine-tuning.
    """

    q_lora_bits: int = field(
        default=4, metadata={"help": "Quantization (precision) bits."}
    )
    """The number of bits to use for quantization in Q-LoRA.

    This is only used if `q_lora` is True.

    Defaults to 4-bit quantization.
    """

    # FIXME the names below use the bnb short for bits-and-bytes
    # If we consider wrapping more quantization libraries a better
    # naming convention should be applied.
    bnb_4bit_quant_type: str = field(
        default="fp4", metadata={"help": "4-bit quantization type (fp4 or nf4)."}
    )
    """The type of 4-bit quantization to use.

    Can be 'fp4' (float point 4) or 'nf4' (normal float 4).
    """

    llm_int8_skip_modules: Optional[list[str]] = field(
        default=None,
        metadata={"help": "Modules that we do not want to convert in 8-bit"},
    )
    """ An explicit list of the modules that we do not want to convert in 8-bit.

    """

    use_bnb_nested_quant: bool = field(
        default=False, metadata={"help": "Use nested quantization."}
    )
    """Whether to use nested quantization.

    Nested quantization can provide additional memory savings.
    """

    bnb_4bit_quant_storage: str = field(
        default="uint8",
        metadata={"help": "Storage type to pack the quantized 4-bit params."},
    )
    """The storage type for packing quantized 4-bit parameters.

    Defaults to 'uint8' for efficient storage.
    """

    bnb_4bit_compute_dtype: str = field(
        default="float32",
        metadata={"help": "The compute type of the quantized parameters."},
    )
    """Compute type of the quantized parameters.
    It can be different than the input type, e.g., it can be set to a lower precision
    for improved speed.

    The string will be converted to the corresponding torch.dtype.

    Valid string options are:
    - "float32" for 32-bit floating point
    - "float16" for 16-bit floating point
    - "bfloat16" for brain floating point
    - "float64" for 64-bit floating point

    Defaults to "float16" for half precision.
    """

    peft_save_mode: PeftSaveMode = PeftSaveMode.ADAPTER_ONLY
    """How to save the final model during PEFT training.

    This option is only used if `TrainingParams.save_final_model` is True.
    By default, only the model adapter is saved to reduce disk usage.
    Options are defined in the `PeftSaveMode` enum and include:
    - ADAPTER_ONLY: Only save the model adapter.
    - ADAPTER_AND_BASE_MODEL: Save the base model in addition to the adapter.
    - MERGED: Merge the adapter and base model's weights and save as a single model.
    """

    def to_bits_and_bytes(self) -> BitsAndBytesConfig:
        """Creates a configuration for quantized models via BitsAndBytes.

        The resulting configuration uses the instantiated peft parameters.
        """
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=self.q_lora_bits == 4,
            load_in_8bit=self.q_lora_bits == 8,
            llm_int8_skip_modules=self.llm_int8_skip_modules,
            bnb_4bit_compute_dtype=self.bnb_4bit_compute_dtype,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.use_bnb_nested_quant,
            bnb_4bit_quant_storage=self.bnb_4bit_quant_storage,
        )
        return quantization_config
