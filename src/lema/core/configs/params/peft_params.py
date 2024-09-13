from dataclasses import dataclass, field
from typing import List, Optional

from peft.utils.peft_types import TaskType

from lema.core.configs.params.base_params import BaseParams


@dataclass
class PeftParams(BaseParams):
    # Lora Params
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA R value."},
    )
    """The rank of the update matrices in LoRA.

    A higher value allows for more expressive adaptations but increases
    the number of trainable parameters.
    """

    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha."},
    )
    """The scaling factor for the LoRA update.

    This value is typically set equal to `lora_r` for stable training.
    """

    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout."},
    )
    """The dropout probability applied to LoRA layers.

    This helps prevent overfitting in the adaptation layers.
    """

    lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": "LoRA target modules."},
    )
    """List of module names to apply LoRA to.

    If None, LoRA will be applied to all linear layers in the model.
    Specify module names to selectively apply LoRA to certain parts of the model.
    """

    lora_modules_to_save: Optional[List[str]] = field(
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

    Defaults to 4-bit quantization.
    """

    # FIXME the names below use the bnb short for bits-and bytes
    # If we consider wrapping more quantization libraries a better
    # naming convention should be applied.
    bnb_4bit_quant_type: str = field(
        default="fp4", metadata={"help": "4-bit quantization type (fp4 or nf4)."}
    )
    """The type of 4-bit quantization to use.

    Can be 'fp4' (float point 4) or 'nf4' (normal float 4).
    """

    use_bnb_nested_quant: bool = field(
        default=False, metadata={"help": "Use nested quantization."}
    )
    """Whether to use nested quantization.

    Nested quantization can provide additional memory savings.
    """

    bnb_4bit_quant_storage: str = field(
        default="uint8",
        metadata={"help": "Storage type to pack the quanitzed 4-bit prarams."},
    )
    """The storage type for packing quantized 4-bit parameters.

    Defaults to 'uint8' for efficient storage.
    """
