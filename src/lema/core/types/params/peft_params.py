from dataclasses import dataclass, field
from typing import List, Optional

from peft.utils.peft_types import TaskType

from lema.core.types.params.base_params import BaseParams


@dataclass
class PeftParams(BaseParams):
    # Lora Params
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA R value."},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha."},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout."},
    )
    lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": "LoRA target modules."},
    )
    lora_modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Model layers to unfreeze and train."},
    )
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

    lora_task_type: TaskType = TaskType.CAUSAL_LM

    # Q-Lora Params
    q_lora: bool = field(default=False, metadata={"help": "Use model quantization."})
    q_lora_bits: int = field(
        default=4, metadata={"help": "Quantization (precision) bits."}
    )
    # FIXME the names below use the bnb short for bits-and bytes
    # If we consider wrapping more quantization libraries a better
    # naming convention should be applied.
    bnb_4bit_quant_type: str = field(
        default="fp4", metadata={"help": "4-bit quantization type (fp4 or nf4)."}
    )
    use_bnb_nested_quant: bool = field(
        default=False, metadata={"help": "Use nested quantization."}
    )
    bnb_4bit_quant_storage: str = field(
        default="uint8",
        metadata={"help": "Storage type to pack the quanitzed 4-bit prarams."},
    )
