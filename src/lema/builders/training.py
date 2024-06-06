from typing import Callable

from transformers import Trainer
from trl import DPOTrainer, SFTTrainer

from lema.core.types import TrainerType


def build_trainer(trainer_type: TrainerType) -> Callable[..., Trainer]:
    """Builds a trainer creator functor based on the provided configuration.

    Args:
        trainer_type (TrainerType): Enum indicating the type of training.

    Returns:
        A builder function that can create an appropriate trainer based on the trainer
        type specified in the configuration. All function arguments supplied by caller
        are forwarded to the trainer's constructor.

    Raises:
        NotImplementedError: If the trainer type specified in the
            configuration is not supported.
    """
    if trainer_type == TrainerType.TRL_SFT:
        return lambda *args, **kwargs: SFTTrainer(*args, **kwargs)

    elif trainer_type == TrainerType.TRL_DPO:
        return lambda *args, **kwargs: DPOTrainer(*args, **kwargs)

    elif trainer_type == TrainerType.HF:
        return lambda *args, **kwargs: Trainer(*args, **kwargs)

    raise NotImplementedError(f"Trainer type {trainer_type} not supported.")


def print_trainable_parameters(model):
    """Prints the number of trainable parameters in the model.

    Note: original code:
    https://github.com/huggingface/peft/blob/main/examples/fp4_finetuning/finetune_fp4_opt_bnb_peft.py

    TODO: Use it with our logger.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        (
            f"Trainable params: {trainable_params} || All params: {all_param} "
            f"|| Trainable%: {100 * trainable_params / all_param :.4f}"
        )
    )
