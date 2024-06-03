import os
from typing import Union

import transformers
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import GPTQConfig

# FIXME: The following import is NOT used, but is needed to populate the registry.
import lema.core.models  # noqa: F401
from lema.core.registry import REGISTRY
from lema.core.types import InferenceConfig, ModelParams, PeftParams, TrainingConfig
from lema.logging import logger


def build_model(config: Union[TrainingConfig, InferenceConfig], **kwargs):
    """Build and return a model based on the provided LeMa configuration.

    Args:
        config: The configuration object containing model config.
        kwargs (dict, optional): Additional keyword arguments for model loading.

    Returns:
        model: The built model.
    """
    custom_model_in_registry = REGISTRY.lookup_model(config.model.model_name)
    if custom_model_in_registry:
        return build_custom_model(custom_model_in_registry)
    else:
        return build_huggingface_model(config, *kwargs)


def build_custom_model(custom_model_in_registry):
    """Build a custom model from our LeMa registry."""
    model_config = custom_model_in_registry.model_config
    model_class = custom_model_in_registry.model_class
    model = model_class(model_config())

    return model


def build_huggingface_model(config: Union[TrainingConfig, InferenceConfig], **kwargs):
    """Download and build the model from the HuggingFace Hub."""
    # TODO: add device_map to config
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    # "auto" is not compatible with distributed training.
    if world_size > 1:
        logger.info(
            f"Building model for distributed training (world_size: {world_size})..."
        )
        device_map = "cuda"
    logger.info(f"Building model using device_map: {device_map}...")

    hf_config = transformers.AutoConfig.from_pretrained(
        config.model.model_name,
        trust_remote_code=config.model.trust_remote_code,
    )

    if (
        isinstance(config, TrainingConfig)
        and config.training.use_peft
        and config.peft.q_lora
    ):
        quantization_config = GPTQConfig(
            bits=config.peft.q_lora_bits, disable_exllama=True
        )
    else:
        quantization_config = None

    model = transformers.AutoModelForCausalLM.from_pretrained(
        config=hf_config,
        torch_dtype=config.model.torch_dtype(),
        device_map=device_map,
        pretrained_model_name_or_path=config.model.model_name,
        trust_remote_code=config.model.trust_remote_code,
        quantization_config=quantization_config,
        **kwargs,
    )

    return model


def build_tokenizer(model_params: ModelParams, **kwargs):
    """Build and return a tokenizer based on the provided LeMa configuration.

    Args:
        model_params (ModelParams): The configuration object containing
            the model parameters.
        **kwargs: Additional keyword arguments for tokenizer loading.

    Returns:
        tokenizer: The tokenizer object built from the configuration.
    """
    # Identify the tokenizer we need to leverage for this model.
    if model_params.tokenizer_name:
        tokenizer_name = model_params.tokenizer_name
    else:
        # If no specific tokenizer is defined, fall back to model's default.
        tokenizer_name = model_params.model_name

    # Download and build the tokenizer from the HuggingFace Hub.
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=model_params.trust_remote_code,
        **kwargs,
    )

    if tokenizer.pad_token is None:
        # Set pad token to eos token if not already set
        # Older models may not have pad token set
        # TODO: should log a warning here
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def build_peft_model(
    base_model, use_gradient_checkpointing: bool, peft_config: PeftParams
):
    """Build a PEFT model based on the given base model and configuration.

    Args:
        base_model: The base model to build the PEFT model on.
        use_gradient_checkpointing: Enable/disable gradient checkpointing.
        peft_config: The desired configuration for LORA.

    Returns:
        The built PEFT model.
    """
    lora_config = LoraConfig(
        r=peft_config.lora_r,
        lora_alpha=peft_config.lora_alpha,
        lora_dropout=peft_config.lora_dropout,
        target_modules=peft_config.lora_target_modules,
        bias=peft_config.lora_bias,  # type: ignore
        task_type=peft_config.lora_task_type,
    )

    if peft_config.q_lora:
        model = prepare_model_for_kbit_training(
            model=base_model,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
    else:
        model = base_model

    model = get_peft_model(model, lora_config)

    return model
