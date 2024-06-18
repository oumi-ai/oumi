import os.path as osp
from typing import Optional

import transformers
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

# FIXME: The following import is NOT used, but is needed to populate the registry.
import lema.core.models  # noqa: F401
from lema.core.registry import REGISTRY
from lema.core.types import ModelParams, PeftParams
from lema.logging import logger
from lema.utils.torch_utils import get_device_rank_info


def build_model(
    model_params: ModelParams,
    peft_params: Optional[PeftParams] = None,
    **kwargs,
):
    """Builds and returns a model based on the provided LeMa configuration.

    Args:
        model_params: The configuration object containing the model parameters.
        peft_params: The configuration object containing the peft parameters.
        kwargs (dict, optional): Additional keyword arguments for model loading.

    Returns:
        model: The built model.
    """
    custom_model_in_registry = REGISTRY.get_model(
        name=model_params.model_name, except_if_missing=False
    )
    if custom_model_in_registry:
        return build_custom_model(custom_model_in_registry)
    else:
        return build_huggingface_model(
            model_params=model_params,
            peft_params=peft_params,
            *kwargs,
        )


def build_custom_model(custom_model_in_registry):
    """Builds a custom model from our LeMa registry."""
    model_config = custom_model_in_registry.model_config
    model_class = custom_model_in_registry.model_class
    model = model_class(model_config())

    return model


def build_huggingface_model(
    model_params: ModelParams,
    peft_params: Optional[PeftParams] = None,
    **kwargs,
):
    """Downloads and builds the model from the HuggingFace Hub."""
    # TODO: add device_map to config
    device_map = "auto"
    device_rank_info = get_device_rank_info()
    # "auto" is not compatible with distributed training.
    if device_rank_info.world_size > 1:
        logger.info(
            f"Building model for distributed training "
            f"(world_size: {device_rank_info.world_size})..."
        )
        device_map = f"cuda:{device_rank_info.local_rank}"
    logger.info(
        f"Building model using device_map: {device_map} ({device_rank_info})..."
    )

    hf_config = transformers.AutoConfig.from_pretrained(
        model_params.model_name,
        trust_remote_code=model_params.trust_remote_code,
        flash_attention_2=model_params.should_use_flash_attention_2,
    )

    if peft_params and peft_params.q_lora:
        # TODO confirm bnb_4bit_compute_dtype must be model_params.torch_dtype always
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=peft_params.q_lora_bits == 4,
            load_in_8bit=peft_params.q_lora_bits == 8,
            bnb_4bit_compute_dtype=model_params.torch_dtype(),
        )
    else:
        quantization_config = None

    model = transformers.AutoModelForCausalLM.from_pretrained(
        config=hf_config,
        torch_dtype=model_params.torch_dtype(),
        device_map=device_map,
        pretrained_model_name_or_path=model_params.model_name,
        trust_remote_code=model_params.trust_remote_code,
        quantization_config=quantization_config,
        **kwargs,
    )

    return model


def build_tokenizer(model_params: ModelParams, **kwargs):
    """Builds and returns a tokenizer based on the provided LeMa configuration.

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
        logger.warning("<pad> token not found: setting <pad> with <eos>.")
        tokenizer.pad_token = tokenizer.eos_token

    if model_params.model_max_length:
        tokenizer.model_max_length = model_params.model_max_length

    if model_params.chat_template:
        tokenizer.chat_template = build_chat_template(model_params.chat_template)

    return tokenizer


def build_peft_model(
    base_model, use_gradient_checkpointing: bool, peft_params: PeftParams
):
    """Builds a PEFT model based on the given base model and params.

    Args:
        base_model: The base model to build the PEFT model on.
        use_gradient_checkpointing: Enable/disable gradient checkpointing.
        peft_params: The desired params for LORA.

    Returns:
        The built PEFT model.
    """
    lora_config = LoraConfig(
        r=peft_params.lora_r,
        lora_alpha=peft_params.lora_alpha,
        lora_dropout=peft_params.lora_dropout,
        target_modules=peft_params.lora_target_modules,
        modules_to_save=peft_params.lora_modules_to_save,
        bias=peft_params.lora_bias,  # type: ignore
        task_type=peft_params.lora_task_type,
    )

    if peft_params.q_lora:
        model = prepare_model_for_kbit_training(
            model=base_model,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
    else:
        model = base_model

    model = get_peft_model(model, lora_config)

    return model


def build_chat_template(template_name: str) -> str:
    """Builds a chat template based on code name.

    NOTE: (internal) This registry is experimental and will be formatted
    better once we have explored chat-template uses/cases (e.g., enumerate,
    use .ninja files like them https://github.com/chujiezheng/chat_templates/tree/main
    , etc.)

    Args:
        template_name (str): the code name describing the chat-template.

    Raises:
        NotImplementedError: if the requested code name does not exist
        in the registry.

    Returns:
        str: a ninja-based chat-template.
    """
    lema_top_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
    chat_template_directory = osp.join(lema_top_dir, "datasets/chat_templates")

    if template_name.lower() == "zephyr":
        chat_template_file = osp.join(chat_template_directory, "zephyr.jinja")
        with open(chat_template_file) as in_file:
            chat_template = in_file.read()
        chat_template = chat_template.replace("    ", "").replace("\n", "")
        return chat_template
    else:
        raise NotImplementedError(
            "Currently only *experimental* template for Zephyr has been added."
        )
