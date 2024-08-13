import os.path as osp
from typing import Optional, Union, cast

import torch
import torch.nn as nn
import transformers
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

from lema.core.distributed import get_device_rank_info
from lema.core.registry import REGISTRY, RegistryType
from lema.core.types import ModelParams, PeftParams
from lema.utils.logging import logger


def _disable_dropout(hf_config: transformers.AutoConfig) -> None:
    """Detects dropout probabilities in config and sets them to 0.0.

    This essentially removes the dropout layer, which can aid the compiled model's
    speed. Dropout is normally not used for LLM training, and also hinders the
    effectiveness of model compilation. We assume any attribute with "drop" in the name
    and a float value is a dropout param. For example, this includes `attn_pdrop` and
    `summary_first_dropout` for GPT2.

    Args:
        hf_config: The HuggingFace model config.
    """
    drop_attrs = []
    for k, v in vars(hf_config).items():
        if "drop" in k and isinstance(v, float):
            setattr(hf_config, k, 0.0)
            drop_attrs.append(k)
    logger.info(
        f"Found these dropout attributes and set their values to 0.0: {drop_attrs}"
    )


def build_model(
    model_params: ModelParams,
    peft_params: Optional[PeftParams] = None,
    **kwargs,
) -> nn.Module:
    """Builds and returns a model based on the provided LeMa configuration.

    Args:
        model_params: The configuration object containing the model parameters.
        peft_params: The configuration object containing the peft parameters.
        kwargs (dict, optional): Additional keyword arguments for model loading.

    Returns:
        model: The built model.
    """
    if REGISTRY.contains(name=model_params.model_name, type=RegistryType.MODEL):
        model = build_lema_model(
            model_params=model_params,
            peft_params=peft_params,
            *kwargs,
        )
    else:
        model = build_huggingface_model(
            model_params=model_params,
            peft_params=peft_params,
            *kwargs,
        )

    if model_params.compile:
        # The output type of torch.compile is Callable, but when I test it it's of type
        # nn.Module. We cast it so that this function can have a useful return type.
        model = cast(nn.Module, torch.compile(model))
        logger.info("Enabled model compilation.")

    return model


def build_lema_model(
    model_params: ModelParams,
    peft_params: Optional[PeftParams] = None,
    **kwargs,
) -> nn.Module:
    """Builds a custom model from our LeMa registry."""
    model_class = REGISTRY[model_params.model_name, RegistryType.MODEL]
    model = model_class(**model_params.model_kwargs)

    if model_params.load_pretrained_weights:
        raise NotImplementedError

    if peft_params and peft_params.q_lora:
        raise NotImplementedError

    if model_params.adapter_model is not None:
        raise NotImplementedError

    dtype = model_params.torch_dtype()
    model = model.to(dtype=dtype)
    # Needed for MFUTrainerCallback
    model.dtype = dtype
    return model


def build_huggingface_model(
    model_params: ModelParams,
    peft_params: Optional[PeftParams] = None,
    **kwargs,
) -> nn.Module:
    """Downloads and builds the model from the HuggingFace Hub."""
    device_map = model_params.device_map
    device_rank_info = get_device_rank_info()

    # "auto" is not compatible with distributed training.
    if device_map == "auto" and device_rank_info.world_size > 1:
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

    # (Experimental) Detects dropout probabilities in config and sets them to 0.0.
    if model_params.model_kwargs.get("disable_dropout"):
        _disable_dropout(hf_config)
        del model_params.model_kwargs["disable_dropout"]

    if peft_params and peft_params.q_lora:
        # TODO confirm bnb_4bit_compute_dtype must be model_params.torch_dtype always
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=peft_params.q_lora_bits == 4,
            load_in_8bit=peft_params.q_lora_bits == 8,
            bnb_4bit_compute_dtype=model_params.torch_dtype(),
            bnb_4bit_quant_type=peft_params.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=peft_params.use_bnb_nested_quant,
            bnb_4bit_quant_storage=peft_params.bnb_4bit_quant_storage,
        )
    else:
        quantization_config = None

    # Both functions instantiate a model from the config, but the main difference is
    # `load_pretrained_weights` also loads the weights, and `from_config` initializes
    # the weights from scratch based on the params in the config and the model class.
    if model_params.load_pretrained_weights:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            config=hf_config,
            torch_dtype=model_params.torch_dtype(),
            device_map=device_map,
            pretrained_model_name_or_path=model_params.model_name,
            trust_remote_code=model_params.trust_remote_code,
            quantization_config=quantization_config,
            **kwargs,
        )
    else:
        # TODO: What about device_map and quantization_config params?
        model = transformers.AutoModelForCausalLM.from_config(
            config=hf_config,
            torch_dtype=model_params.torch_dtype(),
            trust_remote_code=model_params.trust_remote_code,
            **kwargs,
        )

    # Required for FSDP.
    # Context: https://github.com/huggingface/transformers/issues/28499
    model.config.use_cache = False

    # TODO Find a better way to handle it

    # Load pretrained PEFT adapters
    if model_params.adapter_model is not None:
        model = PeftModel.from_pretrained(model, model_params.adapter_model)

    return model


def build_tokenizer(
    model_params: ModelParams, **kwargs
) -> Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast]:
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
