from typing import Optional, Union, cast

import torch
import torch.nn as nn
import transformers
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

from oumi.core.configs import ModelParams, PeftParams
from oumi.core.distributed import get_device_rank_info, is_using_accelerate_fsdp
from oumi.core.registry import REGISTRY, RegistryType
from oumi.utils.io_utils import get_oumi_root_directory, load_file
from oumi.utils.logging import logger
from oumi.utils.torch_naming_heuristics import disable_dropout

try:
    import liger_kernel.transformers  # type: ignore
except ImportError:
    liger_kernel = None


def build_model(
    model_params: ModelParams,
    peft_params: Optional[PeftParams] = None,
    **kwargs,
) -> nn.Module:
    """Builds and returns a model based on the provided Oumi configuration.

    Args:
        model_params: The configuration object containing the model parameters.
        peft_params: The configuration object containing the peft parameters.
        kwargs (dict, optional): Additional keyword arguments for model loading.

    Returns:
        model: The built model.
    """
    if REGISTRY.contains(name=model_params.model_name, type=RegistryType.MODEL):
        model = build_oumi_model(
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

    if model_params.enable_liger_kernel:
        _patch_model_for_liger_kernel(model_params.model_name)

    for layer_name in model_params.freeze_layers:
        if hasattr(model, layer_name):
            logger.info(f"Freezing layer {layer_name}.")

            for param in getattr(model, layer_name).parameters():
                param.requires_grad_(False)
        else:
            logger.warning(f"Layer {layer_name} not found in model.")

    if model_params.compile:
        # The output type of torch.compile is Callable, but when I test it it's of type
        # nn.Module. We cast it so that this function can have a useful return type.
        model = cast(nn.Module, torch.compile(model))
        logger.info("Enabled model compilation.")

    return model


def _patch_model_for_liger_kernel(model_name: str) -> None:
    """Patches the model for Liger Kernel."""
    if liger_kernel is None:
        raise ImportError(
            "Liger Kernel not installed. Please install `pip install liger-kernel`."
        )

    model_name_lower = model_name.lower()

    if "llama" in model_name_lower:
        liger_kernel.transformers.apply_liger_kernel_to_llama()
    elif "mixtral" in model_name_lower:
        liger_kernel.transformers.apply_liger_kernel_to_mixtral()
    elif "mistral" in model_name_lower:
        liger_kernel.transformers.apply_liger_kernel_to_mistral()
    elif "gemma" in model_name_lower:
        liger_kernel.transformers.apply_liger_kernel_to_gemma()
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def build_oumi_model(
    model_params: ModelParams,
    peft_params: Optional[PeftParams] = None,
    **kwargs,
) -> nn.Module:
    """Builds a custom model from our Oumi registry."""
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

    # If we're using FSDP via HF Accelerate, we should not specify the device map
    # so that HF properly initializes the model for FSDP.
    # If we set device_map to "auto", it seems HF will try to shard the model when
    # loading it, which conflicts with FSDP's sharding.
    # If we set device_map to f"cuda:{device_rank_info.local_rank}", it will try to
    # load the model only on rank 0, which will OOM for large models.
    # See https://github.com/huggingface/transformers/pull/25107.
    if is_using_accelerate_fsdp():
        logger.info("Accelerate FSDP run detected! Setting device_map to None.")
        device_map = None
    elif device_map == "auto" and device_rank_info.world_size > 1:
        # "auto" is not compatible with DDP.
        logger.info(
            f"Building model for distributed training "
            f"(world_size: {device_rank_info.world_size})..."
        )
        device_map = f"cuda:{device_rank_info.local_rank}"
    logger.info(
        f"Building model using device_map: {device_map} ({device_rank_info})..."
    )

    hf_config, unused_kwargs = transformers.AutoConfig.from_pretrained(
        model_params.model_name,
        trust_remote_code=model_params.trust_remote_code,
        return_unused_kwargs=True,
    )
    if unused_kwargs:
        logger.warning(f"Unused kwargs found in config: {unused_kwargs}.")

    # (Experimental) Detects dropout probabilities in config and sets them to 0.0.
    if model_params.model_kwargs.get("disable_dropout"):
        disable_dropout(hf_config)
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
    transformers_model_class = _get_transformers_model_class(hf_config)

    if model_params.load_pretrained_weights:
        model = transformers_model_class.from_pretrained(
            config=hf_config,
            torch_dtype=model_params.torch_dtype(),
            device_map=device_map,
            trust_remote_code=model_params.trust_remote_code,
            pretrained_model_name_or_path=model_params.model_name,
            quantization_config=quantization_config,
            attn_implementation=model_params.attn_implementation,
            **kwargs,
        )
    else:
        model = transformers_model_class.from_config(
            config=hf_config,
            torch_dtype=model_params.torch_dtype(),
            trust_remote_code=model_params.trust_remote_code,
            attn_implementation=model_params.attn_implementation,
            **kwargs,
        )

    # Required for FSDP.
    # Context: https://github.com/huggingface/transformers/issues/28499
    model.config.use_cache = False

    # TODO Find a better way to handle it

    # Load pretrained PEFT adapters
    if model_params.adapter_model:
        logger.info(f"Loading PEFT adapter from: {model_params.adapter_model} ...")
        model = PeftModel.from_pretrained(model, model_params.adapter_model)

    return model


def _get_transformers_model_class(config):
    # TODO: Remove this once we have a better way to identify the model class
    # Or we can just ask the user to specify the model class in the config
    if config.model_type in (
        "blip-2",
        "blip",
        "chameleon",
        "idefics",
        "idefics2",
        "idefics3",
        "instructblip",
        "llava",
        "paligemma",
        "qwen2_vl",
        "vipllava",
    ):
        tested_models = {
            "blip-2",
            "llava",
        }  # TODO: OPE-353, make sure we have all models supported

        if config.model_type not in tested_models:
            logger.warning(
                f"Model type {config.model_type} not tested. "
                "Using AutoModelForCausalLM as the model class."
                "If you encounter errors, please open an issue at https://github.com/oumi-ai/oumi."
            )

        auto_model_class = transformers.AutoModelForVision2Seq
    else:
        auto_model_class = transformers.AutoModelForCausalLM
    logger.info(f"Using model class: {auto_model_class} to instantiate model.")
    return auto_model_class


def build_tokenizer(
    model_params: ModelParams, **kwargs
) -> Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast]:
    """Builds and returns a tokenizer based on the provided Oumi configuration.

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

    if tokenizer.chat_template is None:
        logger.warning(
            "No chat template found for tokenizer. "
            "Please specify a chat template using the `chat_template` field. "
            "This will be required in future versions of Oumi."
        )

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

    Args:
        template_name: the code name describing the chat-template.

    Raises:
        FileNotFoundError: if the requested template file does not exist.

    Returns:
        str: a jinja-based chat-template.
    """
    chat_template_directory = get_oumi_root_directory() / "datasets" / "chat_templates"

    template_file = f"{template_name.lower()}.jinja"
    chat_template_file = chat_template_directory / template_file

    if not chat_template_file.exists():
        existing_templates = [f.stem for f in chat_template_directory.glob("*.jinja")]
        error_message = (
            f"Chat template file not found: {chat_template_file}\n"
            f"Existing templates: {', '.join(existing_templates)}\n"
            f"To add a new template, create a .jinja file in {chat_template_directory}"
        )
        raise FileNotFoundError(error_message)

    chat_template = load_file(chat_template_file)

    # Remove indentation and newlines while preserving intentional whitespace
    lines = chat_template.splitlines()
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    chat_template = "".join(cleaned_lines)

    return chat_template
