from typing import Union

import transformers
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import GPTQConfig

from lema.core.types import InferenceConfig, TrainingConfig


def build_model(config: Union[TrainingConfig, InferenceConfig], **kwargs):
    """Build and return a model based on the provided LeMa configuration.

    # TODO: add ability to load model from lema registry

    Args:
        config: The configuration object containing model config.
        kwargs (dict, optional): Additional keyword arguments for model loading.

    Returns:
        model: The built model.
    """
    # TODO: add device_map to config
    device_map = "auto"

    #
    # Load from huggingface hub
    #
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


def build_tokenizer(config: Union[TrainingConfig, InferenceConfig], **kwargs):
    """Build and return a tokenizer based on the provided LeMa configuration.

    TODO: add ability to load tokenizer from lema registry

    Args:
        config: The configuration object containing model config.
        **kwargs: Additional keyword arguments for tokenizer loading.

    Returns:
        tokenizer: The tokenizer object built from the configuration.

    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model.model_name,
        trust_remote_code=config.model.trust_remote_code,
        **kwargs,
    )

    if tokenizer.pad_token is None:
        # Set pad token to eos token if not already set
        # Older models may not have pad token set
        # TODO: should log a warning here
        tokenizer.pad_token = tokenizer.eos_token

    if config.model.model_max_length:
        tokenizer.model_max_length = config.model.model_max_length

    if config.model.chat_template:
        tokenizer.chat_template = chat_template_registry(config.model.chat_template)

    return tokenizer


def build_peft_model(
    base_model, use_gradient_checkpointing: bool, peft_config: TrainingConfig
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


def chat_template_registry(template_name):
    """Selecting a chat template based on code name.

    NOTE: (internal) This registry is experimental and will be formatted
    better once we have explored chat-template uses/cases (e.g., enumerate,
    use .ninja files like them https://github.com/chujiezheng/chat_templates/tree/main
    , etc.)

    Args:
        template_name (str): the code name describing the chat-tamplate.

    Raises:
        NotImplementedError: if the requested code name does not exist
        in the registry.

    Returns:
        str: a ninja-based chat-template.
    """
    if template_name == "Zephyr-7B-LeMa-default":
        return "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"  # noqa
    else:
        raise NotImplementedError(
            "Currently only *experimental* template for Zephyr-7B has been added."
        )
