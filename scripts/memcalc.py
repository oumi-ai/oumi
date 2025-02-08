import argparse
from typing import Union

import torch
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

from oumi.builders.models import build_model
from oumi.core.configs import TrainingConfig
from oumi.core.configs.internal.supported_models import (
    find_model_hf_config,
)
from oumi.utils.torch_utils import count_model_parameters

# TODO: Confirm this number
# The number of VRAM bytes used by CUDA.
# See: https://discuss.pytorch.org/t/what-is-the-initial-1-3gb-allocated-vram-when-first-using-cuda/122079/2
_CUDA_BYTES = 1.3e9


def get_bytes_per_unit(config: TrainingConfig) -> int:
    """Gets the number of bytes used per number per the torch dtype."""
    dtype = config.model.torch_dtype
    if dtype == torch.float64:
        return 8
    elif dtype == torch.float32:
        return 4
    elif dtype == torch.bfloat16 or dtype == torch.float16:
        return 2
    elif dtype == torch.uint8:
        return 1
    else:
        raise ValueError(f"Unsupported torch dtype: {dtype}")


def bytes_to_str(bytes: Union[int, float]) -> str:
    """Converts a number of bytes to a human-readable string."""
    if bytes < 1000:
        return f"{bytes} B"
    elif bytes < 1e6:
        return f"{bytes / 1000:.1f} KB"
    elif bytes < 1e9:
        return f"{bytes / 1e6:.1f} MB"
    else:
        return f"{bytes / 1e9:.1f} GB"


def get_seq_len(config: TrainingConfig, model_config) -> int:
    """Gets the maximum sequence length supported by the model."""
    # GPT2 specific
    seq_len = model_config.n_positions
    if config.model.model_max_length is not None:
        seq_len = config.model.model_max_length
    return seq_len


def get_data_bytes(config: TrainingConfig, model_config, bytes_per_unit: int) -> int:
    """Gets the total number of bytes used by the data batch."""
    batch_size = config.training.per_device_train_batch_size
    model_max_length = get_seq_len(config, model_config)
    return batch_size * model_max_length * bytes_per_unit


# TODO: Find a static way to calculate this
def get_model_bytes(model: torch.nn.Module, bytes_per_unit: int) -> int:
    """Gets the total number of bytes used by the loaded model."""
    num_total_params = count_model_parameters(model)
    return num_total_params.all_params * bytes_per_unit


def get_optim_bytes(config: TrainingConfig, model_bytes: int) -> int:
    """Gets the total number of bytes used by the optimizer."""
    # TODO: Support more optimizer
    optim = config.training.optimizer
    if optim in ["adamw_torch", "adamw_torch_fused"]:
        multiplier = 2
    elif optim == "adafactor":
        multiplier = 0.3
    elif optim == "sgd":
        # TODO: Account for momentum
        multiplier = 0
    else:
        raise ValueError(f"Unsupported optimizer: {optim}")
    print(f"Optimizer {optim} uses {multiplier}x model bytes")
    return int(model_bytes * multiplier)


def get_gradient_bytes(config: TrainingConfig, model_bytes: int) -> int:
    """Gets the total number of bytes used by gradients."""
    if config.training.gradient_accumulation_steps > 1:
        return model_bytes * 2
    return model_bytes


def get_activation_bytes(config: TrainingConfig, model_config, data_bytes: int) -> int:
    """Gets the total number of bytes used by activations."""
    vocab_size = model_config.vocab_size
    num_layers = model_config.n_layer
    hidden_dim = model_config.n_embd
    seq_len = get_seq_len(config, model_config)
    num_kv_heads = model_config.n_head
    # TODO: Verify this formula
    embedding_bytes = vocab_size
    lm_head_bytes = vocab_size
    transformer_bytes = num_layers * (14 * hidden_dim + seq_len * num_kv_heads)
    total = embedding_bytes + lm_head_bytes + transformer_bytes
    return total * data_bytes


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        help="Path to the configuration file",
    )

    args = parser.parse_args()
    config_path = args.config
    if not config_path.endswith("train.yaml"):
        raise ValueError("Only training configurations are currently supported!")

    config = TrainingConfig.from_yaml(config_path)
    config.finalize_and_validate()
    model_config = find_model_hf_config(
        config.model.model_name, trust_remote_code=config.model.trust_remote_code
    )
    print(model_config)
    model = build_model(
        model_params=config.model,
        peft_params=config.peft if config.training.use_peft else None,
    )
    if not isinstance(model_config, GPT2Config):
        raise ValueError("Only GPT2 models are currently supported!")

    bytes_per_unit = get_bytes_per_unit(config)
    print()
    print("-" * 80)
    print(f"Bytes per unit (a number in memory, ex. param, gradient): {bytes_per_unit}")
    print(f"CUDA bytes: {bytes_to_str(_CUDA_BYTES)}")
    data_bytes = get_data_bytes(config, model_config, bytes_per_unit)
    print(f"Data bytes: {bytes_to_str(data_bytes)}")
    model_bytes = get_model_bytes(model, bytes_per_unit)
    print(f"Model bytes: {bytes_to_str(model_bytes)}")
    optim_bytes = get_optim_bytes(config, model_bytes)
    print(f"Optimizer bytes: {bytes_to_str(optim_bytes)}")
    gradient_bytes = get_gradient_bytes(config, model_bytes)
    print(f"Gradient bytes: {bytes_to_str(gradient_bytes)}")
    activation_bytes = get_activation_bytes(config, model_config, data_bytes)
    print(f"Activation bytes: {bytes_to_str(activation_bytes)}")

    total_bytes = (
        _CUDA_BYTES
        + data_bytes
        + model_bytes
        + optim_bytes
        + gradient_bytes
        + activation_bytes
    )
    print(f"Total bytes: {bytes_to_str(total_bytes)}")

    # TODO: Print fields used


if __name__ == "__main__":
    main()
