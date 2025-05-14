"""Calculates the memory usage of a model during training.

Usage:
    python scripts/memcalc.py -c path/to/train/config.yaml
"""

import argparse
from dataclasses import dataclass
from typing import Union

import torch
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

from oumi.builders.models import build_model
from oumi.core.configs import TrainingConfig
from oumi.core.configs.internal.supported_models import (
    find_model_hf_config,
)
from oumi.utils.torch_utils import count_model_parameters, get_dtype_size_in_bytes

# TODO: Confirm this number
# The number of VRAM bytes used by CUDA.
# See: https://discuss.pytorch.org/t/what-is-the-initial-1-3gb-allocated-vram-when-first-using-cuda/122079/2
_CUDA_BYTES = 1.3e9  # 1.3 GB
# TODO: Confirm this number
# Torch dtype for a token in the input batch.
_TOKEN_DTYPE = torch.int32


@dataclass
class ModelConfig:
    """Standardized model configuration, created from a HF model config."""

    vocab_size: int
    """The vocabulary size."""

    seq_len: int
    """The maximum sequence length."""

    num_layers: int
    """The number of attention layers."""

    hidden_dim: int
    """The hidden dimension aka embedding size."""

    num_kv_heads: int
    """The number of key value (KV) heads for the attention layer."""


def num_bytes_to_str(bytes: Union[int, float]) -> str:
    """Returns a human-readable string for a number of bytes."""
    if bytes < 1000:
        return f"{bytes} B"
    elif bytes < 1e6:
        return f"{bytes / 1000:.1f} KB"
    elif bytes < 1e9:
        return f"{bytes / 1e6:.1f} MB"
    else:
        return f"{bytes / 1e9:.1f} GB"


def get_seq_len(config: TrainingConfig, model_config: ModelConfig) -> int:
    """Gets the maximum sequence length supported by the model."""
    seq_len = model_config.seq_len
    if config.model.model_max_length is not None:
        seq_len = config.model.model_max_length
    return seq_len


# TODO: Add support for LlamaConfig
def get_standardized_model_config(hf_model_config) -> ModelConfig:
    """Gets a standardized model config given a HF model config.

    Each HF config may use different field names for the same property. This function
    converts them to a standardized format for use throughout the script.
    """
    if isinstance(hf_model_config, GPT2Config):
        return ModelConfig(
            vocab_size=hf_model_config.vocab_size,
            seq_len=hf_model_config.n_positions,
            num_layers=hf_model_config.n_layer,
            hidden_dim=hf_model_config.n_embd,
            num_kv_heads=hf_model_config.n_head,
        )
    else:
        raise ValueError(f"Unsupported HF model type: {type(hf_model_config)}")


# --------------------------------------------------------------------------------------
# Functions for calculating the memory usage of different parts of the training process.
# --------------------------------------------------------------------------------------


def get_data_bytes(
    config: TrainingConfig, model_config: ModelConfig, bytes_per_unit: int
) -> int:
    """Gets the total number of bytes used by the data batch."""
    batch_size = config.training.per_device_train_batch_size
    model_max_length = get_seq_len(config, model_config)
    return batch_size * model_max_length * get_dtype_size_in_bytes(_TOKEN_DTYPE)


# TODO: Find a static way to calculate this
def get_model_bytes(model: torch.nn.Module, bytes_per_unit: int) -> int:
    """Gets the total number of bytes used by the loaded model."""
    num_total_params = count_model_parameters(model).all_params
    print(f"- Model parameter count: {num_total_params:,}")
    return num_total_params * bytes_per_unit


# TODO: Support more optimizers
def get_optim_bytes(config: TrainingConfig, model_bytes: int) -> int:
    """Gets the total number of bytes used by the optimizer."""
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
    print("- The size of the gradient is the same as that for model weights.")
    if config.training.gradient_accumulation_steps > 1:
        print(
            "- If gradient accumulation is used, a buffer to store gradients is "
            "required. The buffer is the same size as the gradients."
        )
        return model_bytes * 2
    return model_bytes


def get_activation_bytes(
    config: TrainingConfig, model_config: ModelConfig, bytes_per_unit: int
) -> int:
    """Gets the total number of bytes used by activations."""
    vocab_size = model_config.vocab_size
    num_layers = model_config.num_layers
    hidden_dim = model_config.hidden_dim
    seq_len = get_seq_len(config, model_config)
    num_kv_heads = model_config.num_kv_heads
    # TODO: Verify this formula
    embedding_bytes = vocab_size
    lm_head_bytes = vocab_size
    transformer_bytes = num_layers * (14 * hidden_dim + seq_len * num_kv_heads)
    total = embedding_bytes + lm_head_bytes + transformer_bytes

    batch_size = config.training.per_device_train_batch_size
    model_max_length = get_seq_len(config, model_config)
    return total * batch_size * model_max_length * bytes_per_unit


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
    hf_model_config = find_model_hf_config(
        config.model.model_name, trust_remote_code=config.model.trust_remote_code
    )
    print("HuggingFace model config:")
    print(hf_model_config)
    model_config = get_standardized_model_config(hf_model_config)
    print("Standardized model config:")
    print(model_config)
    model = build_model(
        model_params=config.model,
        peft_params=config.peft if config.training.use_peft else None,
    )

    bytes_per_unit = get_dtype_size_in_bytes(config.model.torch_dtype)
    print()
    print("-" * 80)
    print(f"Bytes per memory unit: {bytes_per_unit}")
    print("Bytes used by different parts of the training process:")
    print(f"Base memory usage: {num_bytes_to_str(_CUDA_BYTES)}")
    print("- This includes loading CUDA, GPU kernels, cuDNN/cuBLAS, etc.")
    data_bytes = get_data_bytes(config, model_config, bytes_per_unit)
    print(f"Data (input batches of token ids): {num_bytes_to_str(data_bytes)}")
    model_bytes = get_model_bytes(model, bytes_per_unit)
    print(f"Model weights: {num_bytes_to_str(model_bytes)}")
    activation_bytes = get_activation_bytes(config, model_config, bytes_per_unit)
    print(f"Model activations: {num_bytes_to_str(activation_bytes)}")
    gradient_bytes = get_gradient_bytes(config, model_bytes)
    print(f"Model gradients: {num_bytes_to_str(gradient_bytes)}")
    optim_bytes = get_optim_bytes(config, model_bytes)
    print(f"Optimizer state: {num_bytes_to_str(optim_bytes)}")

    total_bytes = (
        _CUDA_BYTES
        + data_bytes
        + model_bytes
        + activation_bytes
        + gradient_bytes
        + optim_bytes
    )
    print(f"Total bytes: {num_bytes_to_str(total_bytes)}")

    # TODO: Print config fields used


if __name__ == "__main__":
    main()
