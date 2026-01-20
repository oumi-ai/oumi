# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions which use detect-by-name heuristics.

# TODO(OPE-303): These should be replaced with something more robust.
"""

import importlib
from typing import Any

import torch
import torch.nn as nn
import transformers

from oumi.utils.logging import logger
from oumi.utils.torch_utils import _get_parameter_names

_PARAMS_KEY = "params"
_WEIGHT_DECAY_KEY = "weight_decay"


def disable_dropout(hf_config: transformers.PretrainedConfig) -> None:
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


def group_trainable_params(
    model: torch.nn.Module, weight_decay: float
) -> list[dict[str, Any]]:
    """Groups trainable params by weight decay for optimization.

    As a rule of thumb, we generally want to weight decay all 2d matrices, i.e.
    weight tensors for matmuls/embeddings, and not biases/layernorms.

    Args:
        model: The model whose parameters will be optimized.
        weight_decay: The weight decay to apply to the appropriate parameters.

    Returns:
        List[Dict[str, Any]]: A list containing two dictionaries: the first with
            parameters that should be weight decayed and the second with parameters
            that shouldn't.
    """
    # Exclude layernorm and bias tensors.
    decay_parameters = _get_parameter_names(
        model, forbidden_layer_types=[torch.nn.LayerNorm]
    )
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    # Only include trainable params.
    trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    # Group by weight decay.
    return [
        {
            _PARAMS_KEY: [p for n, p in trainable_params if n in decay_parameters],
            _WEIGHT_DECAY_KEY: weight_decay,
        },
        {
            _PARAMS_KEY: [p for n, p in trainable_params if n not in decay_parameters],
            _WEIGHT_DECAY_KEY: 0.0,
        },
    ]


def guess_transformer_layer_cls(model: nn.Module) -> type[nn.Module]:
    """Guess the transformer layer class based on the model architecture."""
    for module in model.modules():
        for layer_pattern in ["layer", "block", "transformerlayer"]:
            layer_name = str(type(module)).lower()

            if layer_pattern in layer_name and "layernorm" not in layer_name:
                return type(module)

    raise ValueError(
        "Unable to guess transformer layer class. Please specify it explicitly."
    )


def _parse_transformer_layer_cls_string(class_names: str) -> list[str]:
    result: list[str] = []
    for class_name in class_names.split(","):
        class_name = class_name.strip()
        if class_name:
            result.append(class_name)
    return result


def _get_module_class_from_name(module: nn.Module, name: str) -> type[nn.Module] | None:
    """Search for a class by name in the model's module tree.

    This matches the behavior of accelerate's get_module_class_from_name function,
    which searches through the model's children recursively to find a module
    whose class name matches the given name.

    Args:
        module: The model to search through.
        name: The class name to search for.

    Returns:
        The class if found, None otherwise.
    """
    if module.__class__.__name__ == name:
        return module.__class__
    for child in module.children():
        result = _get_module_class_from_name(child, name)
        if result is not None:
            return result
    return None


def resolve_transformer_layer_cls_string_as_module_set(
    class_names: str,
    model: nn.Module | None = None,
) -> set[type[nn.Module]]:
    """Get module classes from their string names.

    For simple class names (no module prefix like "LlamaDecoderLayer"):
    - If model is provided, searches the model's module tree first (like accelerate)
    - Falls back to importing from the 'transformers' module

    For fully-qualified names (like "transformers.models.llama.LlamaDecoderLayer"):
    - Uses the standard import approach

    Args:
        class_names: Comma-separated list of class names.
        model: Optional model to search for classes. When provided, simple class
            names are resolved by searching the model tree (matching accelerate's
            behavior). This is the recommended approach for FSDP wrapping.

    Returns:
        Set of module classes.
    """
    result: set[type[nn.Module]] = set()
    for class_name in _parse_transformer_layer_cls_string(class_names):
        parts = class_name.rsplit(".", maxsplit=1)

        if len(parts) == 1:
            # Simple class name (no module prefix)
            # First try to find it in the model tree (like accelerate does)
            if model is not None:
                transformer_cls = _get_module_class_from_name(model, class_name)
                if transformer_cls is not None:
                    result.add(transformer_cls)
                    continue

            # Fall back to importing from transformers module
            # (backward compatibility for cases where model is not provided)
            try:
                module = importlib.import_module("transformers")
                transformer_cls = getattr(module, class_name)
                result.add(transformer_cls)
            except AttributeError:
                raise ValueError(
                    f"Could not find transformer layer class '{class_name}'. "
                    f"Either pass the model to search its module tree, or use a "
                    f"fully-qualified name like 'transformers.models.X.modeling_X.{class_name}'."
                )
        else:
            # Fully-qualified name (e.g., transformers.models.llama.LlamaDecoderLayer)
            module_name, cls_name = parts
            module = importlib.import_module(module_name)
            transformer_cls = getattr(module, cls_name)
            result.add(transformer_cls)

    return result


def simplify_transformer_layer_cls_string(class_names: str) -> str:
    """Replaces fully-qualified class names with pure class names.

    For example, converts 'foo.Block,foo.util.Decoder' to 'Block,Decoder'.

    The `accelerate` library expects the simplified format for the
    FSDP_TRANSFORMER_CLS_TO_WRAP environment variable.
    """
    result = []
    for class_name in _parse_transformer_layer_cls_string(class_names):
        parts = class_name.rsplit(".")
        result.append(parts[-1])
    return ",".join(result)
