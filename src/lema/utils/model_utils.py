import importlib
from typing import Type

import torch.nn as nn


def guess_transformer_layer_cls(model: nn.Module) -> Type[nn.Module]:
    """Guess the transformer layer class based on the model architecture."""
    if hasattr(model, "transformer"):
        return type(model.transformer.h[0])
    elif hasattr(model, "encoder"):
        return type(model.encoder.layer[0])
    else:
        raise ValueError(
            "Unable to guess transformer layer class. Please specify it explicitly."
        )


def get_module_class_from_name(class_name: str) -> Type[nn.Module]:
    """Get a module class from its string name."""
    module_name, class_name = class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
