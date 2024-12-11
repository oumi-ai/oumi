from typing import Any

import torch

from oumi.utils.logging import logger


def _make_serializable(item):
    """Converts all non-serializable values of an item to strings."""
    if type(item) in [str, int, float, bool]:
        # These types are json serializable.
        return item
    elif isinstance(item, dict):
        for key, value in item.items():
            item[key] = _make_serializable(value)
        return item
    elif isinstance(item, list):
        return [_make_serializable(value) for value in item]
    else:
        if item is None:
            return "None"
        elif isinstance(item, torch.dtype):
            return str(item)
        else:
            logger.warning(f"Unexpected type for item: `{item}`")
            return str(item)


def make_dict_serializable(dictionary: dict[str, Any]) -> dict[str, Any]:
    """Converts all non-serializable values of a dictionary to strings."""
    for key, value in dictionary.items():
        dictionary[key] = _make_serializable(value)
    return dictionary
