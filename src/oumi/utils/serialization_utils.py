import dataclasses
import json
from typing import Any

import numpy as np
import torch

from oumi.utils.logging import logger

JSON_FILE_INDENT = 2


class TorchJsonEncoder(json.JSONEncoder):
    # Override default() method
    def default(self, obj):
        """Extending python's JSON Encoder to serialize torch dtype."""
        if obj is None:
            return ""
        # JSON does NOT natively support any torch types.
        elif isinstance(obj, torch.dtype):
            return str(obj)
        # JSON does NOT natively support numpy types.
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            try:
                return super().default(obj)
            except Exception as e:
                logger.error(f"Non-serializable value `{obj}` of type `{type(obj)}`.")
                logger.error(f"Masking exception: {e}.")
                return f"NON_SERIALIZABLE_OBJECT ({type(obj)})"


def json_serializer(obj: Any) -> str:
    """Serializes a Python obj to a JSON formatted string."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        dict_to_serialize = dataclasses.asdict(obj)
    elif isinstance(obj, dict):
        dict_to_serialize = obj
    else:
        raise ValueError(f"Cannot serialize object of type {type(obj)} to JSON.")

    # Attempt to serialize the dictionary to JSON.
    try:
        return json.dumps(
            dict_to_serialize, cls=TorchJsonEncoder, indent=JSON_FILE_INDENT
        )
    except Exception as e:
        error_str = "Non-serializable dict:\n"
        for key, value in dict_to_serialize.items():
            error_str += f" - {key}: {value} (type: {type(value)})\n"
        logger.error(error_str)
        raise Exception(f"Failed to serialize dict to JSON: {e}")
