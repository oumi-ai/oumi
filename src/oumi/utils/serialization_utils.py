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
            except Exception:
                logger.warning(f"Non-serializable value `{obj}` of type `{type(obj)}`.")
                return str(obj)


def convert_all_keys_to_serializable_types(dictionary: dict) -> None:
    """Converts all keys in a hierarchical dictionary to serializable types."""
    serializable_key_types = {str, int, float, bool, None}
    non_serializable_keys = [
        key for key in dictionary if type(key) not in serializable_key_types
    ]
    for key in non_serializable_keys:
        dictionary[str(key)] = dictionary.pop(key)

    # Recursively convert all keys for nested dictionaries.
    for value in dictionary.values():
        if isinstance(value, dict):
            convert_all_keys_to_serializable_types(value)


def json_serializer(obj: Any) -> str:
    """Serializes a Python obj to a JSON formatted string."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        dict_to_serialize = dataclasses.asdict(obj)
    elif isinstance(obj, dict):
        dict_to_serialize = obj
    else:
        raise ValueError(f"Cannot serialize object of type {type(obj)} to JSON.")

    # Ensure all (nested) dictionary keys are serializable.
    if isinstance(dict_to_serialize, dict):
        convert_all_keys_to_serializable_types(dict_to_serialize)

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
