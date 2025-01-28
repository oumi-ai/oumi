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

import hashlib
import logging
import re
from typing import Optional


def sanitize_run_name(run_name: Optional[str]) -> Optional[str]:
    """Computes a sanitized version of wandb run name.

    A valid run name may only contain alphanumeric characters, dashes, underscores,
    and dots, with length not exceeding max limit.

    Args:
        run_name: The original raw value of run name.
    """
    if not run_name:
        return run_name

    # Technically, the limit is 128 chars, but we limit to 100 characters
    # because the system may generate aux artifact names e.g., by prepending a prefix
    # (e.g., "model-") to our original run name, which are also subject
    # to max 128 chars limit.
    _MAX_RUN_NAME_LENGTH = 100

    # Replace all unsupported characters with '_'.
    result = re.sub("[^a-zA-Z0-9\\_\\-\\.]", "_", run_name)
    if len(result) > _MAX_RUN_NAME_LENGTH:
        suffix = "..." + hashlib.shake_128(run_name.encode("utf-8")).hexdigest(8)
        result = result[0 : (_MAX_RUN_NAME_LENGTH - len(suffix))] + suffix

    if result != run_name:
        logger = logging.getLogger("oumi")
        logger.warning(f"Run name '{run_name}' got sanitized to '{result}'")
    return result


def str_to_bool(s: str) -> bool:
    """Convert a string representation to a boolean value.

    This function interprets various string inputs as boolean values.
    It is case-insensitive and recognizes common boolean representations.

    Args:
        s: The string to convert to a boolean.

    Returns:
        bool: The boolean interpretation of the input string.

    Raises:
        ValueError: If the input string cannot be interpreted as a boolean.

    Examples:
        >>> str_to_bool("true") # doctest: +SKIP
        True
        >>> str_to_bool("FALSE") # doctest: +SKIP
        False
        >>> str_to_bool("1") # doctest: +SKIP
        True
        >>> str_to_bool("no") # doctest: +SKIP
        False
    """
    s = s.strip().lower()

    if s in ("true", "yes", "1", "on", "t", "y"):
        return True
    elif s in ("false", "no", "0", "off", "f", "n"):
        return False
    else:
        raise ValueError(f"Cannot convert '{s}' to boolean.")


def compute_utf8_len(s: str) -> int:
    """Computes string length in UTF-8 bytes."""
    # This is inefficient: allocates a temporary copy of string content.
    # FIXME Can we do better?
    return len(s.encode("utf-8"))
