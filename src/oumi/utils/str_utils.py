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
import os
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


def try_str_to_bool(s: str) -> Optional[bool]:
    """Attempts to convert a string representation to a boolean value.

    This function interprets various string inputs as boolean values.
    It is case-insensitive and recognizes common boolean representations.

    Args:
        s: The string to convert to a boolean.

    Returns:
        bool: The boolean interpretation of the input string, or `None`
            for unrecognized string values.

    Examples:
        >>> str_to_bool("true") # doctest: +SKIP
        True
        >>> str_to_bool("FALSE") # doctest: +SKIP
        False
        >>> str_to_bool("1") # doctest: +SKIP
        True
        >>> str_to_bool("no") # doctest: +SKIP
        False
        >>> str_to_bool("peach") # doctest: +SKIP
        None
    """
    s = s.strip().lower()

    if s in ("true", "yes", "1", "on", "t", "y"):
        return True
    elif s in ("false", "no", "0", "off", "f", "n"):
        return False
    return None


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
    result = try_str_to_bool(s)

    if result is None:
        raise ValueError(f"Cannot convert '{s}' to boolean.")
    return result


def compute_utf8_len(s: str) -> int:
    """Computes string length in UTF-8 bytes."""
    # This is inefficient: allocates a temporary copy of string content.
    # FIXME Can we do better?
    return len(s.encode("utf-8"))


def get_editable_install_override() -> bool:
    """Returns whether OUMI_TRY_EDITABLE_INSTALL env var is set to a truthy value."""
    s = os.environ.get("OUMI_TRY_EDITABLE_INSTALL", "")
    mode = s.lower().strip()
    bool_result = try_str_to_bool(mode)
    if bool_result is not None:
        return bool_result
    return False


def set_oumi_install_editable(setup: str) -> str:
    """Try to replace oumi PyPi install with installation from source.

    Args:
        setup (str): The setup script to modify.

    Returns:
        The modified setup script.
    """
    setup_lines = setup.split("\n")
    for i, line in enumerate(setup_lines):
        if line.strip().startswith("#"):
            continue
        pip_idx = line.find("pip")
        install_idx = line.find("install", pip_idx)
        oumi_idx = line.find("oumi", install_idx)
        if not (pip_idx != -1 and install_idx != -1 and oumi_idx != -1):
            continue
        oumi_end_idx = oumi_idx + 4
        while oumi_end_idx < len(line) and not line[oumi_end_idx].isspace():
            oumi_end_idx += 1
        oumi_install = line[oumi_idx:oumi_end_idx]

        setup_lines[i] = line[: oumi_idx - 1] + line[oumi_end_idx:]
        len_whitespace_prefix = len(line) - len(line.lstrip())
        prefix = " " * len_whitespace_prefix
        setup_lines.insert(
            i + 1,
            f"{prefix}pip install uv && uv pip install -e '.{oumi_install[4:]}'",
        )
        break
    return "\n".join(setup_lines)
