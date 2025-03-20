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

import copy
import hashlib
import logging
import os
import re
from typing import Optional

from oumi.core.tokenizers.base_tokenizer import BaseTokenizer


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


def get_editable_install_override_env_var() -> bool:
    """Returns whether OUMI_FORCE_EDITABLE_INSTALL env var is set to a truthy value."""
    s = os.environ.get("OUMI_FORCE_EDITABLE_INSTALL", "")
    mode = s.lower().strip()
    bool_result = try_str_to_bool(mode)
    if bool_result is not None:
        return bool_result
    return False


# Experimental function, only for developer usage.
def set_oumi_install_editable(setup: str) -> str:
    """Tries to replace oumi PyPi installs with editable installation from source.

    For example, the following line:
        `pip install uv && uv pip -q install oumi[gpu,dev] vllm`
    will be replaced with:
        `pip install uv && uv pip -q install -e '.[gpu,dev]' vllm`

    Args:
        setup (str): The bash setup script to modify. May be multi-line.

    Returns:
        The modified setup script.
    """
    setup_lines = setup.split("\n")
    for i, line in enumerate(setup_lines):
        # Skip comments.
        if line.strip().startswith("#"):
            continue

        # In summary, this regex looks for variants of `pip install oumi` and replaces
        # the oumi package with an editable install from the current directory.
        #
        # Tip: Use https://regexr.com/ or an LLM to help understand the regex.
        # It captures any misc. tokens like flags for the pip and
        # install commands, in addition to any optional dependencies oumi is installed
        # with.
        #
        # `((?:[-'\"\w]+ +)*)` matches whitespace-separated tokens potentially
        # containing quotes, such as flag names and values.
        # `((?:[-'\",\[\]\w]+ +)*)` does the same, with the addition of commas and
        # brackets, which may be present for packages with optional dependencies.
        # Since these don't include special characters like && and ;, it shouldn't span
        # across multiple pip install commands.
        # `(?<!-e )` means we don't match if the previous token is -e. This means an
        # editable install of a local dir called "oumi" is being done, so we skip it.
        # NOTE: We ideally should check for `--editable` as well, but Python re doesn't
        # support lookbehinds with variable length.
        # We additionally consume quotation marks around oumi if present.
        # Finally, `(\[[^\]]*\])?['\"]?` captures optional dependencies, if present.
        pattern = (
            r"pip3? +((?:[-'\"\w]+ +)*)install +((?:[-'\",\[\]\w]+ +)*)"
            r"(?<!-e )['\"]?oumi(\[[^\]]*\])?['\"]?"
        )
        # Compared to the pattern we captured, the changes are replacing `oumi` with
        # `.` and adding `-e` to make the install editable.
        replacement = r"pip \1install \2-e '.\3'"

        result = re.sub(pattern, replacement, line)
        if result == line:
            continue
        # Replace the line in the setup script.
        logger = logging.getLogger("oumi")
        logger.info(f"Detected the following oumi installation: `{line}`")
        logger.info(f"Replaced with: `{result}`")
        setup_lines[i] = result
    return "\n".join(setup_lines)


def truncate_to_max_tokens_limit(
    text: str,
    tokenizer: BaseTokenizer,
    *,
    max_tokens: int,
    truncation_side: str = "right",
) -> tuple[str, int]:
    """Truncates text to `max_length` in tokens.

    Args:
        text: A text prompt.
        tokenizer: The tokenizer used for encoding the data.
        max_tokens: Maximum number of tokens to keep.
        truncation_side: The side to truncate the tokens ("right" or "left").

    Returns:
        A tuple containing truncated text prompt and the number of tokens.
    """
    if max_tokens <= 0:
        raise ValueError("`max_tokens` must be a positive integer")
    elif truncation_side not in ("left", "right"):
        raise ValueError(
            f"Invalid truncation_side: '{truncation_side}'. Expected 'left' or 'right'."
        )

    if not text:
        return ("", 0)

    left_side = truncation_side == "left"

    # The `truncation_side` parameter isn't universally supported by all tokenizers.
    # Let's do left-side truncation as post-processing.
    result = tokenizer(
        text,
        return_offsets_mapping=True,
        return_length=True,
        max_length=(None if left_side else max_tokens),
        truncation=(not left_side),  # Left-side truncation is done as post-processing.
    )
    if "offset_mapping" not in result:
        raise RuntimeError(
            f"Tokenizer must return offset mapping for truncation! Got: {result.keys()}"
        )
    token2char_offsets = result["offset_mapping"]
    if not isinstance(token2char_offsets, list):
        raise RuntimeError(
            "offset_mapping returned by tokenizer is not a list! "
            f"Got: {type(token2char_offsets)}"
        )

    truncated_text: str = ""
    num_truncated_tokens: int = 0
    if len(token2char_offsets) > 0:
        num_truncated_tokens = min(len(token2char_offsets), max_tokens)
        if left_side:
            lead_token_start = token2char_offsets[-num_truncated_tokens][0]
            if not (lead_token_start >= 0 and lead_token_start < len(text)):
                raise RuntimeError(
                    f"Truncation error: lead_token_start={lead_token_start} "
                    f"for text of length {len(text)}"
                )
            truncated_text = text[lead_token_start:]
        else:
            last_token_end = token2char_offsets[num_truncated_tokens - 1][1]
            if not (last_token_end >= 0 and last_token_end <= len(text)):
                raise RuntimeError(
                    f"Truncation error: last_token_end={last_token_end} "
                    f"for text of length {len(text)}"
                )
            truncated_text = text[:last_token_end]

    return (truncated_text, num_truncated_tokens)


def truncate_text_pieces_to_max_tokens_limit(
    text_pieces: list[str],
    tokenizer: BaseTokenizer,
    *,
    max_tokens: int,
    truncation_side: str = "right",
) -> list[str]:
    """Truncates text pieces to total length not exceeding `max_length`.

    Args:
        text_pieces: A list of text prompts.
        tokenizer: The tokenizer used for encoding the data.
        max_tokens: Maximum number of tokens to keep in all text pieces combined.
        truncation_side: The side to truncate the tokens ("right" or "left").

    Returns:
        A list of truncated text prompts.
    """
    if max_tokens <= 0:
        raise ValueError("`max_tokens` must be a positive integer")
    remaining_tokens = max_tokens

    result = copy.deepcopy(text_pieces)
    if truncation_side == "left":
        result.reverse()

    for idx, text_piece in enumerate(result):
        if len(text_piece) == 0:
            continue
        elif remaining_tokens > 0:
            truncated_text_piece, num_tokens = truncate_to_max_tokens_limit(
                text_piece,
                tokenizer=tokenizer,
                max_tokens=remaining_tokens,
                truncation_side=truncation_side,
            )
            result[idx] = truncated_text_piece
            remaining_tokens -= num_tokens
        else:
            result[idx] = ""

    if truncation_side == "left":
        result.reverse()

    return result
