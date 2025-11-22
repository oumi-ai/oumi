"""File path validation utilities."""

from pathlib import Path


def validate_and_sanitize_file_path(file_path: str) -> tuple[bool, str, str]:
    """Validate and sanitize a file path for security and safety.

    Args:
        file_path: The file path to validate.

    Returns:
        Tuple of (is_valid, sanitized_path, error_message).
    """
    try:
        from pathvalidate import (
            ValidationError,
            is_valid_filepath,
            sanitize_filepath,
        )
    except ImportError:
        return (
            False,
            "",
            "pathvalidate library is required for file path validation",
        )

    if not file_path:
        return False, "", "File path cannot be empty"

    # Check for unmatched quotes
    stripped = file_path.strip()
    quote_chars = ["'", '"']
    for quote in quote_chars:
        if stripped.startswith(quote) and not stripped.endswith(quote):
            return False, "", f"Unmatched quote in file path: {quote}"
        if stripped.endswith(quote) and not stripped.startswith(quote):
            return False, "", f"Unmatched quote in file path: {quote}"

    # Strip whitespace and quotes
    cleaned_path = file_path.strip().strip("\"'")

    # Check if empty after cleaning
    if not cleaned_path or cleaned_path.isspace():
        return False, "", "File path is empty or contains only whitespace"

    # Detect platform
    import os

    platform_type = "windows" if os.name == "nt" else "posix"

    # Sanitize using pathvalidate
    try:
        sanitized = sanitize_filepath(
            cleaned_path,
            platform=platform_type,
            max_len=255,  # Standard filesystem limit
        )
    except ValidationError as e:
        return False, "", f"Invalid file path: {str(e)}"

    # Verify the sanitized path is valid
    if not is_valid_filepath(sanitized, platform=platform_type):
        return False, "", "File path contains invalid characters or format"

    # Prevent path traversal
    if ".." in sanitized:
        return (
            False,
            "",
            "File path contains potential security risks (path traversal)",
        )

    # Check for quotes in file name
    if any(quote in Path(sanitized).name for quote in ["'", '"']):
        return False, "", "File name cannot contain quote characters"

    return True, sanitized, ""
