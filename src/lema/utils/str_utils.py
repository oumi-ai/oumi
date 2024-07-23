import hashlib
import re


def sanitize_run_name(run_name: str) -> str:
    """Computes a sanitized version of run name.

    A valid run name may only contain alphanumeric characters, dashes, underscores,
    and dots, with length not exceeding 128 characters.

    Args:
        run_name: The original raw value of run name.
    """
    if not run_name:
        return run_name

    _MAX_RUN_NAME_LENGTH = 128

    result = re.sub("[^a-zA-Z0-9\\_\\-\\.]", "_", run_name)
    if len(result) > _MAX_RUN_NAME_LENGTH:
        suffix = "..." + hashlib.shake_128(run_name.encode("utf-8")).hexdigest(8)
        result = result[0 : (_MAX_RUN_NAME_LENGTH - len(suffix))] + suffix
    return result
