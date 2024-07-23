import hashlib
import re

from lema.utils.logging import logger


def sanitize_run_name(run_name: str) -> str:
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
        logger.warning(f"Run name '{run_name}' got sanitized to '{result}'")
    return result
