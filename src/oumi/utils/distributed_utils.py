import os

from oumi.utils.str_utils import str_to_bool


def is_using_accelerate() -> bool:
    """Checks if the training is using Accelerate.

    We do this by checking if the `ACCELERATE_DYNAMO_MODE` environment variable is set.
    This variable should always be set by Accelerate.

    Returns:
        bool: True if Accelerate is being used, False otherwise.
    """
    env_var = os.environ.get("ACCELERATE_DYNAMO_MODE", "false")
    return str_to_bool(env_var)


def is_using_accelerate_fsdp() -> bool:
    """Checks if the training is using Accelerate's FSDP implementation.

    Returns:
        bool: True if Accelerate's FSDP is being used, False otherwise.
    """
    env_var = os.environ.get("ACCELERATE_USE_FSDP", "false")
    return str_to_bool(env_var)
