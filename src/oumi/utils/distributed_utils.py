import os

from oumi.utils.str_utils import str_to_bool


def is_using_accelerate() -> bool:
    """Returns whether the current job was launched with the Accelerate launcher.

    We do this by checking if the `ACCELERATE_DYNAMO_*` environment variables are set.
    These variables should always be set by Accelerate. We check for all of them in case
    Accelerate changes the environment variables in the future.
    """
    return (
        "ACCELERATE_DYNAMO_BACKEND" in os.environ
        or "ACCELERATE_DYNAMO_MODE" in os.environ
        or "ACCELERATE_DYNAMO_USE_FULLGRAPH" in os.environ
        or "ACCELERATE_DYNAMO_USE_DYNAMIC" in os.environ
    )


def is_using_accelerate_fsdp() -> bool:
    """Returns whether the current job is requesting Accelerate FSDP training."""
    return str_to_bool(os.environ.get("ACCELERATE_USE_FSDP", "false"))
