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


def is_using_torchrun() -> bool:
    """Returns whether the current job was launched with torchrun.

    torchrun (torch.distributed.elastic) sets TORCHELASTIC_RUN_ID even
    when running with a single GPU, making this a definitive detection method.
    """
    return "TORCHELASTIC_RUN_ID" in os.environ


def is_under_distributed_launcher() -> bool:
    """Returns whether the current job is running under any distributed launcher.

    Detects both torchrun and accelerate launchers. This is more robust than
    checking WORLD_SIZE > 1, as launchers set specific environment variables
    even when running with a single GPU.
    """
    return is_using_torchrun() or is_using_accelerate()
