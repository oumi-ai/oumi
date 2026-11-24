# Copyright 2026 - Oumi
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

"""Module verl imports through ``actor_rollout_ref.model.external_lib``.

verl imports it in every process that builds the model config. In the trainer
process, verl's trainer base and its dataset module are already loaded by then,
so this patches them.
"""

import importlib.metadata
import sys

from oumi_verl_ext.grpo_metrics import patch_compute_metrics
from oumi_verl_ext.tool_schemas import patch_length_filter

# The patches rely on verl internals, so any other version may silently skip them.
_verl_version = importlib.metadata.version("verl")
if _verl_version != "0.9.1":
    raise RuntimeError(f"oumi_verl_ext needs verl 0.9.1, found {_verl_version}.")

_trainer_base = sys.modules.get("verl.trainer.ppo.v1.trainer_base")
if _trainer_base is not None:
    patch_compute_metrics(_trainer_base)
    # The trainer builds its datasets, filtering prompts by length, after this.
    patch_length_filter(sys.modules["verl.utils.dataset.rl_dataset"])
