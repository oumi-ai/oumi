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

"""Install Oumi's VERL FSDP1 worker patches.

VERL imports a single module named by ``actor_rollout_ref.model.external_lib`` in
each worker, so this module imports every patch module.
"""

import oumi.utils.verl_utils.dynamic_bsz_dp_sync  # noqa: F401
import oumi.utils.verl_utils.fsdp1_rank_buffer_sync  # noqa: F401
