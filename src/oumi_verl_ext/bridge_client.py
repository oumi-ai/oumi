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

"""Constants shared by the oumi bridge server and the verl-side shims."""

from typing import Final

BRIDGE_URL_ENV: Final = "OUMI_VERL_BRIDGE_URL"
HEALTH_PATH: Final = "/health"
REWARD_PATH: Final = "/reward"
TOOL_PATH_PREFIX: Final = "/tool/"
BRIDGE_REWARD_PATH: Final = "pkg://oumi_verl_ext.bridge_reward"
BRIDGE_REWARD_NAME: Final = "oumi_registry_reward"
