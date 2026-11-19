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

"""Entry point verl imports through ``VERL_USE_EXTERNAL_MODULES``.

verl imports this inside ``import verl`` in every process, so it imports no verl
module; oumi's GRPO metrics are installed once verl's trainer base has loaded.
"""

import logging

from oumi_verl_ext.grpo_metrics_patch import install_grpo_metrics
from oumi_verl_ext.hooks import register_post_import_hook
from oumi_verl_ext.version import check_verl_version

# verl sets the root logger to WARNING before it imports this module.
logging.getLogger("oumi_verl_ext").setLevel(logging.INFO)
check_verl_version()
register_post_import_hook("verl.trainer.ppo.v1.trainer_base", install_grpo_metrics)
