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

from typing import Any

from oumi.core.configs.synthesis_config import SynthesisConfig
from oumi.core.synthesis.synthesis_pipeline import SynthesisPipeline


def synthesize(config: SynthesisConfig) -> list[dict[str, Any]]:
    """Synthesize a dataset using the provided configuration.

    Args:
        config: The synthesis configuration.

    Returns:
        A list of dictionaries representing the synthesized dataset.
    """
    pipeline = SynthesisPipeline(config)
    return pipeline.synthesize()
