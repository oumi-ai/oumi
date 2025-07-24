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

from oumi.core.configs.params.synthesis_params import GeneratedAttribute
from oumi.core.synthesis.attribute_synthesizer import AttributeSynthesizer
from oumi.utils.logging import logger


class DataSynthesizer:
    """Synthesizes data using attributes from the dataset plan."""

    def __init__(
        self,
        generated_attributes: list[GeneratedAttribute],
        attribute_synthesizer: AttributeSynthesizer,
    ):
        """Initialize the synthesizer."""
        self._generated_attributes = generated_attributes
        self._attribute_synthesizer = attribute_synthesizer

    def synthesize(
        self,
        dataset_plan_samples: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Synthesize data using attributes from the dataset plan.

        If there are multiple generated attributes, each will be synthesized
        sequentially, with their resulting output added to the dataset plan records.
        """
        if not dataset_plan_samples:
            return dataset_plan_samples

        for generated_attribute in self._generated_attributes:
            logger.info(f"Synthesizing generated attribute: {generated_attribute.id}")
            synthesized_attribute_records = self._attribute_synthesizer.synthesize(
                dataset_plan_samples,
                generated_attribute,
            )
            for i, record in enumerate(synthesized_attribute_records):
                dataset_plan_samples[i].update(record)

        return dataset_plan_samples
