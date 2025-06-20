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

import random
from typing import Optional

from oumi.core.configs.params.synthesis_params import (
    AttributeCombination,
    GeneralSynthesisParams,
    PermutableAttribute,
)


class DatasetPlanner:
    def plan(
        self,
        synthesis_params: GeneralSynthesisParams,
        sample_count: int,
    ) -> list[dict]:
        """Plan the dataset.

        Args:
            synthesis_params: The synthesis parameters.
            sample_count: The number of samples to plan.

        Returns:
            A list of dictionaries, each representing a sample of the dataset with
            the attribute values for each attribute.
        """
        permutable_attributes = self._plan_permutable_attributes(
            synthesis_params.permutable_attributes,
            synthesis_params.combination_sampling,
            sample_count,
        )

        dataset_plan = permutable_attributes

        return dataset_plan

    def _plan_permutable_attributes(
        self,
        permutable_attributes: Optional[list[PermutableAttribute]],
        combination_sampling: Optional[list[AttributeCombination]],
        sample_count: int,
    ) -> list[dict]:
        if sample_count < 0:
            raise ValueError("Count must be positive")
        elif (
            sample_count == 0
            or permutable_attributes is None
            or len(permutable_attributes) == 0
        ):
            return []

        sampling_overrides = combination_sampling or []

        cumulative_override_probability = sum(
            [combination.sample_rate for combination in sampling_overrides]
        )

        # If cumulative probability is greater than 1, raise an error
        if cumulative_override_probability > 1.0:
            raise ValueError(
                "Cumulative probability of combination sampling must "
                "be less than or equal to 1.0."
            )

        # Generate sample count permutations of the permutable attributes
        samples = [{} for _ in range(sample_count)]
        attribute_distributions = {
            perm_attr.id: perm_attr.get_value_distribution()
            for perm_attr in permutable_attributes
        }

        if cumulative_override_probability == 0.0:
            normalized_override_sample_rates = [0.0] * len(sampling_overrides)
        else:
            normalized_override_sample_rates = [
                combination.sample_rate / cumulative_override_probability
                for combination in sampling_overrides
            ]

        possible_sampling_overrides = [
            override.combination for override in sampling_overrides
        ]

        for i, sample in enumerate(samples):
            sample_combination = {}

            random_number = random.random()

            # If random number < cumulative probability, sample from an override
            sampled_from_override = False
            if random_number < cumulative_override_probability:
                combination_to_sample = random.choices(
                    sampling_overrides,
                    normalized_override_sample_rates,
                    k=1,
                )[0]
                sample_combination = combination_to_sample.combination
                sampled_from_override = True

            original_sample_combination = {k: v for k, v in sample_combination.items()}
            # Sample remaining attributes
            while len(sample_combination.keys()) == 0 or (
                not sampled_from_override
                and _check_if_matches_override(
                    sample_combination,
                    possible_sampling_overrides,
                )
            ):
                sample_combination = {
                    k: v for k, v in original_sample_combination.items()
                }
                for perm_attr in permutable_attributes:
                    # If the attribute is already in the sample combination, skip it.
                    if perm_attr.id in sample_combination:
                        continue

                    value_distribution = attribute_distributions[perm_attr.id]
                    sample_combination[perm_attr.id] = random.choices(
                        list(value_distribution.keys()),
                        list(value_distribution.values()),
                        k=1,
                    )[0]

            for key, value in sample_combination.items():
                samples[i][key] = value

        return samples


def _check_if_matches_override(
    sample_combination: dict,
    override_combinations: list[dict],
) -> bool:
    # For each override combination, check if it's contained in the sample
    for override_combination in override_combinations:
        # Check if all attributes in the override combination are in the sample
        all_attributes_present = all(
            attr in sample_combination for attr in override_combination
        )
        if not all_attributes_present:
            # Attributes not present, move on to next override
            continue

        # Check if the attribute values are the same
        all_values_match = all(
            sample_combination[attr] == override_combination[attr]
            for attr in override_combination
        )
        if not all_values_match:
            # Values don't match, move on to next override
            continue

        # We've got a match, resample
        return True

    return False
