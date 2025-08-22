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
    DatasetSource,
    DocumentSource,
    ExampleSource,
    GeneralSynthesisParams,
    SampledAttribute,
)
from oumi.core.synthesis.dataset_ingestion import DatasetReader
from oumi.core.synthesis.document_ingestion import DocumentReader, DocumentSegmenter


class DatasetPlanner:
    """Planner for the dataset's attributes for inference."""

    def __init__(
        self,
        document_reader: Optional[DocumentReader] = None,
        dataset_reader: Optional[DatasetReader] = None,
    ):
        """Initialize the dataset planner."""
        self._document_reader = document_reader or DocumentReader()
        self._dataset_reader = dataset_reader or DatasetReader()

    def plan(
        self,
        synthesis_params: GeneralSynthesisParams,
        sample_count: int,
    ) -> list[dict]:
        """Setup the dataset's attributes for inference.

        This function will create a list of dictionaries, with each dictionary
        representing a sample of the dataset with a particular attribute value for
        each attribute.

        - Example sources are used to populate the dataset plan with a set of examples
          for specific attributes, with each example being used round-robin.
        - Document sources are used to populate the dataset plan with documents and/or
          document segments, each sample of a document source being used round-robin.
        - Dataset sources are used to populate the dataset plan with values for the
          attributes, with each sample of a dataset source being used round-robin.
        - Permutable attributes have their values sampled from a distribution.
        - Combination sampling overrides the distribution for particular attribute-value
          combinations.

        The final list of dictionaries will be used to create a dataset.

        Args:
            synthesis_params: The synthesis parameters.
            sample_count: The number of samples to plan.

        Returns:
            A list of dictionaries, each representing a sample of the dataset with
            the attribute values for each attribute.
        """
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")

        example_sample_sets = self._ingest_example_sources(
            synthesis_params.input_examples
        )

        dataset_sample_sets = self._ingest_dataset_sources(synthesis_params.input_data)

        document_sample_sets = self._ingest_document_sources(
            synthesis_params.input_documents
        )

        permutable_attribute_samples = self._plan_permutable_attributes(
            synthesis_params.sampled_attributes,
            synthesis_params.combination_sampling,
            sample_count,
        )

        return self._create_dataset_plan(
            sample_count,
            permutable_attribute_samples,
            example_sample_sets,
            dataset_sample_sets,
            document_sample_sets,
        )

    def _create_dataset_plan(
        self,
        sample_count: int,
        permutable_attribute_samples: list[dict],
        example_sample_sets: list[list[dict]],
        dataset_sample_sets: list[list[dict]],
        document_sample_sets: list[list[dict]],
    ) -> list[dict]:
        """Create the final dataset plan."""
        samples = []
        for i in range(sample_count):
            sample = {}
            for example_set in example_sample_sets:
                index = i % len(example_set)
                sample.update(example_set[index])

            for dataset in dataset_sample_sets:
                index = i % len(dataset)
                sample.update(dataset[index])

            for document_set in document_sample_sets:
                index = i % len(document_set)
                sample.update(document_set[index])

            samples.append(sample)

        for i in range(len(permutable_attribute_samples)):
            samples[i].update(permutable_attribute_samples[i])

        if len(samples[-1].keys()) == 0:
            raise ValueError(
                "Empty sample created after planning, "
                "you must have at least one defined attribute for synthesis."
            )

        return samples

    def _ingest_example_sources(
        self,
        example_sources: Optional[list[ExampleSource]],
    ) -> list[list[dict]]:
        """Read examples from the example sources."""
        if example_sources is None or len(example_sources) == 0:
            return []

        results = [example_source.examples for example_source in example_sources]
        return results

    def _ingest_dataset_sources(
        self,
        dataset_sources: Optional[list[DatasetSource]],
    ) -> list[list[dict]]:
        """Read in datasets from the dataset sources."""
        if dataset_sources is None or len(dataset_sources) == 0:
            return []

        return [
            self._dataset_reader.read(dataset_source)
            for dataset_source in dataset_sources
        ]

    def _ingest_document_sources(
        self,
        document_sources: Optional[list[DocumentSource]],
    ) -> list[list[dict]]:
        """Read documents from the document sources and segment them if necessary."""
        if not document_sources:
            return []

        per_source_records = []
        for document_source in document_sources:
            records = []
            path = document_source.path
            documents = self._document_reader.read(path)
            if document_source.segmentation_params is None:
                for document in documents:
                    records.append({document_source.id: document})
            else:
                segmenter = DocumentSegmenter(document_source.segmentation_params)
                for document in documents:
                    segments = segmenter.segment(document)
                    for segment in segments:
                        record = {document_source.segmentation_params.id: segment}
                        if document_source.segmentation_params.keep_original_text:
                            record[document_source.id] = document
                        records.append(record)

            per_source_records.append(records)

        return per_source_records

    def _plan_permutable_attributes(
        self,
        permutable_attributes: Optional[list[SampledAttribute]],
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

        # Generate `sample_count` permutations of the permutable attributes
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

        samples = []
        for _ in range(sample_count):
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

            samples.append(sample_combination)

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
