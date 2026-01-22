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
        document_reader: DocumentReader | None = None,
        dataset_reader: DatasetReader | None = None,
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

        Source Types and Sampling Modes:
        - Example sources, Document sources, and Dataset sources support two modes:
          * Round-robin mode (default): When num_shots is None or 1, items are cycled
            sequentially across synthesis samples. Each item's attributes are spread
            directly into the sample dict (e.g., {field1: value1, field2: value2}).
            Reference in templates using the field name (e.g., {field1}).
          * Dynamic sampling mode: When num_shots > 1, N items are randomly sampled
            per synthesis sample. Items are stored as a list under the source's id
            (e.g., {source_id: [{field1: v1}, {field2: v2}]}). Reference in templates
            using bracket notation: {source_id[0].field1}.

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
            example_sources=synthesis_params.input_examples,
            dataset_sources=synthesis_params.input_data,
            document_sources=synthesis_params.input_documents,
        )

    def _create_dataset_plan(
        self,
        sample_count: int,
        permutable_attribute_samples: list[dict],
        example_sample_sets: list[list[dict]],
        dataset_sample_sets: list[list[dict]],
        document_sample_sets: list[list[dict]],
        example_sources: list[ExampleSource] | None = None,
        dataset_sources: list[DatasetSource] | None = None,
        document_sources: list[DocumentSource] | None = None,
    ) -> list[dict]:
        """Create the final dataset plan with optional dynamic sampling."""
        samples = []
        for i in range(sample_count):
            sample = {}

            # Handle example sources
            for ex_idx, example_set in enumerate(example_sample_sets):
                example_source = example_sources[ex_idx] if example_sources else None

                if (
                    example_source
                    and example_source.num_shots
                    and example_source.num_shots > 1
                ):
                    # Dynamic sampling mode
                    sampled = self._sample_items(example_set, example_source.num_shots)
                    sample[example_source.id] = sampled
                else:
                    # Round-robin mode (existing behavior)
                    index = i % len(example_set)
                    sample.update(example_set[index])

            # Handle dataset sources
            for ds_idx, dataset in enumerate(dataset_sample_sets):
                dataset_source = dataset_sources[ds_idx] if dataset_sources else None

                if (
                    dataset_source
                    and dataset_source.num_shots
                    and dataset_source.num_shots > 1
                ):
                    # Dynamic sampling mode
                    sampled = self._sample_items(dataset, dataset_source.num_shots)
                    sample[dataset_source.id] = sampled
                else:
                    # Round-robin mode (existing behavior)
                    index = i % len(dataset)
                    sample.update(dataset[index])

            # Handle document sources
            for doc_idx, document_set in enumerate(document_sample_sets):
                document_source = (
                    document_sources[doc_idx] if document_sources else None
                )

                if (
                    document_source
                    and document_source.num_shots
                    and document_source.num_shots > 1
                ):
                    # Dynamic sampling mode
                    sampled = self._sample_items(
                        document_set, document_source.num_shots
                    )
                    sample[document_source.id] = sampled
                else:
                    # Round-robin mode (existing behavior)
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
        example_sources: list[ExampleSource] | None,
    ) -> list[list[dict]]:
        """Read examples from the example sources."""
        if example_sources is None or len(example_sources) == 0:
            return []

        results = [example_source.examples for example_source in example_sources]
        return results

    def _ingest_dataset_sources(
        self,
        dataset_sources: list[DatasetSource] | None,
    ) -> list[list[dict]]:
        """Read in datasets from the dataset sources."""
        if dataset_sources is None or len(dataset_sources) == 0:
            return []

        results = []
        for dataset_source in dataset_sources:
            dataset = self._dataset_reader.read(dataset_source)
            if not dataset:
                raise ValueError(
                    f"Dataset source '{dataset_source.path}' is empty. "
                    "Please check the file or apply different filters."
                )
            results.append(dataset)
        return results

    def _ingest_document_sources(
        self,
        document_sources: list[DocumentSource] | None,
    ) -> list[list[dict]]:
        """Read documents from the document sources and segment them if necessary."""
        if not document_sources:
            return []

        per_source_records = []
        for document_source in document_sources:
            records = []
            path = document_source.path
            documents = self._document_reader.read(path)
            non_empty_documents = [
                document
                for document in documents
                if document and document.strip() != ""
            ]
            if not non_empty_documents:
                raise ValueError(
                    "No non-empty documents were found in the document source, "
                    "please check the document source."
                )

            if document_source.segmentation_params is None:
                for document in non_empty_documents:
                    records.append({document_source.id: document})
            else:
                segmenter = DocumentSegmenter(document_source.segmentation_params)
                for document in non_empty_documents:
                    segments = segmenter.segment(document)
                    non_empty_segments = [
                        segment
                        for segment in segments
                        if segment and segment.strip() != ""
                    ]

                    if not non_empty_segments:
                        raise ValueError(
                            "Document segmentation returned only empty segments."
                        )

                    for segment in non_empty_segments:
                        record = {document_source.segmentation_params.id: segment}
                        if document_source.segmentation_params.keep_original_text:
                            record[document_source.id] = document
                        records.append(record)

            if not records:
                raise ValueError(
                    "No records were created from document source, check to ensure "
                    "that at least one document is not empty."
                )

            per_source_records.append(records)

        return per_source_records

    def _sample_items(
        self,
        items: list[dict],
        num_shots: int,
    ) -> list[dict]:
        """Randomly sample items.

        Args:
            items: List of items to sample from.
            num_shots: Number of items to sample.

        Returns:
            Sampled list of items.
        """
        # Sample without replacement
        sampled_indices = random.sample(range(len(items)), min(num_shots, len(items)))
        sampled = [items[idx] for idx in sampled_indices]

        return sampled

    def _plan_permutable_attributes(
        self,
        permutable_attributes: list[SampledAttribute] | None,
        combination_sampling: list[AttributeCombination] | None,
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
