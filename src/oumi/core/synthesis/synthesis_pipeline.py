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

import re
from pathlib import Path
from typing import Any

from oumi.core.configs.synthesis_config import SynthesisConfig
from oumi.core.synthesis.attribute_synthesizer import AttributeSynthesizer
from oumi.core.synthesis.attribute_transformation import AttributeTransformer
from oumi.core.synthesis.data_synthesizer import DataSynthesizer
from oumi.core.synthesis.dataset_planner import DatasetPlanner
from oumi.utils.io_utils import save_jsonlines
from oumi.utils.logging import logger


class SynthesisPipeline:
    """Pipeline for synthesizing a dataset."""

    def __init__(self, config: SynthesisConfig):
        """Initialize the synthesis pipeline."""
        self._config = config
        attribute_synthesizer = AttributeSynthesizer(
            config.strategy_params, config.inference_config
        )
        self._attribute_transformer = AttributeTransformer(config.strategy_params)
        self._dataset_planner = DatasetPlanner()
        self._data_synthesizer = (
            DataSynthesizer(
                config.strategy_params.generated_attributes,
                attribute_synthesizer,
            )
            if config.strategy_params.generated_attributes
            else None
        )

    def synthesize(self) -> list[dict[str, Any]]:
        """Synthesize a dataset."""
        # Populate the dataset plan with column values for each non-generated attribute
        logger.info(
            f"Loading dependencies to synthesize dataset with "
            f"{self._config.num_samples} samples"
        )
        dataset = self._dataset_planner.plan(
            self._config.strategy_params,
            self._config.num_samples,
        )

        # Synthesize the generated attributes
        logger.info("Synthesizing generated attributes")
        if self._data_synthesizer:
            dataset = self._data_synthesizer.synthesize(dataset)

        # Add the transformed attributes to the dataset
        logger.info("Adding transformed attributes")
        if self._config.strategy_params.transformed_attributes:
            dataset = self._attribute_transformer.transform(dataset)

        # If passthrough attributes are specified, keep only those attributes
        logger.info("Keeping passthrough attributes")
        if self._config.strategy_params.passthrough_attributes:
            dataset = self._passthrough_attributes(dataset)

        # Save the dataset to the output path
        logger.info("Saving dataset")
        if self._config.output_path:
            self._save_dataset(dataset)

        logger.info("Synthesis complete")
        return dataset

    def _passthrough_attributes(
        self,
        dataset: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Keep only the passthrough attributes in the dataset.

        Supports both simple keys and bracket notation for nested access:
        - Simple: "conversation" -> sample["conversation"]
        - Bracket: "examples[0].field" -> sample["examples"][0]["field"]
        """
        if not self._config.strategy_params.passthrough_attributes:
            return dataset

        passthrough_attributes = self._config.strategy_params.passthrough_attributes

        # Separate simple keys from bracket notation paths
        simple_keys = set()
        bracket_paths = []

        for attr in passthrough_attributes:
            if "[" in attr and "]" in attr:
                bracket_paths.append(attr)
            else:
                simple_keys.add(attr)

        result = []
        for sample in dataset:
            filtered_sample = {}

            # Add simple passthrough attributes
            for key in simple_keys:
                if key in sample:
                    filtered_sample[key] = sample[key]

            # Add bracket notation attributes
            for path in bracket_paths:
                try:
                    value = self._extract_nested_value(sample, path)
                    # Store using the full path as the key
                    filtered_sample[path] = value
                except (KeyError, IndexError, ValueError):
                    # Skip if path doesn't exist in sample
                    pass

            result.append(filtered_sample)

        return result

    def _extract_nested_value(self, sample: dict[str, Any], path: str) -> Any:
        """Extract a value from a nested structure using bracket notation.

        Args:
            sample: The sample dictionary to extract from.
            path: Path like "examples[0].field" or "data[1].nested.value"

        Returns:
            The extracted value.

        Raises:
            KeyError: If a key doesn't exist.
            IndexError: If an index is out of range.
            ValueError: If the path format is invalid.
        """
        # Parse the path: "examples[0].field" -> ["examples", "[0]", "field"]
        # Match: word, [index], or .word
        pattern = r"([^\[\].]+|\[\d+\])"
        parts = re.findall(pattern, path)

        current: Any = sample
        for part in parts:
            if part.startswith("[") and part.endswith("]"):
                # Array index access
                index = int(part[1:-1])
                if not isinstance(current, list):
                    raise ValueError(
                        f"Cannot index into non-list type {type(current).__name__}"
                    )
                current = current[index]
            else:
                # Dictionary key access
                if isinstance(current, dict):
                    current = current[part]
                else:
                    raise ValueError(
                        f"Cannot access key '{part}' on non-dict type "
                        f"{type(current).__name__}"
                    )

        return current

    def _save_dataset(self, dataset: list[dict[str, Any]]):
        """Save the dataset to the output path."""
        if not self._config.output_path:
            raise ValueError("SynthesisConfig.output_path is not specified.")

        path_str = self._config.output_path
        path = Path(path_str)
        parent = path.parent
        if not parent.exists():
            parent.mkdir(parents=True)

        if path.suffix == ".jsonl":
            save_jsonlines(path, dataset)
        else:
            raise ValueError(f"Unsupported output path: {path_str}")
