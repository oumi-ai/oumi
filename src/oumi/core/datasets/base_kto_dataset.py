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

"""Base dataset class for KTO (Kahneman-Tversky Optimization).

This module provides a base class for datasets used in KTO training.
Unlike DPO which requires preference pairs, KTO works with simple binary feedback
indicating whether an output is desirable or undesirable.
"""

from collections.abc import Iterable
from typing import Any, Optional

import datasets
from typing_extensions import override

from oumi.core.datasets.base_map_dataset import BaseMapDataset, _InferredFeatureMap

_PROMPT_KEY = "prompt"
_COMPLETION_KEY = "completion"
_LABEL_KEY = "label"  # True for desirable, False for undesirable

_ROLE = "role"
_CONTENT = "content"
_ASSISTANT = "assistant"


class BaseExperimentalKtoDataset(BaseMapDataset):
    """Base class for KTO (Kahneman-Tversky Optimization) datasets.

    This class provides a comprehensive foundation for creating KTO datasets that work
    with binary feedback signals rather than preference pairs. KTO is an alignment
    method that optimizes language models based on simple binary labels indicating
    whether outputs are desirable or undesirable, making it simpler than preference-
    based methods like DPO which require paired comparisons.

    The class handles the standardization of diverse dataset formats into the
    consistent KTO format required by training frameworks. It supports both
    string-based completions and chat-formatted conversations, automatically
    extracting assistant responses when needed.

    Key Features:
        - Standardized KTO format with prompt, completion, and binary label
        - Automatic handling of chat format vs string format completions
        - Optimized feature schema for efficient dataset processing
        - Memory-efficient processing for large datasets
        - Consistent API across different KTO dataset implementations

    Dataset Format:
        The standardized KTO format includes:
        - prompt (str): The input text given to the model
        - completion (str): The model's response to be evaluated
        - label (bool): True for desirable responses, False for undesirable ones

    Usage:
        Subclasses should implement the `_load_data()` method to load their specific
        dataset format and optionally override `_transform_kto_example()` for custom
        preprocessing logic.

    Warning:
        This class is experimental and subject to change as KTO training methods
        evolve and mature.

    See Also:
        - TRL KTO Trainer: https://huggingface.co/docs/trl/main/en/kto_trainer
        - KTO Paper: https://arxiv.org/abs/2402.01306
    """

    def __init__(
        self,
        *,
        dataset_name: Optional[str] = None,
        dataset_path: Optional[str] = None,
        split: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialize a new KTO dataset instance.

        Creates a new KTO dataset that will load and transform data into the
        standardized KTO format required for training. The constructor handles
        dataset discovery, loading, and initial setup but defers actual data
        loading until needed.

        Args:
            dataset_name (Optional[str]): Name of the dataset to load from Hugging Face
                Hub or a custom identifier. If None, uses the class's default_dataset.
            dataset_path (Optional[str]): Local path to dataset files. Takes precedence
                over dataset_name if provided. Supports .jsonl, .parquet, and cached
                HF dataset formats.
            split (Optional[str]): Dataset split to use (e.g., 'train', 'test',
                'validation'). If None and multiple splits exist, an error will be
                raised.
            **kwargs: Additional keyword arguments passed to the parent BaseMapDataset
                constructor for extended configuration options.

        Raises:
            ValueError: If neither dataset_name nor default_dataset is provided.
            FileNotFoundError: If dataset_path is specified but the file doesn't exist.

        Note:
            The actual dataset loading is deferred until the first access to ensure
            efficient initialization and allow for lazy loading patterns.
        """
        super().__init__(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            split=split,
            **kwargs,
        )

        self._data = self._load_data()

    def _transform_kto_example(self, sample: dict) -> dict:
        """Validate and transform a sample to the KTO format.

        This method processes raw dataset samples and converts them into the
        standardized format required for KTO training. Unlike DPO which requires
        preference pairs, KTO works with simple binary feedback indicating whether
        an output is desirable or undesirable.

        The method extracts the prompt, completion, and binary label from the
        input sample, handling both string and chat format completions. Chat format
        completions are processed to extract the assistant's response.

        Args:
            sample (dict): A dictionary containing the raw sample data with keys:
                - prompt: The input prompt text
                - completion: The model's response (string or chat format list)
                - label: Boolean indicating if the response is desirable (True) or
                  undesirable (False)

        Returns:
            dict: A dictionary with the standardized KTO format expected by TRL:
                - prompt (str): The input prompt text
                - completion (str): The model's response text
                - label (bool): Binary feedback - True for desirable responses,
                  False for undesirable responses

        Raises:
            ValueError: If the completion is in chat format but no assistant turn
                is found.
        """
        prompt = sample[_PROMPT_KEY]
        completion = sample[_COMPLETION_KEY]
        label = sample[_LABEL_KEY]

        # Extract text from completion if it's in chat format
        if isinstance(completion, list):
            completion = self._extract_from_chat_format(completion)

        return {
            _PROMPT_KEY: prompt,
            _COMPLETION_KEY: completion,
            _LABEL_KEY: label,
        }

    @override
    def transform(self, sample: dict) -> dict:
        """Transform a raw dataset sample into the standardized format.

        This is the main entry point for processing dataset samples. It delegates
        to the KTO-specific transformation method to ensure consistent formatting
        across all KTO datasets.

        Args:
            sample (dict): A raw dataset sample containing prompt, completion,
                and label information.

        Returns:
            dict: The transformed sample in KTO format with standardized keys:
                prompt, completion, and label.
        """
        return self._transform_kto_example(sample)

    def _extract_from_chat_format(self, sample) -> str:
        """Extract the assistant's response from a chat-formatted completion.

        This method handles completions that are provided in chat format (as a list
        of message dictionaries) by extracting the content from the last assistant
        turn. If the sample is already a string, it returns it unchanged.

        The method searches backwards through the conversation to find the most recent
        assistant response, which is typically the completion we want to evaluate.

        Args:
            sample: The completion data, either as a string or a list of message
                dictionaries in chat format. Each message dict should have 'role'
                and 'content' keys.

        Returns:
            str: The extracted assistant response content as a string.

        Raises:
            ValueError: If the sample is in chat format (list) but no message
                with role 'assistant' is found.

        Example:
            >>> chat_format = [
            ...     {"role": "user", "content": "Hello"},
            ...     {"role": "assistant", "content": "Hi there!"}
            ... ]
            >>> self._extract_from_chat_format(chat_format)
            'Hi there!'
        """
        if not isinstance(sample, list):
            return sample

        for turn in sample[::-1]:
            if turn[_ROLE] == _ASSISTANT:
                return turn[_CONTENT]

        raise ValueError("No chat turn was found with an 'assistant' role.")

    @property
    def _kto_features(self) -> datasets.Features:
        """Get the explicit feature schema definition for KTO training datasets.

        This property defines the standardized schema that all KTO datasets must
        conform to. The schema ensures type safety and consistency across different
        KTO dataset implementations and is used by the Hugging Face datasets library
        for efficient serialization and validation.

        The KTO format requires exactly three fields:
        - prompt: The input text that was given to the model
        - completion: The model's generated response to be evaluated
        - label: Binary feedback indicating if the completion is desirable

        Returns:
            datasets.Features: A Features object defining the schema with:
                - prompt (string): Input prompt text
                - completion (string): Model response text
                - label (bool): True for desirable responses, False for undesirable
        """
        return datasets.Features(
            {
                "prompt": datasets.Value("string"),
                "completion": datasets.Value("string"),
                "label": datasets.Value("bool"),
            }
        )

    @override
    def _detect_features_and_estimate_element_size_bytes(
        self, samples_iter: Iterable[dict[str, Any]]
    ) -> _InferredFeatureMap:
        """Detect dataset features and estimate memory requirements for KTO datasets.

        This method overrides the base implementation to provide KTO-specific feature
        detection and memory estimation. Instead of inferring features from sample data,
        it uses the predefined KTO schema to ensure consistency and optimize for the
        specific structure of KTO training data.

        The method samples a subset of the dataset to estimate average element size,
        which is used for memory management and efficient dataset processing. The
        feature map is marked as optimized since we use explicit KTO features rather
        than inferred ones.

        Args:
            samples_iter (Iterable[dict[str, Any]]): An iterable of dataset samples
                to analyze for feature detection and size estimation.

        Returns:
            _InferredFeatureMap: A named tuple containing:
                - feature_map: The explicit KTO features schema
                - is_feature_map_optimized: Always True for KTO datasets
                - element_size_in_bytes: Estimated average size per sample
                - multimodal: Always False for text-only KTO data
        """
        from oumi.utils.torch_utils import estimate_sample_dict_size_in_bytes

        # Collect a few samples to estimate average size
        samples = []
        samples_iterator = iter(samples_iter)
        for _ in range(min(10, len(self))):  # Use up to 10 samples
            try:
                samples.append(next(samples_iterator))
            except StopIteration:
                break

        # Calculate estimated element size based on actual samples
        element_size = 1024  # Default fallback
        if samples:
            # Get average size of samples
            element_size = sum(
                estimate_sample_dict_size_in_bytes(s) for s in samples
            ) // len(samples)
            # Add 20% buffer for safety
            element_size = int(element_size * 1.2)

        # Return features optimized for KTO training with proper size estimate
        return _InferredFeatureMap(
            feature_map=self._kto_features,
            is_feature_map_optimized=True,
            element_size_in_bytes=element_size,
            multimodal=False,
        )
