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

import copy
import dataclasses
import hashlib
import json
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import jsonlines
from hdrh.histogram import HdrHistogram
from tqdm import tqdm

from oumi.core.configs import (
    GenerationParams,
    InferenceConfig,
    ModelParams,
)
from oumi.core.types.conversation import Conversation
from oumi.utils.logging import logger
from oumi.utils.math_utils import is_power_of_two


class BaseInferenceEngine(ABC):
    """Base class for running model inference."""

    _model_params: ModelParams
    """The model parameters."""

    _generation_params: GenerationParams
    """The generation parameters."""

    def __init__(
        self,
        model_params: ModelParams,
        *,
        generation_params: Optional[GenerationParams] = None,
    ):
        """Initializes the inference engine.

        Args:
            model_params: The model parameters.
            generation_params: The generation parameters.
        """
        self._model_params = copy.deepcopy(model_params)
        if generation_params:
            self._check_unsupported_params(generation_params)
        else:
            generation_params = GenerationParams()
        self._generation_params = generation_params

        self._latency_histogram_online = HdrHistogram(1, 60 * 1000, 1)
        self._latency_histogram_from_file = HdrHistogram(20, 180 * 1000, 1)
        self._dataset_hash = None

    def infer(
        self,
        input: Optional[list[Conversation]] = None,
        inference_config: Optional[InferenceConfig] = None,
    ) -> list[Conversation]:
        """Runs model inference.

        Args:
            input: A list of conversations to run inference on. Optional.
            inference_config: Parameters for inference.
                If not specified, a default config is inferred.

        Returns:
            List[Conversation]: Inference output.
        """
        if input is not None and (
            inference_config and inference_config.input_path is not None
        ):
            raise ValueError(
                "Only one of input or inference_config.input_path should be provided."
            )

        # Ensure the inference config has up-to-date generation parameters.
        if inference_config:
            if inference_config.generation:
                self._check_unsupported_params(inference_config.generation)
            elif self._generation_params:
                inference_config = copy.deepcopy(inference_config)
                inference_config.generation = self._generation_params

                # Warn the user: They provided an inference config without generation
                # params, so what was the point of providing it in the first place?
                logger.warning(
                    "No generation parameters provided in the inference config. Using "
                    "the generation parameters that the engine was initialized with."
                )

        output_path = inference_config.output_path if inference_config else None

        # Load input conversations either from file or use provided input
        conversations_to_process: list[Conversation] = []
        if input is not None:
            conversations_to_process = input
        elif inference_config and inference_config.input_path is not None:
            conversations_to_process = self._read_conversations(
                inference_config.input_path,
            )
        else:
            raise ValueError(
                "One of input or inference_config.input_path must be provided."
            )

        unique_conversation_ids = set()
        for i, conversation in enumerate(conversations_to_process):
            if conversation.conversation_id is not None:
                if conversation.conversation_id in unique_conversation_ids:
                    raise ValueError(
                        f"Conversation ID {conversation.conversation_id} is not unique."
                    )
                unique_conversation_ids.add(conversation.conversation_id)
                continue

            # Generate a deterministic conversation ID based on index if none exists
            messages = conversation.messages
            all_str_message_content = ",".join(
                [
                    message.content
                    for message in messages
                    if isinstance(message.content, str)
                ]
            )
            content_hash = hashlib.sha256(all_str_message_content.encode()).hexdigest()
            id_name = str(i) + "_" + content_hash

            conversation.conversation_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_DNS,
                    id_name,
                )
            )
            unique_conversation_ids.add(conversation.conversation_id)

        # Calculate the hash of the dataset to use for referencing the inference
        # filename between runs, given the same model and generation parameters.
        dataset_hash = ""
        if len(conversations_to_process) > 0:
            row_hashes = [
                hashlib.sha256(c.to_json().encode()).hexdigest()
                for c in conversations_to_process
            ]
            dataset_hash = hashlib.sha256(",".join(row_hashes).encode()).hexdigest()
        self._dataset_hash = dataset_hash

        completed_conversations = self._load_from_scratch(output_path)

        # Filter out already completed conversations
        remaining_conversations = self._filter_incomplete_conversations(
            conversations_to_process, completed_conversations
        )

        if len(remaining_conversations) < len(conversations_to_process):
            logger.info(
                f"Found {len(completed_conversations)} completed conversations. "
                f"Processing remaining {len(remaining_conversations)} conversations."
            )

        # Run inference only on remaining conversations
        start_time = time.perf_counter()
        histogram = self._latency_histogram_online
        inference_results = self._infer_online(
            remaining_conversations, inference_config
        )
        histogram.record_value((time.perf_counter() - start_time) * 1e3)
        self._maybe_log_latency_histogram(histogram)

        if len(inference_results) == len(conversations_to_process):
            final_results = inference_results
        else:
            # Incomplete inference results were saved to scratch file.
            # Load all results from scratch to get all results.
            final_results = self._load_from_scratch(output_path)
            if len(final_results) != len(conversations_to_process):
                raise ValueError(
                    f"Number of final results ({len(final_results)}) does not match "
                    f"number of conversations to process "
                    f"({len(conversations_to_process)})."
                )

        self._cleanup_scratch_file(output_path)

        sorted_conversations = {
            conv.conversation_id: conv for conv in conversations_to_process
        }

        for conv in final_results:
            sorted_conversations[conv.conversation_id] = conv

        final_results = list(sorted_conversations.values())

        if inference_config and inference_config.output_path:
            self._save_conversations(final_results, inference_config.output_path)

        return final_results

    def _maybe_log_latency_histogram(self, histogram: Optional[HdrHistogram]) -> None:
        """Logs the histogram if it is not None.

        Args:
            histogram: The histogram to log.
        """
        if histogram is None:
            return
        total_count = histogram.get_total_count()
        # TODO: Define a better way to enable/configure this logging.
        if not (
            isinstance(total_count, int)
            and total_count >= 2
            and is_power_of_two(total_count)
        ):
            return

        p50 = histogram.get_value_at_percentile(50)
        p90 = histogram.get_value_at_percentile(90)
        p99 = histogram.get_value_at_percentile(99)
        logger.debug(
            f"{self.__class__.__name__}: "
            f"Latency Histogram: {total_count} samples recorded:"
            f"\tp50: {p50:.1f}ms\tp90: {p90:.1f}ms\tp99: {p99:.1f}ms"
        )

    def _read_conversations(self, input_filepath: str) -> list[Conversation]:
        """Reads conversations from a file in Oumi chat format.

        Args:
            input_filepath: The path to the file containing the conversations.

        Returns:
            List[Conversation]: A list of conversations read from the file.
        """
        conversations = []
        with open(input_filepath) as f:
            for line in f:
                # Only parse non-empty lines.
                if line.strip():
                    conversation = Conversation.from_json(line)
                    conversations.append(conversation)
        return conversations

    def _load_from_scratch(self, output_filepath: Optional[str]) -> list[Conversation]:
        """Loads conversations from a scratch file.

        Args:
            output_filepath: The path to the output file. This is used to determine the
                location of the scratch file.

        Returns:
            list[Conversation]: A list of conversations loaded from the scratch file,
                or an empty list if the scratch file does not exist or is empty.
        """
        scratch_filepath = self._get_scratch_filepath(output_filepath)
        if not Path(scratch_filepath).exists():
            return []

        conversations = []
        with jsonlines.open(scratch_filepath, mode="r") as reader:
            for line in reader:
                conversations.append(Conversation.from_dict(line))
        return conversations

    def _filter_incomplete_conversations(
        self,
        input_conversations: list[Conversation],
        completed_conversations: list[Conversation],
    ) -> list[Conversation]:
        """Filters out conversations that have already been completed.

        Args:
            input_conversations: List of conversations to run inference on
            completed_conversations: List of conversations already completed

        Returns:
            list[Conversation]: List of conversations that still need inference results
        """
        completed_ids = {
            conv.conversation_id
            for conv in completed_conversations
            if conv.conversation_id is not None
        }
        return [
            conv
            for conv in input_conversations
            if conv.conversation_id not in completed_ids
        ]

    def _get_scratch_filepath(self, output_filepath: Optional[str]) -> str:
        """Returns a scratch filepath for the given output filepath.

        For example, if the output filepath is "/foo/bar/output.json", the scratch
        filepath will be "/foo/bar/scratch/output.json"

        If no output filepath is provided, a temporary file is used and placed in the
        current working directory under the name "tmp/temp_inference_output.jsonl".

        Args:
            output_filepath: The output filepath.

        Returns:
            str: The scratch filepath.
        """
        if output_filepath is not None:
            original_filepath = Path(output_filepath)
            return str(original_filepath.parent / "scratch" / original_filepath.name)

        model_params = self._model_params
        model_params_str = json.dumps(dataclasses.asdict(model_params))
        generation_params = self._generation_params
        generation_params_str = json.dumps(dataclasses.asdict(generation_params))
        inference_hash = hashlib.sha256(
            f"{model_params_str}_{generation_params_str}_{self._dataset_hash}".encode()
        ).hexdigest()

        path_prefix = Path.home() / ".cache" / "oumi" / "tmp"
        return str(path_prefix / f"temp_inference_output_{inference_hash}.jsonl")

    def _save_conversation_to_scratch(
        self, conversation: Conversation, output_filepath: Optional[str]
    ) -> None:
        """Appends a conversation to a file in Oumi chat format.

        Args:
            conversation: The conversation to save.
            output_filepath: The path to the file where the conversation should be
                saved.
        """
        # Make the directory if it doesn't exist.
        scratch_filepath = self._get_scratch_filepath(output_filepath)
        Path(scratch_filepath).parent.mkdir(parents=True, exist_ok=True)
        with jsonlines.open(scratch_filepath, mode="a") as writer:
            json_obj = conversation.to_dict()
            writer.write(json_obj)

    def _cleanup_scratch_file(self, output_filepath: Optional[str]) -> None:
        """Delete the scratch file from the file system if it exists.

        Args:
            output_filepath: The path to the output file. This is used to determine the
                location of the scratch file.
        """
        scratch_filepath = self._get_scratch_filepath(output_filepath)
        if Path(scratch_filepath).exists():
            Path(scratch_filepath).unlink()

    def _save_conversations(
        self, conversations: list[Conversation], output_filepath: str
    ) -> None:
        """Saves conversations to a file in Oumi chat format.

        Args:
            conversations: A list of conversations to save.
            output_filepath: The path to the file where the conversations should be
                saved.
        """
        # Make the directory if it doesn't exist.
        Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
        with jsonlines.open(output_filepath, mode="w") as writer:
            for conversation in tqdm(conversations, desc="Saving conversations"):
                json_obj = conversation.to_dict()
                writer.write(json_obj)

    def _check_unsupported_params(self, generation_params: GenerationParams):
        """Checks for unsupported parameters and logs warnings.

        If a parameter is not supported, and a non-default value is provided,
        a warning is logged.
        """
        supported_params = self.get_supported_params()
        default_generation_params = GenerationParams()

        for param_name, value in generation_params:
            if param_name not in supported_params:
                is_non_default_value = (
                    getattr(default_generation_params, param_name) != value
                )

                if is_non_default_value:
                    logger.warning(
                        f"{self.__class__.__name__} does not support {param_name}. "
                        f"Received value: {param_name}={value}. "
                        "This parameter will be ignored."
                    )

    @abstractmethod
    def get_supported_params(self) -> set[str]:
        """Returns a set of supported generation parameters for this engine.

        Override this method in derived classes to specify which parameters
        are supported.

        Returns:
            Set[str]: A set of supported parameter names.
        """
        raise NotImplementedError

    @abstractmethod
    def _infer_online(
        self,
        input: list[Conversation],
        inference_config: Optional[InferenceConfig] = None,
    ) -> list[Conversation]:
        """Runs model inference online.

        Args:
            input: A list of conversations to run inference on.
            inference_config: Parameters for inference.

        Returns:
            List[Conversation]: Inference output.
        """
        raise NotImplementedError

    def apply_chat_template(
        self, conversation: Conversation, **tokenizer_kwargs
    ) -> str:
        """Applies the chat template to the conversation.

        Args:
            conversation: The conversation to apply the chat template to.
            tokenizer_kwargs: Additional keyword arguments to pass to the tokenizer.

        Returns:
            str: The conversation with the chat template applied.
        """
        tokenizer = getattr(self, "_tokenizer", None)

        if tokenizer is None:
            raise ValueError("Tokenizer is not initialized.")

        if tokenizer.chat_template is None:
            raise ValueError("Tokenizer does not have a chat template.")

        if "tokenize" not in tokenizer_kwargs:
            tokenizer_kwargs["tokenize"] = False

        return tokenizer.apply_chat_template(conversation, **tokenizer_kwargs)
