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

from oumi.builders.inference_engines import build_inference_engine
from oumi.core.configs.inference_config import InferenceConfig
from oumi.core.configs.inference_engine_type import InferenceEngineType
from oumi.core.configs.params.synthesis_params import (
    GeneralSynthesisParams,
    GeneratedAttribute,
    GeneratedAttributePostprocessingParams,
    TextMessage,
)
from oumi.core.synthesis.attribute_formatter import AttributeFormatter
from oumi.core.types.conversation import Conversation, Message
from oumi.inference.remote_inference_engine import BatchInfo
from oumi.utils.logging import logger


class AttributeSynthesizer:
    """Synthesizes values for a generated attribute based on the given samples.

    Args:
        params: The parameters for the attribute synthesizer.
        inference_config: The configuration for the inference engine.
    """

    def __init__(
        self,
        params: GeneralSynthesisParams,
        inference_config: InferenceConfig,
    ):
        """Initialize the synthesizer."""
        self._params = params
        self._formatter = AttributeFormatter(params)

        self._inference_engine = build_inference_engine(
            engine_type=inference_config.engine or InferenceEngineType.NATIVE,
            model_params=inference_config.model,
            remote_params=inference_config.remote_params,
        )
        self._inference_config = inference_config

    def synthesize(
        self,
        samples: list[dict],
        generated_attribute: GeneratedAttribute,
    ) -> list[dict[str, str]]:
        """Synthesize a value for the generated attribute.

        Order will be identical to the order of the samples.

        Args:
            samples: The samples to synthesize values for.
            generated_attribute: The generated attribute to synthesize a value for.

        Returns:
            A list of dictionaries, one for each sample, with the generated attribute
            value added to the dictionary.
        """
        inference_conversations: list[Conversation] = []
        for sample in samples:
            inference_conversations.append(
                self._format_instructions(
                    sample,
                    generated_attribute.instruction_messages,
                )
            )

        inference_results = self._inference_engine.infer(
            inference_conversations,
            inference_config=self._inference_config,
        )

        return self._process_inference_results(inference_results, generated_attribute)

    def synthesize_batch(
        self,
        samples: list[dict],
        generated_attribute: GeneratedAttribute,
    ) -> str:
        """Submit a batch inference job for attribute synthesis.

        Args:
            samples: The samples to synthesize values for.
            generated_attribute: The generated attribute to synthesize a value for.

        Returns:
            The batch ID that can be used with get_batch_status and get_batch_results.

        Raises:
            NotImplementedError: If the inference engine does not support batch.
        """
        if not hasattr(self._inference_engine, "infer_batch"):
            raise NotImplementedError(
                f"Inference engine {type(self._inference_engine).__name__} does not "
                "support batch inference. Use synthesize() instead."
            )

        inference_conversations: list[Conversation] = []
        for sample in samples:
            inference_conversations.append(
                self._format_instructions(
                    sample,
                    generated_attribute.instruction_messages,
                )
            )

        batch_id = self._inference_engine.infer_batch(  # type: ignore[attr-defined]
            inference_conversations,
            inference_config=self._inference_config,
        )
        return batch_id

    def get_batch_status(self, batch_id: str) -> BatchInfo:
        """Get the status of a batch inference job.

        Args:
            batch_id: The batch ID returned from synthesize_batch().

        Returns:
            BatchInfo containing the job status and progress information.

        Raises:
            NotImplementedError: If the inference engine does not support batch.
        """
        if not hasattr(self._inference_engine, "get_batch_status"):
            raise NotImplementedError(
                f"Inference engine {type(self._inference_engine).__name__} does not "
                "support batch inference."
            )
        return self._inference_engine.get_batch_status(batch_id)  # type: ignore[attr-defined]

    def get_batch_results(
        self,
        batch_id: str,
        samples: list[dict],
        generated_attribute: GeneratedAttribute,
    ) -> list[dict[str, str]]:
        """Get results from a completed batch inference job.

        Args:
            batch_id: The batch ID returned from synthesize_batch().
            samples: The original samples that were submitted for synthesis.
            generated_attribute: The generated attribute configuration.

        Returns:
            A list of dictionaries, one for each sample, with the generated attribute
            value added to the dictionary (same format as synthesize()).

        Raises:
            NotImplementedError: If the inference engine does not support batch.
        """
        if not hasattr(self._inference_engine, "get_batch_results"):
            raise NotImplementedError(
                f"Inference engine {type(self._inference_engine).__name__} does not "
                "support batch inference."
            )

        inference_conversations: list[Conversation] = []
        for sample in samples:
            inference_conversations.append(
                self._format_instructions(
                    sample,
                    generated_attribute.instruction_messages,
                )
            )

        inference_results = self._inference_engine.get_batch_results(  # type: ignore[attr-defined]
            batch_id, inference_conversations
        )

        return self._process_inference_results(inference_results, generated_attribute)

    def _extract_response(
        self,
        inference_conversations: list[Conversation],
    ) -> list[str]:
        """Get the inference results from the inference conversations.

        If the inference result is not a string, an empty string will be returned.
        """
        return [
            inference_result.messages[-1].content
            if isinstance(inference_result.messages[-1].content, str)
            else ""
            for inference_result in inference_conversations
        ]

    def _format_instructions(
        self,
        sample: dict,
        instruction_messages: list[TextMessage],
    ) -> Conversation:
        """Format the instructions for the sample."""
        new_messages = []
        for turn in instruction_messages:
            if not isinstance(turn.content, str):
                new_messages.append(turn)
                continue

            formatted_content = self._formatter.format(
                sample,
                turn.content,
                missing_values_allowed=False,
            )
            new_message = Message(
                role=turn.role,
                content=formatted_content,
            )
            new_messages.append(new_message)

        return Conversation(messages=new_messages)

    def _process_inference_results(
        self,
        inference_results: list[Conversation],
        generated_attribute: GeneratedAttribute,
    ) -> list[dict[str, str]]:
        """Extract and postprocess inference results.

        Args:
            inference_results: The inference results from the inference engine.
            generated_attribute: The generated attribute configuration.

        Returns:
            A list of dictionaries with the processed attribute values.
        """
        original_responses = self._extract_response(inference_results)

        if not generated_attribute.postprocessing_params:
            return [
                {generated_attribute.id: response} for response in original_responses
            ]

        keep_original = (
            generated_attribute.postprocessing_params.keep_original_text_attribute
        )
        if keep_original:
            records = [
                {generated_attribute.id: response} for response in original_responses
            ]
        else:
            records = [{} for _ in original_responses]

        for i, original_response in enumerate(original_responses):
            new_id = generated_attribute.postprocessing_params.id
            new_response = original_response
            try:
                new_response = self._postprocess_sample(
                    original_response, generated_attribute.postprocessing_params
                )
            except ValueError as e:
                logger.warning(
                    f"Error postprocessing inference result: {e}. Leaving as-is and "
                    "skipping."
                )
            finally:
                records[i][new_id] = new_response

        return records

    def _postprocess_sample(
        self,
        response: str,
        postprocessing_params: GeneratedAttributePostprocessingParams,
    ) -> str:
        """Postprocess the response, removing extraneous text.

        Order of operations:
        1. If regex is provided, use the first match.
        2. Cut off everything before the first occurrence of the prefix and after the
        last occurrence of the suffix.
        3. Strip whitespace.
        4. Add prefix and suffix to what remains.

        Args:
            response: The response to postprocess.
            postprocessing_params: The postprocessing parameters.

        Returns:
            The postprocessed response.
        """
        if postprocessing_params.regex:
            match = re.search(postprocessing_params.regex, response)
            if match:
                response = match.group(0)

        # Cut off prefix and suffix
        if postprocessing_params.cut_prefix:
            prefix_loc = response.find(postprocessing_params.cut_prefix)
            if prefix_loc != -1:
                response = response[
                    prefix_loc + len(postprocessing_params.cut_prefix) :
                ]
        if postprocessing_params.cut_suffix:
            suffix_loc = response.rfind(postprocessing_params.cut_suffix)
            if suffix_loc != -1:
                response = response[:suffix_loc]

        # Strip whitespace
        if postprocessing_params.strip_whitespace:
            response = response.strip()

        # Add prefix and suffix
        if postprocessing_params.added_prefix:
            response = postprocessing_params.added_prefix + response
        if postprocessing_params.added_suffix:
            response = response + postprocessing_params.added_suffix

        return response
