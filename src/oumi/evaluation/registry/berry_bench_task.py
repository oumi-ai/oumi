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

import json
import re
from typing import Any, Optional

from oumi.core.configs.params.evaluation_params import EvaluationTaskParams
from oumi.core.inference.base_inference_engine import BaseInferenceEngine
from oumi.core.registry import register_evaluation_function
from oumi.datasets.grpo.berry_bench import BerryBenchGrpoDataset
from oumi.utils.logging import logger


def _extract_json(response: str) -> Optional[dict]:
    r"""Returns the json answer extracted from ```json ...```, or None otherwise."""
    logger.info(f"response: {response}")
    # re.DOTALL lets '.' match newlines. Most LLMs use newlines in their JSON outputs.
    regex_result = re.findall("```json(.*)```", response, re.DOTALL)
    logger.info(f"result: {regex_result}")
    if not regex_result or len(regex_result) != 1:
        return None
    json_str = regex_result[0]
    try:
        return json.loads(json_str)
    except json.decoder.JSONDecodeError:
        return None


@register_evaluation_function("berry_bench")
def berry_bench(
    task_params: EvaluationTaskParams,
    inference_engine: BaseInferenceEngine,
) -> dict[str, Any]:
    """Custom evaluation function registered as `berry_bench`."""
    dataset = BerryBenchGrpoDataset(split="test")
    num_samples = task_params.num_samples
    if num_samples is None:
        num_samples = len(dataset)
    input_conversations = [dataset.conversation(i) for i in range(num_samples)]
    conversations = inference_engine.infer(input_conversations)
    logger.info(f"Finished inference on {len(conversations)} conversations!")
    if len(conversations) > 0:
        logger.info(f"Sample conversation: {conversations[0]}")

    count = 0  # The number of examples with correct answers extracted.
    total = 0  # All examples.
    valid_count = 0  # The number of examples with valid answers extracted.
    for i, conversation in enumerate(conversations):
        total += 1
        # Grab the model's response
        response = conversation.last_message()
        # Ignore cases where model didn't respond or it's a multimodal response.
        # For now, we focus on text-only responses.
        if not response or not isinstance(response.content, str):
            continue
        # Count the example as correct if the extracted prediction is correct.
        prediction = _extract_json(response.content)
        if prediction is None:
            continue
        valid_count += 1
        expected_json_str = conversation.metadata["expected_response"]
        expected_json = json.loads(expected_json_str)
        if prediction == expected_json:
            count += 1

    return {
        # Accuracy across all examples.
        "accuracy": count / total if total > 0 else 0,
        # Accuracy when only counting examples with properly extracted answers.
        "properly_extracted_accuracy": count / valid_count if valid_count > 0 else 0,
        "num_samples": num_samples,
        # These three values sum up to num_samples.
        "num_correct_answers": count,
        "num_incorrect_answers": valid_count - count,
        "num_invalid_answers": total - valid_count,
    }
