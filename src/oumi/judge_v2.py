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


from oumi.core.configs.inference_config import InferenceConfig
from oumi.core.configs.judge_config_v2 import JudgeConfig
from oumi.judges_v2.base_judge import JudgeOutput
from oumi.judges_v2.oumi_judge import OumiJudge


def judge_dataset(
    judge_config: JudgeConfig,
    inference_config: InferenceConfig,
    dataset: list[dict[str, str]],
) -> list[JudgeOutput]:
    """Judge a dataset using the Oumi Judge framework.

    This function evaluates a dataset by instantiating an OumiJudge with the provided
    configuration and running batch inference on all input data.

    The function performs the following steps:
        1. Initializes an OumiJudge with the provided configuration.
        2. Passes the entire dataset to the judge for batch evaluation.
        3. Returns structured JudgeOutput objects containing parsed results.

    Args:
        judge_config: The configuration for the judge, including prompt template,
            response format, and output field specifications.
        inference_config: The configuration for inference, including model settings,
            generation parameters, and engine type.
        dataset: List of dictionaries containing input data for evaluation. Each
            dictionary should contain key-value pairs that match placeholders in
            the judge's prompt template (e.g., {'question': '...', 'answer': '...'}).

    Returns:
        List[JudgeOutput]: A list of structured judgment results, each containing:
            - raw_output: The original response from the judge model
            - parsed_output: Extracted field values from structured formats (XML/JSON)
            - field_values: Typed values for each expected output field
            - field_scores: Numeric scores for applicable fields

    Example:
        >>> judge_config = JudgeConfig(
        ...     prompt_template="Is this answer helpful? "
        ...                     "Question: {question} Answer: {answer}",
        ...     judgment_type=JudgeOutputType.BOOL,
        ...     response_format=JudgeResponseFormat.JSON
        ...     ...
        ... )
        >>> dataset = [
        ...     {'question': 'What is 2+2?', 'answer': '4'},
        ...     {'question': 'How to cook?', 'answer': 'I dont know'}
        ... ]
        >>> judged_outputs = judge_dataset(judge_config, inference_config, dataset)
        >>> for output in judged_outputs:
        ...     print(output.field_values)  # e.g., {'judgment': True}
    """
    judge = OumiJudge(judge_config=judge_config, inference_config=inference_config)
    return judge.judge(inputs=dataset)
