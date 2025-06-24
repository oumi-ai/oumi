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


from pathlib import Path
from typing import Optional, Union

from oumi.core.configs.judge_config_v2 import JudgeConfig
from oumi.judges_v2.base_judge import JudgeOutput
from oumi.judges_v2.simple_judge import SimpleJudge
from oumi.utils.io_utils import load_jsonlines


def judge_dataset(
    judge_config: Union[JudgeConfig, str],
    dataset: list[dict[str, str]],
    output_file: Optional[Union[str, Path]] = None,
) -> list[JudgeOutput]:
    """Judge a dataset using Oumi's Judge framework.

    This function evaluates a dataset by instantiating a SimpleJudge with the provided
    configuration and running batch inference on all input data.

    The function performs the following steps:
        1. Initializes a SimpleJudge with the provided configuration.
        2. Passes the entire dataset to the judge for batch evaluation.
        3. Returns structured JudgeOutput objects containing parsed results.

    Args:
        judge_config: JudgeConfig object or path to a judge config file.
        dataset: List of dictionaries containing input data for evaluation. Each
            dictionary should contain key-value pairs that match placeholders in
            the judge's prompt template (e.g., {'question': '...', 'answer': '...'}).
        output_file: Optional path to save the judge results as a JSONL file.
            If provided, the results will be saved to this file.

    Returns:
        List[JudgeOutput]: A list of structured judgment results, each containing:
            - raw_output: The original response from the judge model
            - parsed_output: Extracted field values from structured formats (XML/JSON)
            - field_values: Typed values for each expected output field
            - field_scores: Numeric scores for applicable fields

    Example:
        >>> judge_config = JudgeConfig( # doctest: +SKIP
        ...     judge_params=JudgeParams(
        ...         prompt_template="Is this helpful? {question}, {answer}",
        ...         response_format=JudgeResponseFormat.XML,
        ...         judgment_type=JudgeOutputType.BOOL,
        ...         include_explanation=False
        ...     ),
        ...     inference_config=InferenceConfig(
        ...         model=ModelParams(model_name="gpt-4.1"),
        ...         generation=GenerationParams(max_tokens=100),
        ...         engine=InferenceEngineType.OPENAI
        ...     )
        ... )
        >>> dataset = [
        ...     {'question': 'What is 2+2?', 'answer': '4'},
        ...     {'question': 'How to cook?', 'answer': 'I dont know'}
        ... ]
        >>> judged_outputs = judge_dataset(judge_config, dataset)
        >>> for output in judged_outputs:
        ...     print(output.field_values)  # e.g., {'judgment': True}
    """
    judge = SimpleJudge(judge_config=judge_config)
    judge_outputs = judge.judge(inputs=dataset)

    # Save `judge_outputs` into a file, if an `output_file` was provided
    if output_file:
        with open(output_file, "w") as f:
            for judge_output in judge_outputs:
                f.write(judge_output.to_json() + "\n")

    return judge_outputs


def judge_file(
    judge_config: Union[JudgeConfig, str],
    input_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
) -> list[JudgeOutput]:
    """Judge a dataset from a JSONL file using Oumi's Judge framework.

    This is a convenience wrapper around judge_dataset. It loads the dataset from a
        JSONL file and then calls judge_dataset to perform the evaluation.

    Args:
        judge_config: JudgeConfig object or path to a judge config.
        input_file: Path to the input JSONL file containing the dataset.
        output_file: Optional path to save the judge results as a JSONL file.
            If provided, the results will be saved to this file.

    Returns:
        List[JudgeOutput]: A list of structured judgment results, each containing:
            - raw_output: The original response from the judge model
            - parsed_output: Extracted field values from structured formats (XML/JSON)
            - field_values: Typed values for each expected output field
            - field_scores: Numeric scores for applicable fields

    Raises:
        FileNotFoundError: If the input file doesn't exist.
    """
    dataset = load_jsonlines(input_file)
    return judge_dataset(
        judge_config=judge_config,
        dataset=dataset,
        output_file=output_file,
    )
