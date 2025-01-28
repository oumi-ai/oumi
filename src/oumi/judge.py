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

from typing import Any

from oumi.core.configs import JudgeConfig
from oumi.core.datasets import BaseSftDataset
from oumi.core.types.conversation import Conversation
from oumi.judges.oumi_judge import OumiXmlJudge as Judge


def judge_dataset(config: JudgeConfig, dataset: BaseSftDataset) -> list[dict[str, Any]]:
    """Judge a dataset.

    This function evaluates a given dataset using a specified Judge configuration.

    The function performs the following steps:

        1. Initializes the Judge with the provided configuration.
        2. Iterates through the dataset to extract conversation inputs.
        3. Uses the Judge to evaluate each conversation input.
        4. Collects and returns the judged outputs.

    Args:
        config: The configuration for the judge.
        dataset: The dataset to be judged. This dataset
            should be compatible with the Supervised Finetuning Dataset class.

    Returns:
        List[Dict[str, Any]]: A list of judgement results for each conversation.

        >>> # Example output:
        [
            {'helpful': True, 'safe': False},
            {'helpful': True, 'safe': True},
        ]

    Example:
        >>> config = JudgeConfig(...)
        >>> dataset = SomeDataset(...)
        >>> judged_outputs = judge_dataset(config, dataset)
        >>> for output in judged_outputs:
        ...     print(output)
    """
    judge = Judge(config)
    judge_inputs = [dataset.conversation(idx) for idx in range(len(dataset))]
    judge_outputs = judge.judge(judge_inputs)
    return judge_outputs


def judge_conversations(
    config: JudgeConfig, judge_inputs: list[Conversation]
) -> list[dict[str, Any]]:
    """Judge a list of conversations.

    This function evaluates a list of conversations using the specified Judge.

    The function performs the following steps:

        1. Initializes the Judge with the provided configuration.
        2. Uses the Judge to evaluate each conversation input.
        3. Collects and returns the judged outputs.

    Args:
        config: The configuration for the judge.
        judge_inputs: A list of Conversation objects to be judged.

    Returns:
        List[Dict[str, Any]]: A list of judgement results for each conversation.

        >>> # Example output:
        [
            {'helpful': True, 'safe': False},
            {'helpful': True, 'safe': True},
        ]

    Example:
        >>> config = JudgeConfig(...)
        >>> judge_inputs = [Conversation(...), Conversation(...)]
        >>> judged_outputs = judge_conversations(config, judge_inputs)
        >>> for output in judged_outputs:
        ...     print(output)
    """
    judge = Judge(config)
    judge_outputs = judge.judge(judge_inputs)
    return judge_outputs
