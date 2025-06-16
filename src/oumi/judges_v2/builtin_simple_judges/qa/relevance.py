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

"""Built-in judge configuration for evaluating answer relevance in Q&A scenarios."""

from oumi.core.configs.judge_config_v2 import (
    JudgeConfig,
    JudgeOutputType,
    JudgeResponseFormat,
)

RELEVANCE_JUDGE_CONFIG = JudgeConfig(
    system_instruction="""
You are an expert evaluator tasked with assessing the relevance of an answer to a given
question.

Specifically, you need to assess whether the answer:
- Responds to the specific question being asked
- Stays on topic and doesn't drift to unrelated subjects
- Provides information that is pertinent to what was requested

Note: An answer can be relevant even if it's incomplete, incorrect, or admits
uncertainty.
""",
    prompt_template="""
Here is the data:
[BEGIN DATA]
***
[Question]:
{question}
***
[Answer]:
{answer}
***
[END DATA]
""",
    response_format=JudgeResponseFormat.JSON,
    judgment_type=JudgeOutputType.BOOL,
    include_explanation=True,
)
