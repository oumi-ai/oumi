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

"""This module provides access to various judge configurations for the Oumi project.

The judges are used to evaluate the quality of AI-generated responses based on
different criteria such as helpfulness, honesty, and safety.
"""

from oumi.judges.base_judge import (
    BaseJudge,
    BaseJudgeOutput,
)
from oumi.judges.judge_court import (
    oumi_v1_xml_claude_sonnet_judge,
    oumi_v1_xml_deepseek_r1_judge_hosted_by_deepseek,
    oumi_v1_xml_deepseek_r1_judge_hosted_by_sambanova,
    oumi_v1_xml_deepseek_r1_judge_hosted_by_together,
    oumi_v1_xml_gpt4o_judge,
    oumi_v1_xml_local_judge,
)
from oumi.judges.oumi_judge import (
    OumiJudgeInput,
    OumiJudgeOutput,
    OumiXmlJudge,
)

__all__ = [
    "oumi_v1_xml_claude_sonnet_judge",
    "oumi_v1_xml_gpt4o_judge",
    "oumi_v1_xml_local_judge",
    "oumi_v1_xml_deepseek_r1_judge_hosted_by_deepseek",
    "oumi_v1_xml_deepseek_r1_judge_hosted_by_sambanova",
    "oumi_v1_xml_deepseek_r1_judge_hosted_by_together",
    "OumiXmlJudge",
    "OumiJudgeInput",
    "OumiJudgeOutput",
    "BaseJudge",
    "BaseJudgeOutput",
]
