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
    "OumiXmlJudge",
    "OumiJudgeInput",
    "OumiJudgeOutput",
    "BaseJudge",
    "BaseJudgeOutput",
]
