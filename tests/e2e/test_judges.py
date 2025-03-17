import unittest

import pytest

from oumi.core.types.conversation import Conversation, Message, Role
from oumi.judges.judge_court import (
    oumi_v1_xml_deepseek_r1_judge_hosted_by_deepseek,
    oumi_v1_xml_deepseek_r1_judge_hosted_by_sambanova,
    oumi_v1_xml_deepseek_r1_judge_hosted_by_together,
)
from oumi.judges.oumi_judge import OumiXmlJudge as OumiJudge


class TestDeepSeekR1Judge(unittest.TestCase):
    @pytest.mark.skip("Skipping this test due to unable to add funds")
    def test_deepseek_r1_judge_config_hosted_by_deepseek(self):
        conversations = [
            Conversation(
                messages=[
                    Message(
                        role=Role.USER, content="What is the sum of 1 and 1 in binary?"
                    ),
                    Message(role=Role.ASSISTANT, content="The sum is 11 in binary."),
                ]
            ),
            Conversation(
                messages=[
                    Message(role=Role.USER, content="What's the capital of France?"),
                    Message(role=Role.ASSISTANT, content="French people love Paris!"),
                ]
            ),
        ]

        # Get the judge configuration
        config = oumi_v1_xml_deepseek_r1_judge_hosted_by_deepseek()
        judge = OumiJudge(config)
        judge_output = judge.judge(conversations)

        # Print the results
        print(judge_output)

    @pytest.mark.skip("Skipping this test due to unable to R1 API on waiting list")
    def test_deepseek_r1_judge_config_hosted_by_sambanova(self):
        conversations = [
            Conversation(
                messages=[
                    Message(
                        role=Role.USER, content="What is the sum of 1 and 1 in binary?"
                    ),
                    Message(role=Role.ASSISTANT, content="The sum is 11 in binary."),
                ]
            ),
            Conversation(
                messages=[
                    Message(role=Role.USER, content="What's the capital of France?"),
                    Message(role=Role.ASSISTANT, content="French people love Paris!"),
                ]
            ),
        ]

        # Get the judge configuration
        config = oumi_v1_xml_deepseek_r1_judge_hosted_by_sambanova()
        judge = OumiJudge(config)
        judge_output = judge.judge(conversations)

        # Print the results
        print(judge_output)

    def test_deepseek_r1_judge_config_hosted_by_together(self):
        conversations = [
            Conversation(
                messages=[
                    Message(
                        role=Role.USER, content="What is the sum of 1 and 1 in binary?"
                    ),
                    Message(role=Role.ASSISTANT, content="The sum is 11 in binary."),
                ]
            ),
            Conversation(
                messages=[
                    Message(role=Role.USER, content="What's the capital of France?"),
                    Message(role=Role.ASSISTANT, content="French people love Paris!"),
                ]
            ),
        ]

        # Get the judge configuration
        config = oumi_v1_xml_deepseek_r1_judge_hosted_by_together()
        judge = OumiJudge(config)
        judge_output = judge.judge(conversations)

        # Print the results
        print(judge_output)


if __name__ == "__main__":
    unittest.main()
