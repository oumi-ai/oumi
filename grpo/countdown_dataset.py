import pandas as pd
from typing_extensions import override

from oumi.core.datasets.base_grpo_dataset import BaseExperimentalGrpoDataset
from oumi.core.registry import register_dataset
from oumi.core.types.conversation import Conversation


@register_dataset("countdown_data")
class CountdownDataset(BaseExperimentalGrpoDataset):
    """Dataset class for the countdown dataset.

    A sample from the dataset:
    {
        "messages": [
            {
                "content": "You are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer.",
                "role": "system"
            },
            {
                "content": "Using the numbers [73, 30, 38, 97], create an equation that equals 92. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 = 1 </answer>.",
                "role": "user"
            },
            {
                "content": "Let's think step by step.\n\nWe are given the numbers: [73, 30, 38, 97]  \nWe need to use each number exactly once with basic arithmetic operations (+, \u2212, \u00d7, \u00f7) to get a result of 92.\n\nOur goal: form an equation using all four numbers once, such that the result is 92.\n\nStep 1: Try to find combinations that get close to 92.\n\nLet\u2019s look at differences between numbers:\n\n- 97 \u2212 73 = 24  \n- 97 \u2212 38 = 59  \n- 97 \u2212 30 = 67  \n- 73 \u2212 38 = 35  \n- 73 \u2212 30 = 43  \n- 38 \u2212 30 = 8  \n\nNot immediately helpful.\n\nWhat if we try to build 92 via addition or subtraction?\n\nTry:  \n97 \u2212 30 = 67 \u2192 67 + 38 = 105 \u2192 too big  \n97 \u2212 38 = 59 \u2192 59 + 30 = 89 \u2192 89 + 73 = 162 \u2192 too big  \n73 + 30 = 103 \u2192 103 \u2212 38 = 65 \u2192 65 + 97 = 162\n\nTry: 97 \u2212 (73 \u2212 38) = 97 \u2212 35 = 62 \u2192 62 + 30 = 92! \u2705\n\nWait \u2014 let's verify:\n\n73 \u2212 38 = 35  \n97 \u2212 35 = 62  \n62 + 30 = 92\n\nYes! That works.\n\nSo the full expression is:\n\n97 \u2212 (73 \u2212 38) + 30 = 92\n\nNow, check if we used all numbers:  \n73 \u2705, 38 \u2705, 97 \u2705, 30 \u2705  \nEach used once.  \nOnly basic operations: subtraction and addition.\n\nSo this is valid.\n\nNow format as requested:\n\n<answer> 97 - (73 - 38) + 30 = 92 </answer>",
                "role": "assistant"
            }
        ],
        "metadata": {
            "target": 92,
            "nums": [
                73,
                30,
                38,
                97
            ]
        }
    }
    """  # noqa: E501

    default_dataset = "countdown_dataset"

    @override
    def transform(self, sample: pd.Series) -> dict:
        """Validate and transform the sample into Python `dict`."""
        sample_dict = sample.to_dict()
        output_dict = {
            "prompt": sample_dict["messages"],
            "nums": sample_dict["metadata"]["nums"],
            "target": sample_dict["metadata"]["target"],
        }
        return output_dict

    @override
    def transform_conversation(self, sample: pd.Series) -> Conversation:
        """Converts the input sample to a Conversation.

        Args:
            sample (dict): The input example.

        Returns:
            Conversation: The resulting conversation.

        """
        # Example is already in conversation format and only needs light processing.
        sample_dict = sample.to_dict()
        return Conversation.from_dict(sample_dict)
