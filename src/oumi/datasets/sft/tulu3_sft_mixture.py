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


from typing import Union

import pandas as pd

from oumi.core.datasets import BaseSftDataset
from oumi.core.registry import register_dataset
from oumi.core.types.conversation import Conversation, Message, Role


@register_dataset("allenai/tulu-3-sft-mixture")
class Tulu3MixtureDataset(BaseSftDataset):
    default_dataset = "allenai/tulu-3-sft-mixture"

    def transform_conversation(self, example: Union[dict, pd.Series]) -> Conversation:
        """Convert the example into a Conversation.

        Args:
            example (dict or Pandas Series): An example containing a `messages`
               field which is a list of dicts with `content` and `role` string
               fields
        """
        messages = [self._transform_message(msg) for msg in example["messages"]]
        return Conversation(messages=messages)

    @classmethod
    def _transform_message(cls, message: dict) -> Message:
        roles = {"system": Role.SYSTEM, "assistant": Role.ASSISTANT, "user": Role.USER}
        content = message["content"]
        role_str = message["role"].lower().strip()
        try:
            role = roles[role_str]
            return Message(role=role, content=content)
        except KeyError as e:
            raise ValueError(
                f"Unknown role {message['role']}, was expecting one of {roles.keys()}"
            ) from e
