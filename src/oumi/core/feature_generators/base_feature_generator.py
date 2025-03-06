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

from abc import ABC, abstractmethod

from oumi.core.types.conversation import Conversation


class BaseConversationFeatureGenerator(ABC):
    """Applies `processor` to generate model inputs from an input `Conversation`."""

    @abstractmethod
    def transform_conversation(self, conversation: Conversation) -> dict:
        """Transforms a single Oumi conversation into a dictionary of model inputs.

        Args:
            conversation: An input conversation.

        Returns:
            dict: A dictionary of inputs for a model.
        """
        raise NotImplementedError

    @abstractmethod
    def transform_conversations(self, conversations: list[Conversation]) -> dict:
        """Transforms a list of Oumi conversations into a dictionary of model inputs.

        Args:
            conversations: A list of input conversations.

        Returns:
            dict: A dictionary of inputs for a model.
        """
        raise NotImplementedError
