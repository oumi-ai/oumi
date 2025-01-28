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

from typing import Union, cast

import numpy as np
import pandas as pd
from typing_extensions import override

from oumi.core.datasets import BaseSftDataset
from oumi.core.registry import register_dataset
from oumi.core.types.conversation import Conversation, Message, Role


@register_dataset("argilla/databricks-dolly-15k-curated-en")
class ArgillaDollyDataset(BaseSftDataset):
    """Dataset class for the Databricks Dolly 15k curated dataset."""

    default_dataset = "argilla/databricks-dolly-15k-curated-en"

    def __init__(self, *, use_new_fields: bool = True, **kwargs) -> None:
        """Initialize the DollyDataset.

        Args:
            use_new_fields (bool): Whether to use the new fields
            (new-instruction, new-context, new-response) instead of the original fields.
            Defaults to True.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """
        self.use_new_fields = use_new_fields
        super().__init__(**kwargs)

    @override
    def transform_conversation(self, example: Union[dict, pd.Series]) -> Conversation:
        """Transform a dataset example into a Conversation object.

        Args:
            example: A single example from the dataset.

        Returns:
            Conversation: A Conversation object containing the transformed messages.
        """
        messages = []

        col_prefix = "new" if self.use_new_fields else "original"

        instruction = self._get_field_value(example, f"{col_prefix}-instruction")
        context = self._get_field_value(example, f"{col_prefix}-context")
        response = self._get_field_value(example, f"{col_prefix}-response")

        # Construct the user message
        user_content = instruction
        if context:
            user_content += f"\n\nContext: {context}"

        messages.append(Message(role=Role.USER, content=user_content))
        messages.append(Message(role=Role.ASSISTANT, content=response))

        return Conversation(messages=messages)

    @staticmethod
    def _get_field_value(example: Union[dict, pd.Series], field: str) -> str:
        """Helper method to get the value from a field.

        Args:
            example (Union[Dict, pd.Series]): A single example from the dataset.
            field (str): The field name to retrieve.

        Returns:
            str: The value of the field.
        """
        value = example[field]

        if isinstance(value, str):
            return value
        if isinstance(value, (dict, pd.Series)) and "value" in value:
            return cast(
                str,
                value["value"][0]
                if isinstance(value["value"], (list, np.ndarray))
                else value["value"],
            )
        raise RuntimeError(f"Unable to parse field: {field}")
