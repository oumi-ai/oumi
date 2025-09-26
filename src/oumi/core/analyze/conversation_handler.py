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

from typing import Any, Optional

import pandas as pd
from tqdm import tqdm

from oumi.core.analyze.column_types import ColumnType
from oumi.core.datasets import BaseMapDataset
from oumi.utils.logging import logger


class ConversationHandler:
    """Handles conversation-specific DataFrame preparation.

    This class is responsible for:
    - Converting conversations to DataFrames
    - Rendering conversations for token counting
    - Preparing conversation datasets for analysis
    - Managing conversation-specific data transformations

    Note: This class does NOT apply analyzers - that's handled by DataFrameAnalyzer.
    """

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        schema: Optional[dict] = None,
    ):
        """Initialize the conversation handler.

        Args:
            tokenizer: Optional tokenizer for rendering conversations
            schema: Column schema for DataFrame dtypes
        """
        self.tokenizer = tokenizer
        self.schema = schema or {}

    def conversation_to_dataframes(
        self, conversation, conversation_id: str, conv_idx: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Convert a conversation to conversations and messages DataFrames.

        Args:
            conversation: The conversation object to convert
            conversation_id: The conversation ID
            conv_idx: The conversation index

        Returns:
            Tuple of (conversations_df, messages_df)
        """
        # Create messages DataFrame (one row per message)
        message_records = []
        for msg_idx, message in enumerate(conversation.messages):
            message_data = {
                "item_index": conv_idx,
                "row_index": msg_idx,
                "role": message.role.value,
                "content": self._extract_message_content(message),
            }

            # Add message ID if available
            if message.id:
                message_data["message_id"] = message.id

            message_records.append(message_data)

        # Create messages_df with proper dtypes
        if message_records:
            messages_df = pd.DataFrame(message_records)
        else:
            messages_df = pd.DataFrame(
                {
                    "item_index": [],
                    "row_index": [],
                    "role": [],
                    "content": [],
                    "message_id": [],
                }
            )

        # Set proper dtypes from schema
        self._set_dtypes_from_config(messages_df)

        # Create conversations DataFrame (one row per conversation)
        conversation_data = {
            "item_index": conv_idx,  # Keep column name for system compatibility
            "item_id": conversation_id,
            "item_type": "conversation",
        }

        # Add rendered sample for token counting if tokenizer is available
        if self.tokenizer is not None:
            try:
                rendered_sample = self._render_conversation_for_tokens(conversation)
                conversation_data["rendered_item"] = rendered_sample
            except Exception as e:
                logger.warning(
                    f"Failed to render conversation {conversation_id} for token "
                    f"counting: {e}"
                )

        # Add any additional metadata from the conversation
        if hasattr(conversation, "metadata") and conversation.metadata:
            conversation_data.update(conversation.metadata)

        # Create conversations_df with proper dtypes
        conversations_df = pd.DataFrame([conversation_data])

        # Set proper dtypes from schema
        self._set_dtypes_from_config(conversations_df)

        return conversations_df, messages_df

    def _extract_message_content(self, message) -> str:
        """Extract content from a message, handling both text and multimodal content.

        Args:
            message: The message object to extract content from

        Returns:
            String representation of the message content
        """
        if isinstance(message.content, str):
            # Simple text content
            return message.content
        elif isinstance(message.content, list):
            # Multimodal content - extract text parts and describe non-text parts
            content_parts = []
            for item in message.content:
                if hasattr(item, "is_text") and item.is_text():
                    content_parts.append(item.content)
                else:
                    # For non-text content, add a description
                    content_type = getattr(item, "type", "unknown")
                    type_name = (
                        content_type.upper()
                        if hasattr(content_type, "upper")
                        else content_type
                    )
                    content_parts.append(f"[{type_name}]")
            return " ".join(content_parts)
        else:
            # Fallback for unexpected content types
            return str(message.content)

    def _render_conversation_for_tokens(self, conversation) -> str:
        """Render a conversation using the tokenizer's chat template for token counting.

        Args:
            conversation: The conversation object to render

        Returns:
            Rendered conversation string

        Raises:
            ValueError: If tokenizer is not available
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for conversation rendering")

        # Use the tokenizer's chat template to render the conversation
        prompt_text = self.tokenizer.apply_chat_template(
            conversation,  # type: ignore
            tokenize=False,
            add_generation_prompt=False,
        )
        return str(prompt_text)

    def convert_dataset_to_dataframes(
        self,
        dataset: BaseMapDataset,
        items_to_analyze: int,
        dataset_name: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Convert a conversation dataset to conversations and messages DataFrames.

        This method converts all conversations to complete DataFrames that are ready
        for analysis. The DataFrameAnalyzer will then work with these complete
        DataFrames.

        Args:
            dataset: The conversation dataset to process
            items_to_analyze: Number of items to analyze
            dataset_name: Name of the dataset for progress display

        Returns:
            Tuple of (conversations_df, messages_df) ready for analysis

        Raises:
            ValueError: If dataset is not provided
        """
        if dataset is None:
            raise ValueError("Dataset must be provided for conversation processing")

        conversation_df_list = []
        message_df_list = []

        for conversation_idx in tqdm(
            range(items_to_analyze),
            desc=f"Converting {dataset_name} to DataFrames",
            unit="item",
        ):
            conversation = dataset.conversation(conversation_idx)
            conversation_id = conversation.conversation_id or str(conversation_idx)
            conversation_df, message_df = self.conversation_to_dataframes(
                conversation, conversation_id, conversation_idx
            )

            # Collect all DataFrames for concatenation
            if not conversation_df.empty:
                conversation_df_list.append(conversation_df)
            if not message_df.empty:
                message_df_list.append(message_df)

        # Create complete DataFrames by concatenating all individual DataFrames
        conversations_df = (
            pd.concat(conversation_df_list, ignore_index=True)
            if conversation_df_list
            else pd.DataFrame()
        )
        messages_df = (
            pd.concat(message_df_list, ignore_index=True)
            if message_df_list
            else pd.DataFrame()
        )

        return conversations_df, messages_df

    def _set_dtypes_from_config(self, df: pd.DataFrame) -> None:
        """Set DataFrame column dtypes based on schema.

        Args:
            df: DataFrame to set dtypes for
        """
        if not self.schema:
            return

        # Simple mapping from type values to pandas operations
        dtype_mapping = {
            ColumnType.STRING: lambda col: col.astype("string"),
            ColumnType.INT: lambda col: col.astype("int64"),
            ColumnType.FLOAT: lambda col: col.astype("float64"),
            ColumnType.TIMESTAMP: lambda col: pd.to_datetime(col),
            ColumnType.BOOL: lambda col: col.astype("boolean"),
            ColumnType.CATEGORICAL: lambda col: col.astype("category"),
            ColumnType.OBJECT: lambda col: col,  # Keep as-is
        }

        for col_name, col_config in self.schema.items():
            if col_name in df.columns and "type" in col_config:
                dtype = col_config["type"]
                try:
                    if dtype in dtype_mapping:
                        df[col_name] = dtype_mapping[dtype](df[col_name])
                except Exception as e:
                    logger.warning(
                        f"Failed to set dtype '{dtype}' for column '{col_name}': {e}"
                    )
