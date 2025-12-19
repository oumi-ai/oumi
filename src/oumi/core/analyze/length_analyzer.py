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

"""Length analyzer for text content."""

import re

import pandas as pd
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from oumi.core.analyze.column_types import ColumnType, ContentType
from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.registry import register_sample_analyzer


@register_sample_analyzer("length")
class LengthAnalyzer(SampleAnalyzer):
    """Analyzer that computes various length metrics for text content."""

    def __init__(
        self,
        *,
        char_count: bool = True,
        word_count: bool = True,
        sentence_count: bool = True,
        token_count: bool = False,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        include_special_tokens: bool = True,
    ):
        """Initialize the LengthAnalyzer.

        Args:
            char_count: Whether to compute character count
            word_count: Whether to compute word count
            sentence_count: Whether to compute sentence count
            token_count: Whether to compute token count
            tokenizer: Tokenizer to use for token counting
                (required if token_count=True)
            include_special_tokens: Whether to include special tokens in token count.
                Defaults to True to match training tokenization. Set to False for raw
                content analysis only.
        """
        self.char_count = char_count
        self.word_count = word_count
        self.sentence_count = sentence_count
        self.token_count = token_count
        self.tokenizer = tokenizer
        self.include_special_tokens = include_special_tokens
        # Validate tokenizer requirements
        if self.token_count and tokenizer is None:
            raise ValueError(
                "tokenizer must be provided when token_count=True. "
                "Set token_count=False or provide a tokenizer."
            )

    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> tuple[pd.DataFrame, dict]:
        """Analyze text fields and return metrics.

        Args:
            df: Input DataFrame with text fields
            schema: Column schema dict to identify text fields

        Returns:
            Tuple of (DataFrame with added field-level analysis columns,
            generated column schema dict)
        """
        result_df = df.copy()
        generated_schema = {}

        if not schema:
            raise ValueError(
                "schema is required to identify text fields for length analysis. "
                "Please provide a column schema dict that specifies which "
                "columns contain text content."
            )

        text_columns = [
            col
            for col, config in schema.items()
            if config.get("content_type") == ContentType.TEXT and col in df.columns
        ]

        if not text_columns:
            # No text fields in this DataFrame - this is expected for some DataFrames
            # (e.g., conversation-level DataFrames). Return unchanged.
            return result_df, generated_schema

        # Get analyzer ID for column naming (defaults to "length")
        analyzer_id = getattr(self, "analyzer_id", "length")

        for column in text_columns:
            if self.char_count:
                col_name = f"{column}_{analyzer_id}_char_count"
                result_df[col_name] = df[column].astype(str).str.len()
                generated_schema[col_name] = {
                    "type": ColumnType.INT,
                    "content_type": ContentType.NUMERIC,
                    "description": f"Character count for {column}",
                }

            if self.word_count:
                col_name = f"{column}_{analyzer_id}_word_count"
                result_df[col_name] = df[column].astype(str).str.split().str.len()
                generated_schema[col_name] = {
                    "type": ColumnType.INT,
                    "content_type": ContentType.NUMERIC,
                    "description": f"Word count for {column}",
                }

            if self.sentence_count:
                col_name = f"{column}_{analyzer_id}_sentence_count"
                result_df[col_name] = (
                    df[column]
                    .astype(str)
                    .apply(
                        lambda text: len(
                            [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
                        )
                    )
                )
                generated_schema[col_name] = {
                    "type": ColumnType.INT,
                    "content_type": ContentType.NUMERIC,
                    "description": f"Sentence count for {column}",
                }

            if self.token_count and self.tokenizer is not None:
                tokenizer = self.tokenizer  # Type assertion for pyright
                col_name = f"{column}_{analyzer_id}_token_count"
                result_df[col_name] = (
                    df[column]
                    .astype(str)
                    .apply(
                        lambda text: len(
                            tokenizer.encode(
                                text, add_special_tokens=self.include_special_tokens
                            )
                        )
                    )
                )
                generated_schema[col_name] = {
                    "type": ColumnType.INT,
                    "content_type": ContentType.NUMERIC,
                    "description": f"Token count for {column}",
                }

        return result_df, generated_schema
