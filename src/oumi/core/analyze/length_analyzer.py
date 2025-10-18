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
from typing import Optional, Union

import pandas as pd
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from oumi.core.analyze.column_types import ContentType
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
        tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None,
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
    ) -> pd.DataFrame:
        """Analyze text fields and return metrics.

        Args:
            df: Input DataFrame with text fields
            schema: Column schema dict to identify text fields

        Returns:
            DataFrame with added field-level analysis columns
        """
        result_df = df.copy()

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
            raise ValueError(
                "No text fields found in the DataFrame for length analysis. "
                "Please ensure your schema specifies columns with"
                "content_type='text'and that those columns exist in the DataFrame."
            )

        # Analyze each text field and add field-level metrics
        for column in text_columns:
            if self.char_count:
                result_df[f"{column}_char_count"] = df[column].astype(str).str.len()

            if self.word_count:
                result_df[f"{column}_word_count"] = (
                    df[column].astype(str).str.split().str.len()
                )

            if self.sentence_count:
                result_df[f"{column}_sentence_count"] = (
                    df[column]
                    .astype(str)
                    .apply(
                        lambda text: len(
                            [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
                        )
                    )
                )

            if self.token_count and self.tokenizer is not None:
                tokenizer = self.tokenizer  # Type assertion for pyright
                result_df[f"{column}_token_count"] = (
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

        return result_df
