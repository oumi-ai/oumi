# Copyright 2024 Oumi AI, Inc.
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
from typing import Any, Optional, Union

import pandas as pd
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

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

        # Store field-level results
        self._field_df = None
        self._sample_df = None

        # Validate tokenizer requirements
        if self.token_count and tokenizer is None:
            raise ValueError(
                "tokenizer must be provided when token_count=True. "
                "Set token_count=False or provide a tokenizer."
            )

    def analyze_fields(
        self,
        df: pd.DataFrame,
        text_fields: list[str],
        tokenizer: Optional[Any] = None,
    ) -> pd.DataFrame:
        """Analyze individual text fields and add field-level metrics to DataFrame.

        Args:
            df: Input DataFrame with text fields
            text_fields: List of field names that contain text content to analyze
            tokenizer: Optional tokenizer to use for analysis (ignored - uses instance
                tokenizer)

        Returns:
            DataFrame with added field-level analysis columns
        """
        result_df = df.copy()

        # Find text fields that exist in the DataFrame
        available_text_fields = [col for col in text_fields if col in df.columns]

        if not available_text_fields:
            return result_df

        # Analyze each text field
        for field_name in available_text_fields:
            if field_name in df.columns:
                # Get non-null values for this field
                non_null_values = df[field_name].dropna()
                if not non_null_values.empty:
                    # Use the first non-null value for analysis
                    text_content = str(non_null_values.iloc[0])

                    # Compute metrics for this field
                    if self.char_count:
                        result_df[f"{field_name}_char_count"] = len(text_content)

                    if self.word_count:
                        result_df[f"{field_name}_word_count"] = len(
                            text_content.split()
                        )

                    if self.sentence_count:
                        sentences = re.split(r"[.!?]+", text_content)
                        result_df[f"{field_name}_sentence_count"] = len(
                            [s.strip() for s in sentences if s.strip()]
                        )

                    if self.token_count and self.tokenizer is not None:
                        tokens = self.tokenizer.encode(
                            text_content, add_special_tokens=self.include_special_tokens
                        )
                        result_df[f"{field_name}_token_count"] = len(tokens)

        return result_df

    def analyze_sample(
        self,
        df: pd.DataFrame,
        text_fields: list[str],
        tokenizer: Optional[Any] = None,
    ) -> pd.DataFrame:
        """Analyze samples as a whole and add sample-level metrics to DataFrame.

        This method performs both field-level and sample-level analysis internally,
        storing results in class attributes for later access.

        Args:
            df: Input DataFrame with text fields
            text_fields: List of field names that contain text content to analyze
            tokenizer: Optional tokenizer to use for analysis (ignored - uses instance
                tokenizer)

        Returns:
            DataFrame with added sample-level analysis columns
        """
        # First, analyze fields to get field-level metrics
        field_result_df = self.analyze_fields(df, text_fields, tokenizer)

        # Store field-level results
        self._field_df = field_result_df.copy()

        # Sum field-level metrics to get sample-level metrics
        result_df = df.copy()

        # Find all field-level metric columns and sum them
        available_text_fields = [col for col in text_fields if col in df.columns]

        if self.char_count:
            char_columns = [
                f"{field}_char_count" for field in available_text_fields
                if f"{field}_char_count" in field_result_df.columns
            ]
            if char_columns:
                result_df["sample_length_char_count"] = field_result_df[
                    char_columns
                ].sum(axis=1)

        if self.word_count:
            word_columns = [
                f"{field}_word_count" for field in available_text_fields
                if f"{field}_word_count" in field_result_df.columns
            ]
            if word_columns:
                result_df["sample_length_word_count"] = field_result_df[
                    word_columns
                ].sum(axis=1)

        if self.sentence_count:
            sentence_columns = [
                f"{field}_sentence_count" for field in available_text_fields
                if f"{field}_sentence_count" in field_result_df.columns
            ]
            if sentence_columns:
                result_df["sample_length_sentence_count"] = field_result_df[
                    sentence_columns
                ].sum(axis=1)

        if self.token_count:
            token_columns = [
                f"{field}_token_count" for field in available_text_fields
                if f"{field}_token_count" in field_result_df.columns
            ]
            if token_columns:
                result_df["sample_length_token_count"] = field_result_df[
                    token_columns
                ].sum(axis=1)

        # For token counting, prefer rendered sample if available (more accurate)
        if (
            "rendered_sample" in df.columns
            and self.token_count
            and self.tokenizer is not None
        ):
            result_df["sample_length_token_count"] = df["rendered_sample"].apply(
                lambda text: len(
                    self.tokenizer.encode(  # type: ignore
                        text, add_special_tokens=self.include_special_tokens
                    )
                )
            )

        # Store sample-level results
        self._sample_df = result_df.copy()

        return result_df


    def get_field_results(self) -> Optional[pd.DataFrame]:
        """Get field-level analysis results.

        Returns:
            DataFrame with field-level analysis results, or None if no analysis
            has been performed
        """
        return self._field_df

    def get_sample_results(self) -> Optional[pd.DataFrame]:
        """Get sample-level analysis results.

        Returns:
            DataFrame with sample-level analysis results, or None if no analysis
            has been performed
        """
        return self._sample_df
