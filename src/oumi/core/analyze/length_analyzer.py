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
from typing import TYPE_CHECKING, Optional, Union

import pandas as pd

from oumi.core.analyze.column_types import ContentType
from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.registry import register_sample_analyzer

if TYPE_CHECKING:
    import tiktoken
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

# Default tiktoken encoding for GPT-5 style tokenization
DEFAULT_TIKTOKEN_ENCODING = "o200k_base"


@register_sample_analyzer("length")
class LengthAnalyzer(SampleAnalyzer):
    """Analyzer that computes various length metrics for text content."""

    def __init__(
        self,
        *,
        char_count: bool = False,
        word_count: bool = False,
        sentence_count: bool = False,
        token_count: bool = True,
        tokenizer: Optional[
            Union["PreTrainedTokenizer", "PreTrainedTokenizerFast"]
        ] = None,
        tiktoken_encoding: Optional[str] = DEFAULT_TIKTOKEN_ENCODING,
        include_special_tokens: bool = True,
    ):
        """Initialize the LengthAnalyzer.

        Args:
            char_count: Whether to compute character count. Defaults to False.
            word_count: Whether to compute word count. Defaults to False.
            sentence_count: Whether to compute sentence count. Defaults to False.
            token_count: Whether to compute token count. Defaults to True.
            tokenizer: HuggingFace tokenizer to use for token counting.
                If provided, this takes precedence over tiktoken_encoding.
            tiktoken_encoding: tiktoken encoding name for token counting.
                Defaults to "o200k_base" (GPT-4o/GPT-5 encoding).
                Set to None to disable tiktoken. Common encodings:
                - "o200k_base": GPT-4o, GPT-5 (200k vocab, latest)
                - "cl100k_base": GPT-4, GPT-3.5-turbo (100k vocab)
            include_special_tokens: Whether to include special tokens in token count.
                Defaults to True to match training tokenization. Set to False for raw
                content analysis only. Only applies to HuggingFace tokenizers.
        """
        self.char_count = char_count
        self.word_count = word_count
        self.sentence_count = sentence_count
        self.token_count = token_count
        self.tokenizer = tokenizer
        self.tiktoken_encoding = tiktoken_encoding
        self.include_special_tokens = include_special_tokens
        self._tiktoken_encoder: Optional["tiktoken.Encoding"] = None

        # Validate tokenizer requirements
        if self.token_count and tokenizer is None and tiktoken_encoding is None:
            raise ValueError(
                "Either tokenizer or tiktoken_encoding must be provided when "
                "token_count=True. Set token_count=False or provide a tokenizer/"
                "tiktoken_encoding."
            )

        # Initialize tiktoken encoder if needed
        if self.token_count and tokenizer is None and tiktoken_encoding is not None:
            self._init_tiktoken_encoder()

    def _init_tiktoken_encoder(self) -> None:
        """Initialize the tiktoken encoder."""
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "tiktoken is required for token counting with tiktoken_encoding. "
                "Install it with: pip install tiktoken"
            )

        if self.tiktoken_encoding is None:
            raise ValueError("tiktoken_encoding must be set to initialize tiktoken")
        self._tiktoken_encoder = tiktoken.get_encoding(self.tiktoken_encoding)

    def _count_tokens_tiktoken(self, text: str) -> int:
        """Count tokens using tiktoken encoder."""
        if self._tiktoken_encoder is None:
            raise ValueError("tiktoken encoder not initialized")
        return len(self._tiktoken_encoder.encode(text))

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
            # No text columns to analyze in this DataFrame, return unchanged
            return result_df

        # Get analyzer ID for column naming (defaults to "length")
        analyzer_id = getattr(self, "analyzer_id", "length")
        for column in text_columns:
            if self.char_count:
                col_name = f"{column}_{analyzer_id}_char_count"
                result_df[col_name] = df[column].astype(str).str.len()

            if self.word_count:
                col_name = f"{column}_{analyzer_id}_word_count"
                result_df[col_name] = df[column].astype(str).str.split().str.len()

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

            if self.token_count:
                col_name = f"{column}_{analyzer_id}_token_count"
                if self.tokenizer is not None:
                    # Use HuggingFace tokenizer (takes precedence)
                    tokenizer = self.tokenizer  # Type assertion for pyright
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
                elif self._tiktoken_encoder is not None:
                    # Use tiktoken encoder
                    result_df[col_name] = (
                        df[column]
                        .astype(str)
                        .apply(self._count_tokens_tiktoken)
                    )

        return result_df
