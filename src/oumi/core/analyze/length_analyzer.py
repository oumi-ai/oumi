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
from typing import Any, Optional, Union

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from oumi.core.analyze.dataset_analyzer import (
    FieldAnalysisResult,
    SampleAnalysisResult,
)
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
        """Initialize the length analyzer.

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

    def analyze_fields(
        self,
        text_fields: list[tuple[str, str]],
        tokenizer: Optional[Any] = None
    ) -> list[FieldAnalysisResult]:
        """Analyze individual text fields.
        
        Args:
            text_fields: List of (field_name, text_content) tuples
            tokenizer: Optional tokenizer to use for analysis
            
        Returns:
            List of FieldAnalysisResult objects, one for each field
        """
        field_results = []
        for field_idx, (field_name, text_content) in enumerate(text_fields):
            metrics = self.compute_length_metrics(text_content)
            
            field_result = FieldAnalysisResult(
                field_name=field_name,
                field_index=field_idx,
                text_content=text_content,
                analyzer_metrics=metrics
            )
            field_results.append(field_result)
        
        return field_results

    def analyze_sample(
        self, 
        sample: dict, 
        text_fields: list[str],
        tokenizer: Optional[Any] = None
    ) -> SampleAnalysisResult:
        """Analyze the entire sample.
        
        Args:
            sample: The sample dictionary to analyze
            text_fields: List of field names that contain text content to analyze
            tokenizer: Optional tokenizer to use for analysis (ignored - uses instance tokenizer)
            
        Returns:
            SampleAnalysisResult for the entire sample
        """
        sample_id = sample.get("id", "unknown")
        sample_metrics = {}
        
        # For sample-level metrics, we need to handle different cases:
        # 1. If there's a rendered sample, use it for token counting (more accurate)
        # 2. For other metrics, aggregate all text fields
        
        # Define which metrics should use rendered sample
        RENDERED_SAMPLE_METRICS = {"token_count"}
        
        if "rendered_sample" in sample and sample["rendered_sample"]:
            # Use rendered sample for token counting (more accurate)
            rendered_text = sample["rendered_sample"]
            rendered_metrics = self.compute_length_metrics(rendered_text)
            
            # Only use token_count from rendered sample
            if self.token_count and "token_count" in rendered_metrics:
                sample_metrics["token_count"] = rendered_metrics["token_count"]
        
        # For all other metrics, aggregate only the text fields that are being analyzed
        all_text_content = []
        for field_name in text_fields:
            if field_name in sample and isinstance(sample[field_name], str) and sample[field_name].strip():
                all_text_content.append(sample[field_name])
        
        if all_text_content:
            # Compute metrics for the combined text content
            combined_text = " ".join(all_text_content)
            combined_metrics = self.compute_length_metrics(combined_text)
            
            # Add metrics without prefix since this is sample-level analysis
            for metric_name, metric_value in combined_metrics.items():
                # Skip token_count if we already got it from rendered conversation
                if metric_name == "token_count" and "token_count" in sample_metrics:
                    continue
                sample_metrics[metric_name] = metric_value
        
        return SampleAnalysisResult(
            sample_id=sample_id,
            analyzer_metrics=sample_metrics
        )



    def compute_length_metrics(self, text_content: str) -> dict[str, Any]:
        """Compute length metrics for a single text content.

        This is a helper function that can be used by both message-level and
        conversation-level analysis.

        Args:
            text_content: The text content to analyze
            tokenizer: Optional tokenizer to use for token counting

        Returns:
            Dictionary containing requested length metrics
        """
        metrics = {}

        if self.char_count:
            metrics["char_count"] = len(text_content)

        if self.word_count:
            # Simple word count - split on whitespace
            metrics["word_count"] = len(text_content.split())

        if self.sentence_count:
            # Simple sentence count - split on common sentence endings
            sentences = re.split(r"[.!?]+", text_content)
            # Filter out empty strings
            sentences = [s.strip() for s in sentences if s.strip()]
            metrics["sentence_count"] = len(sentences)

        if self.token_count:
            # Use instance tokenizer only
            tokenizer_to_use = self.tokenizer
            if tokenizer_to_use is not None:
                # Use tokenizer for accurate token count
                tokens = tokenizer_to_use.encode(
                    text_content, add_special_tokens=self.include_special_tokens
                )
                metrics["token_count"] = len(tokens)

        return metrics


