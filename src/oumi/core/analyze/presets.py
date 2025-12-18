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

"""Analyzer presets for common analysis configurations.

Presets provide pre-configured analyzer combinations optimized for specific
use cases, making it easier to get started with dataset analysis.
"""

from typing import Any

from oumi.core.configs.analyze_config import SampleAnalyzerParams


def get_preset(preset_name: str) -> list[SampleAnalyzerParams]:
    """Get analyzer configuration for a named preset.

    Available presets:
        - sft_quality: Comprehensive analysis for SFT (instruction tuning) datasets.
            Includes length, diversity, format, and quality analyzers.

    Args:
        preset_name: Name of the preset to load.

    Returns:
        List of SampleAnalyzerParams for the preset.

    Raises:
        ValueError: If preset name is not recognized.

    Example:
        >>> from oumi.core.analyze.presets import get_preset
        >>> analyzers = get_preset("sft_quality")
        >>> config = AnalyzeConfig(
        ...     dataset_name="my_dataset",
        ...     analyzers=analyzers,
        ... )
    """
    presets = {
        "sft_quality": _get_sft_quality_preset(),
        "sft_comprehensive": _get_sft_comprehensive_preset(),
        "sft_fast": _get_sft_fast_preset(),
    }

    if preset_name not in presets:
        available = ", ".join(sorted(presets.keys()))
        raise ValueError(
            f"Unknown preset '{preset_name}'. Available presets: {available}"
        )

    return presets[preset_name]


def list_presets() -> dict[str, str]:
    """List all available presets with descriptions.

    Returns:
        Dictionary mapping preset names to descriptions.
    """
    return {
        "sft_quality": (
            "Comprehensive analysis for SFT (instruction tuning) datasets. "
            "Includes length, diversity, format, and quality analyzers optimized "
            "for conversation-style training data."
        ),
        "sft_comprehensive": (
            "Full SFT analysis with all analyzers including training quality, "
            "cost optimization, and data hygiene metrics. Best for thorough "
            "dataset evaluation before training."
        ),
        "sft_fast": (
            "Fast heuristic-only analysis for SFT datasets. Excludes embedding "
            "and LLM-based analyzers for quick iteration. Good for initial "
            "data exploration."
        ),
    }


def _get_sft_quality_preset() -> list[SampleAnalyzerParams]:
    """Get the SFT quality preset configuration.

    This preset is optimized for analyzing supervised fine-tuning datasets
    in conversation format. It includes:

    1. Length Analyzer: Token counts
    2. Diversity Analyzer: Vocabulary richness metrics
    3. Format Analyzer: Markdown, code, JSON detection
    4. Quality Analyzer: PII, encoding issues, special tokens, repetition

    Returns:
        List of analyzer configurations.
    """
    return [
        # Length analysis for understanding content size distribution
        SampleAnalyzerParams(
            id="length",
            params={
                "token_count": False,  # Requires tokenizer, disabled by default
            },
        ),
        # Diversity analysis for vocabulary richness
        SampleAnalyzerParams(
            id="diversity",
            params={
                "unique_words_ratio": True,
                "type_token_ratio": True,
                "vocabulary_richness": True,
                "hapax_legomena_ratio": False,  # Advanced metric
                "case_sensitive": False,  # Case-insensitive by default
            },
        ),
        # Format analysis for content structure
        SampleAnalyzerParams(
            id="format",
            params={
                "detect_markdown": True,
                "detect_json": True,
                "detect_code_blocks": True,
                "detect_urls": True,
                "detect_emails": False,  # Less common, disabled
                "compute_complexity": True,
            },
        ),
        # Quality analysis for safety and data issues
        SampleAnalyzerParams(
            id="quality",
            params={
                "detect_pii": True,
                "detect_emails": True,
                "detect_phones": True,
                "detect_ssn": True,
                "detect_credit_cards": True,
                "detect_ip_addresses": False,
                "detect_api_keys": True,
                "detect_language": False,  # Requires langdetect
                "detect_encoding_issues": True,
                "detect_special_tokens": True,
                "detect_repetition": True,
                "repetition_ngram_size": 3,
                "repetition_threshold": 0.3,
                "compute_quality_score": True,
            },
        ),
    ]


def get_preset_with_tokenizer(
    preset_name: str, tokenizer_name: str
) -> tuple[list[SampleAnalyzerParams], dict[str, Any]]:
    """Get a preset with tokenizer configuration enabled.

    This is a convenience function that returns both analyzer params and
    the tokenizer configuration needed for token counting.

    Args:
        preset_name: Name of the preset to load.
        tokenizer_name: HuggingFace model/tokenizer name for token counting.

    Returns:
        Tuple of (analyzer_params, tokenizer_config).

    Example:
        >>> analyzers, tok_config = get_preset_with_tokenizer(
        ...     "sft_quality", "meta-llama/Llama-3.1-8B"
        ... )
        >>> config = AnalyzeConfig(
        ...     dataset_name="my_dataset",
        ...     analyzers=analyzers,
        ...     tokenizer_name=tok_config["tokenizer_name"],
        ... )
    """
    analyzers = get_preset(preset_name)

    # Enable token counting in length analyzer
    for analyzer in analyzers:
        if analyzer.id == "length":
            analyzer.params["token_count"] = True

    tokenizer_config = {
        "tokenizer_name": tokenizer_name,
        "tokenizer_kwargs": {},
        "trust_remote_code": False,
    }

    return analyzers, tokenizer_config


def get_preset_with_language_detection(
    preset_name: str,
) -> list[SampleAnalyzerParams]:
    """Get a preset with language detection enabled.

    Note: Requires the langdetect package to be installed.

    Args:
        preset_name: Name of the preset to load.

    Returns:
        List of analyzer configurations with language detection enabled.
    """
    analyzers = get_preset(preset_name)

    # Enable language detection in quality analyzer
    for analyzer in analyzers:
        if analyzer.id == "quality":
            analyzer.params["detect_language"] = True

    return analyzers


def _get_sft_comprehensive_preset() -> list[SampleAnalyzerParams]:
    """Get the comprehensive SFT preset configuration.

    This preset includes all analyzers for thorough dataset evaluation:

    1. Length Analyzer: Token counts (using tiktoken)
    2. Diversity Analyzer: Vocabulary richness metrics
    3. Format Analyzer: Markdown, code, JSON detection
    4. Quality Analyzer: PII, encoding issues, special tokens, repetition
    5. Training Quality Analyzer: Instruction clarity, response completeness
    6. Cost Analyzer: Context window utilization

    Returns:
        List of analyzer configurations.
    """
    return [
        # Length analysis with token counting (tiktoken by default)
        SampleAnalyzerParams(
            id="length",
            params={
                "token_count": True,  # Enabled for cost analysis
            },
        ),
        # Diversity analysis for vocabulary richness
        SampleAnalyzerParams(
            id="diversity",
            params={
                "unique_words_ratio": True,
                "type_token_ratio": True,
                "vocabulary_richness": True,
                "hapax_legomena_ratio": False,
                "case_sensitive": False,
            },
        ),
        # Format analysis for content structure
        SampleAnalyzerParams(
            id="format",
            params={
                "detect_markdown": True,
                "detect_json": True,
                "detect_code_blocks": True,
                "detect_urls": True,
                "detect_emails": False,
                "compute_complexity": True,
            },
        ),
        # Quality analysis for safety and data issues
        SampleAnalyzerParams(
            id="quality",
            params={
                "detect_pii": True,
                "detect_emails": True,
                "detect_phones": True,
                "detect_ssn": True,
                "detect_credit_cards": True,
                "detect_ip_addresses": False,
                "detect_api_keys": True,
                "detect_language": False,
                "detect_encoding_issues": True,
                "detect_special_tokens": True,
                "detect_repetition": True,
                "repetition_ngram_size": 3,
                "repetition_threshold": 0.3,
                "compute_quality_score": True,
            },
        ),
        # Training quality analysis for SFT effectiveness
        SampleAnalyzerParams(
            id="training_quality",
            params={
                "compute_instruction_clarity": True,
                "compute_response_completeness": True,
                "compute_turn_quality": True,
            },
        ),
        # Cost analysis for training optimization
        SampleAnalyzerParams(
            id="cost",
            params={
                "target_context_windows": [4096, 8192, 16384],
                "compute_packing_efficiency": True,
                "packing_overhead_tokens": 10,
            },
        ),
    ]


def _get_sft_fast_preset() -> list[SampleAnalyzerParams]:
    """Get the fast SFT preset configuration.

    This preset includes only fast heuristic analyzers, excluding:
    - Embedding-based analysis
    - LLM judge analysis
    - Token counting (unless tiktoken is available)

    Optimized for quick iteration and initial data exploration.

    Returns:
        List of analyzer configurations.
    """
    return [
        # Length analysis (basic metrics only)
        SampleAnalyzerParams(
            id="length",
            params={
                "token_count": True,  # Uses tiktoken by default, fast
            },
        ),
        # Diversity analysis (fast metrics)
        SampleAnalyzerParams(
            id="diversity",
            params={
                "unique_words_ratio": True,
                "type_token_ratio": True,
                "vocabulary_richness": False,  # Disabled for speed
                "hapax_legomena_ratio": False,
                "case_sensitive": False,
            },
        ),
        # Format analysis (basic detection)
        SampleAnalyzerParams(
            id="format",
            params={
                "detect_markdown": True,
                "detect_json": True,
                "detect_code_blocks": True,
                "detect_urls": False,
                "detect_emails": False,
                "compute_complexity": False,  # Disabled for speed
            },
        ),
        # Training quality (heuristic-based, fast)
        SampleAnalyzerParams(
            id="training_quality",
            params={
                "compute_instruction_clarity": True,
                "compute_response_completeness": True,
                "compute_turn_quality": True,
            },
        ),
    ]
