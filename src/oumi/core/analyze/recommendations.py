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

"""Recommendations engine for generating actionable insights from analysis results."""

import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd

from oumi.utils.analysis_utils import (
    DistributionType,
    compute_multimodal_outliers,
    detect_distribution_type,
)


class RecommendationCategory(str, Enum):
    """Category of recommendation."""

    WARNING = "warning"
    INSIGHT = "insight"
    SUGGESTION = "suggestion"


class RecommendationSeverity(str, Enum):
    """Severity level of recommendation."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Recommendation:
    """A single recommendation with metadata.

    Attributes:
        category: The type of recommendation (warning, insight, suggestion).
        severity: How important/urgent the recommendation is.
        title: Short title summarizing the recommendation.
        description: Detailed description of the issue and recommended action.
        affected_samples: Number of samples affected by this issue.
        metric_name: Optional name of the metric this recommendation relates to.
        threshold: Optional threshold value that was exceeded.
        details: Optional additional details as a dictionary.
        sample_indices: Optional list of DataFrame indices for affected samples.
            Limited to max 20 samples for display in reports.
    """

    category: RecommendationCategory
    severity: RecommendationSeverity
    title: str
    description: str
    affected_samples: int
    metric_name: Optional[str] = None
    threshold: Optional[float] = None
    details: dict[str, Any] = field(default_factory=dict)
    sample_indices: list[int] = field(default_factory=list)

    # Maximum number of sample indices to store for display
    MAX_SAMPLE_INDICES: int = field(default=20, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert recommendation to a dictionary.

        Returns:
            Dictionary representation of the recommendation.
        """
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "affected_samples": self.affected_samples,
            "metric_name": self.metric_name,
            "threshold": self.threshold,
            "details": self.details,
            "sample_indices": self.sample_indices[:20],  # Limit to 20 for display
        }


class RecommendationsEngine:
    """Engine for generating recommendations from analysis results.

    This engine analyzes the results from dataset analysis and generates
    actionable recommendations to help users improve their datasets.

    The engine checks for:
    - Outliers: Values that are significantly above or below the mean
    - Duplicates: Exact duplicate content in messages
    - Distribution issues: Imbalanced role distributions, length distributions
    - Empty content: Messages with empty or very short content
    - Token length issues: Messages that may exceed model context windows
    """

    # Common model context window sizes for reference
    CONTEXT_WINDOW_SIZES = {
        "4k": 4096,
        "8k": 8192,
        "16k": 16384,
        "32k": 32768,
        "128k": 131072,
    }

    # Common special tokens that indicate data leakage
    SPECIAL_TOKEN_PATTERNS = [
        re.compile(
            r"<\|(?:endoftext|im_start|im_end|pad|unk|sep|cls)\|>", re.IGNORECASE
        ),
        re.compile(r"\[/?(?:INST|SYS|/INST)\]", re.IGNORECASE),
        re.compile(r"</?s>"),
        re.compile(r"<<SYS>>|<</SYS>>"),
        re.compile(r"<\|(?:system|user|assistant|begin_of_text|end_of_text)\|>"),
        re.compile(r"<\|eot_id\|>"),
        re.compile(r"<\|start_header_id\|>"),
        re.compile(r"<\|end_header_id\|>"),
    ]

    # Common instruction format patterns
    INSTRUCTION_FORMAT_PATTERNS = {
        "alpaca": re.compile(
            r"###\s*(Instruction|Input|Response|Output):", re.IGNORECASE
        ),
        "vicuna": re.compile(r"(USER|ASSISTANT|HUMAN|AI):", re.IGNORECASE),
        "chatml": re.compile(r"<\|im_(start|end)\|>", re.IGNORECASE),
        "llama": re.compile(r"\[INST\]|\[/INST\]", re.IGNORECASE),
        "openai": re.compile(r'"role"\s*:\s*"(user|assistant|system)"', re.IGNORECASE),
    }

    def __init__(
        self,
        *,
        outlier_std_threshold: float = 3.0,
        duplicate_warn_threshold: float = 0.05,
        imbalance_threshold: float = 0.8,
        empty_content_threshold: int = 5,
        short_content_threshold: int = 10,
        token_warn_thresholds: Optional[list[int]] = None,
        language_consistency_threshold: float = 0.9,
        pii_warn_threshold: float = 0.01,
        quality_score_threshold: float = 0.5,
    ):
        """Initialize the RecommendationsEngine.

        Args:
            outlier_std_threshold: Number of standard deviations from mean
                to consider a value an outlier. Default is 3.0.
            duplicate_warn_threshold: Fraction of duplicates to trigger a
                warning. Default is 0.05 (5%).
            imbalance_threshold: Fraction threshold for role imbalance warning.
                If any role has more than this fraction of messages, warn.
                Default is 0.8 (80%).
            empty_content_threshold: Character count below which content is
                considered empty. Default is 5.
            short_content_threshold: Word count below which content is
                considered very short. Default is 10.
            token_warn_thresholds: List of token counts to warn about
                (e.g., samples exceeding common context windows).
                Default is [4096, 8192, 16384].
            language_consistency_threshold: Fraction of samples that should be
                in the dominant language. Default is 0.9 (90%).
            pii_warn_threshold: Fraction of samples with PII to trigger warning.
                Default is 0.01 (1%).
            quality_score_threshold: Quality score below which to warn.
                Default is 0.5.
        """
        self.outlier_std_threshold = outlier_std_threshold
        self.duplicate_warn_threshold = duplicate_warn_threshold
        self.imbalance_threshold = imbalance_threshold
        self.empty_content_threshold = empty_content_threshold
        self.short_content_threshold = short_content_threshold
        self.token_warn_thresholds = token_warn_thresholds or [4096, 8192, 16384]
        self.language_consistency_threshold = language_consistency_threshold
        self.pii_warn_threshold = pii_warn_threshold
        self.quality_score_threshold = quality_score_threshold

        # Training quality thresholds
        self.instruction_clarity_threshold = 0.5
        self.response_completeness_threshold = 0.5

    def generate_recommendations(
        self,
        message_df: pd.DataFrame,
        conversation_df: pd.DataFrame,
        analysis_summary: dict[str, Any],
    ) -> list[Recommendation]:
        """Generate all recommendations from analysis results.

        Args:
            message_df: DataFrame with message-level analysis results.
            conversation_df: DataFrame with conversation-level analysis results.
            analysis_summary: Summary statistics from the analysis.

        Returns:
            List of Recommendation objects, sorted by severity (high first).
        """
        recommendations = []

        # Check for various issues
        recommendations.extend(self._check_outliers(message_df, analysis_summary))
        recommendations.extend(self._check_duplicates(message_df))
        recommendations.extend(self._check_empty_content(message_df))
        recommendations.extend(self._check_short_content(message_df))
        recommendations.extend(self._check_role_distribution(message_df))
        recommendations.extend(self._check_token_lengths(message_df, analysis_summary))
        recommendations.extend(
            self._check_conversation_length_distribution(conversation_df)
        )

        # New checks for quality analyzer results
        recommendations.extend(self._check_language_consistency(message_df))
        recommendations.extend(self._check_special_token_leakage(message_df))
        recommendations.extend(self._check_instruction_format_consistency(message_df))
        recommendations.extend(self._check_pii_detected(message_df))
        recommendations.extend(self._check_quality_scores(message_df))
        recommendations.extend(self._check_encoding_issues(message_df))
        recommendations.extend(self._check_high_repetition(message_df))

        # Training quality checks
        recommendations.extend(self._check_instruction_clarity(message_df))
        recommendations.extend(self._check_response_completeness(message_df))
        recommendations.extend(self._check_truncated_responses(message_df))

        # Content pattern checks (AI-specific quality issues)
        recommendations.extend(self._check_placeholder_text(message_df))
        recommendations.extend(self._check_hallucinated_experiences(message_df))
        recommendations.extend(self._check_nooutput_markers(message_df))
        recommendations.extend(self._check_refusal_patterns(message_df))

        # Instruction diversity checks
        recommendations.extend(self._check_instruction_diversity(message_df))
        recommendations.extend(self._check_concentrated_clusters(message_df))

        # IFD (Instruction-Following Difficulty) checks
        recommendations.extend(self._check_ifd_scores(message_df))

        # New analyzer checks (from "Fixing It in Post" paper)
        recommendations.extend(self._check_task_category_distribution(message_df))
        recommendations.extend(self._check_instruct_reward_scores(message_df))
        recommendations.extend(self._check_input_quality_issues(message_df))
        recommendations.extend(self._check_conversation_structure(message_df))
        recommendations.extend(self._check_safety_issues(message_df))
        recommendations.extend(self._check_difficulty_distribution(message_df))
        recommendations.extend(self._check_response_completeness_analyzer(message_df))

        # Sort by severity (high first, then medium, then low)
        severity_order = {
            RecommendationSeverity.HIGH: 0,
            RecommendationSeverity.MEDIUM: 1,
            RecommendationSeverity.LOW: 2,
        }
        recommendations.sort(key=lambda r: severity_order[r.severity])

        return recommendations

    def _check_outliers(
        self,
        df: pd.DataFrame,
        analysis_summary: dict[str, Any],
    ) -> list[Recommendation]:
        """Check for outliers in numeric analysis columns.

        Uses multimodal distribution detection to avoid false positives in
        bimodal/multimodal distributions. For unimodal distributions, uses
        standard 3-sigma detection. For multimodal distributions, detects
        outliers within each mode separately.

        Args:
            df: DataFrame with analysis results.
            analysis_summary: Summary statistics.

        Returns:
            List of recommendations for outlier issues.
        """
        recommendations = []

        # Get all numeric columns that look like analysis results
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        analysis_cols = [
            col
            for col in numeric_cols
            if any(
                pattern in col
                for pattern in ["_length_", "_diversity_", "_format_", "_count"]
            )
        ]

        for col in analysis_cols:
            if col not in df.columns:
                continue

            series = df[col].dropna()
            if len(series) < 2:
                continue

            # Detect distribution type first
            dist_result = detect_distribution_type(series)

            # Skip if insufficient data
            if dist_result.distribution_type == DistributionType.INSUFFICIENT_DATA:
                continue

            # Add insight about multimodal distribution if detected
            if dist_result.distribution_type in (
                DistributionType.BIMODAL,
                DistributionType.MULTIMODAL,
            ):
                recommendations.extend(
                    self._create_multimodal_insight(col, dist_result)
                )

            # Detect outliers using multimodal-aware method
            outlier_mask, outlier_details = compute_multimodal_outliers(
                series, dist_result, self.outlier_std_threshold
            )

            total_outliers = outlier_details["num_outliers"]

            if total_outliers > 0:
                outlier_pct = (total_outliers / len(series)) * 100
                severity = (
                    RecommendationSeverity.HIGH
                    if outlier_pct > 5
                    else (
                        RecommendationSeverity.MEDIUM
                        if outlier_pct > 1
                        else RecommendationSeverity.LOW
                    )
                )

                # Create human-readable metric name
                metric_name = col.replace("_", " ").replace("text content ", "")

                # Get indices of outlier samples (limit to 20)
                outlier_indices = series[outlier_mask].index.tolist()[:20]

                # Build description based on distribution type
                if dist_result.distribution_type == DistributionType.UNIMODAL:
                    mean = dist_result.global_statistics["mean"]
                    std = dist_result.global_statistics["std"]
                    upper_threshold = mean + (self.outlier_std_threshold * std)
                    lower_threshold = mean - (self.outlier_std_threshold * std)

                    # Count high/low outliers for unimodal
                    values = series.to_numpy()
                    high_outliers = int(np.sum(values > upper_threshold))
                    low_outliers = int(np.sum(values < lower_threshold))

                    description = (
                        f"Found {total_outliers} samples ({outlier_pct:.1f}%) with "
                        f"values outside {self.outlier_std_threshold} standard "
                        f"deviations from the mean. High outliers: {high_outliers}, "
                        f"Low outliers: {low_outliers}. Consider reviewing these "
                        f"samples for potential data quality issues."
                    )
                    details = {
                        "distribution_type": "unimodal",
                        "mean": round(mean, 2),
                        "std": round(std, 2),
                        "upper_threshold": round(upper_threshold, 2),
                        "lower_threshold": round(lower_threshold, 2),
                        "high_outliers": high_outliers,
                        "low_outliers": low_outliers,
                    }
                else:
                    # Multimodal description
                    mode_info = ", ".join(
                        f"mode {ms.mode_id}: μ={ms.mean}, σ={ms.std}"
                        for ms in dist_result.mode_statistics
                    )
                    description = (
                        f"Found {total_outliers} samples ({outlier_pct:.1f}%) "
                        f"that are outliers within their respective modes. "
                        f"Distribution has {dist_result.num_modes} modes "
                        f"({mode_info}). Outliers are samples more than "
                        f"{self.outlier_std_threshold} std from mode mean."
                    )
                    details = {
                        "distribution_type": dist_result.distribution_type.value,
                        "num_modes": dist_result.num_modes,
                        "mode_statistics": [
                            {
                                "mode_id": ms.mode_id,
                                "mean": ms.mean,
                                "std": ms.std,
                                "count": ms.count,
                            }
                            for ms in dist_result.mode_statistics
                        ],
                        "outliers_per_mode": outlier_details.get(
                            "outliers_per_mode", {}
                        ),
                    }

                recommendations.append(
                    Recommendation(
                        category=RecommendationCategory.WARNING,
                        severity=severity,
                        title=f"Outliers detected in {metric_name}",
                        description=description,
                        affected_samples=total_outliers,
                        metric_name=col,
                        threshold=self.outlier_std_threshold,
                        details=details,
                        sample_indices=outlier_indices,
                    )
                )

        return recommendations

    def _create_multimodal_insight(
        self, col: str, dist_result
    ) -> list[Recommendation]:
        """Create an insight recommendation about a multimodal distribution.

        Args:
            col: Column name
            dist_result: Distribution analysis result

        Returns:
            List containing a single insight recommendation
        """
        metric_name = col.replace("_", " ").replace("text content ", "")

        # Format mode information
        mode_summaries = []
        for ms in dist_result.mode_statistics:
            pct = ms.weight * 100
            mode_summaries.append(f"Mode {ms.mode_id + 1}: {ms.count} samples "
                                  f"({pct:.1f}%), mean={ms.mean}, std={ms.std}")

        mode_text = "; ".join(mode_summaries)

        return [
            Recommendation(
                category=RecommendationCategory.INSIGHT,
                severity=RecommendationSeverity.LOW,
                title=f"Multimodal distribution in {metric_name}",
                description=(
                    f"Detected {dist_result.num_modes} distinct modes in the "
                    f"distribution (confidence: {dist_result.confidence:.0%}). "
                    f"{mode_text}. This may indicate different types of content "
                    f"(e.g., short questions vs long explanations). "
                    f"Outlier detection uses per-mode statistics to avoid "
                    f"false positives."
                ),
                affected_samples=dist_result.global_statistics["count"],
                metric_name=col,
                details={
                    "distribution_type": dist_result.distribution_type.value,
                    "num_modes": dist_result.num_modes,
                    "confidence": dist_result.confidence,
                    "mode_statistics": [
                        {
                            "mode_id": ms.mode_id,
                            "mean": ms.mean,
                            "std": ms.std,
                            "count": ms.count,
                            "weight": ms.weight,
                        }
                        for ms in dist_result.mode_statistics
                    ],
                },
            )
        ]

    def _check_duplicates(self, df: pd.DataFrame) -> list[Recommendation]:
        """Check for duplicate content in messages.

        Excludes system role messages from duplicate detection, since system
        prompts are typically identical across conversations by design.

        Args:
            df: DataFrame with message data.

        Returns:
            List of recommendations for duplicate issues.
        """
        recommendations = []

        if "text_content" not in df.columns:
            return recommendations

        # Filter out system messages for duplicate detection
        # System prompts are typically identical by design in instruction datasets
        if "role" in df.columns:
            non_system_df = df[df["role"] != "system"]
        else:
            non_system_df = df

        if len(non_system_df) == 0:
            return recommendations

        # Check for exact duplicates (excluding system messages)
        total_messages = len(non_system_df)
        duplicates = non_system_df["text_content"].duplicated(keep=False)
        duplicate_count = duplicates.sum()

        if duplicate_count > 0:
            duplicate_pct = duplicate_count / total_messages
            unique_duplicated = non_system_df[duplicates]["text_content"].nunique()

            if duplicate_pct >= self.duplicate_warn_threshold:
                severity = (
                    RecommendationSeverity.HIGH
                    if duplicate_pct > 0.2
                    else (
                        RecommendationSeverity.MEDIUM
                        if duplicate_pct > 0.1
                        else RecommendationSeverity.LOW
                    )
                )

                # Get indices of duplicate samples (limit to 20)
                duplicate_indices = non_system_df[duplicates].index.tolist()[:20]

                recommendations.append(
                    Recommendation(
                        category=RecommendationCategory.WARNING,
                        severity=severity,
                        title="Duplicate content detected",
                        description=(
                            f"Found {duplicate_count} messages ({duplicate_pct * 100:.1f}%) "
                            f"that are exact duplicates (excluding system prompts), "
                            f"representing {unique_duplicated} unique repeated texts. "
                            f"Consider deduplicating your dataset to improve training "
                            f"diversity."
                        ),
                        affected_samples=duplicate_count,
                        metric_name="text_content",
                        threshold=self.duplicate_warn_threshold,
                        details={
                            "duplicate_count": int(duplicate_count),
                            "unique_duplicated_texts": int(unique_duplicated),
                            "duplicate_percentage": round(duplicate_pct * 100, 2),
                        },
                        sample_indices=duplicate_indices,
                    )
                )

        return recommendations

    def _check_empty_content(self, df: pd.DataFrame) -> list[Recommendation]:
        """Check for empty or near-empty content.

        Args:
            df: DataFrame with message data.

        Returns:
            List of recommendations for empty content issues.
        """
        recommendations = []

        if "text_content" not in df.columns:
            return recommendations

        # Find empty or very short content
        char_counts = df["text_content"].astype(str).str.len()
        empty_mask = char_counts <= self.empty_content_threshold
        empty_count = empty_mask.sum()

        if empty_count > 0:
            empty_pct = (empty_count / len(df)) * 100
            severity = (
                RecommendationSeverity.HIGH
                if empty_pct > 5
                else (
                    RecommendationSeverity.MEDIUM
                    if empty_pct > 1
                    else RecommendationSeverity.LOW
                )
            )

            # Get indices of empty samples (limit to 20)
            empty_indices = df[empty_mask].index.tolist()[:20]

            recommendations.append(
                Recommendation(
                    category=RecommendationCategory.WARNING,
                    severity=severity,
                    title="Empty or near-empty messages detected",
                    description=(
                        f"Found {empty_count} messages ({empty_pct:.1f}%) with "
                        f"{self.empty_content_threshold} or fewer characters. "
                        f"These may indicate data quality issues or placeholder "
                        f"content that should be reviewed."
                    ),
                    affected_samples=int(empty_count),
                    metric_name="text_content",
                    threshold=float(self.empty_content_threshold),
                    details={"empty_count": int(empty_count)},
                    sample_indices=empty_indices,
                )
            )

        return recommendations

    def _check_short_content(self, df: pd.DataFrame) -> list[Recommendation]:
        """Check for very short content.

        Args:
            df: DataFrame with message data.

        Returns:
            List of recommendations for short content issues.
        """
        recommendations = []

        # Look for word count column
        word_count_cols = [col for col in df.columns if "word_count" in col]

        for col in word_count_cols:
            short_mask = df[col] < self.short_content_threshold
            short_count = short_mask.sum()
            short_pct = (short_count / len(df)) * 100

            if short_count > 0 and short_pct > 10:
                severity = (
                    RecommendationSeverity.MEDIUM
                    if short_pct > 25
                    else RecommendationSeverity.LOW
                )

                # Get indices of short samples (limit to 20)
                short_indices = df[short_mask].index.tolist()[:20]

                recommendations.append(
                    Recommendation(
                        category=RecommendationCategory.INSIGHT,
                        severity=severity,
                        title="Many short messages detected",
                        description=(
                            f"Found {short_count} messages ({short_pct:.1f}%) with "
                            f"fewer than {self.short_content_threshold} words. "
                            f"This may be intentional (e.g., short responses) or "
                            f"indicate low-quality samples worth reviewing."
                        ),
                        affected_samples=int(short_count),
                        metric_name=col,
                        threshold=float(self.short_content_threshold),
                        details={"short_count": int(short_count)},
                        sample_indices=short_indices,
                    )
                )
                break  # Only report once

        return recommendations

    def _check_role_distribution(self, df: pd.DataFrame) -> list[Recommendation]:
        """Check for imbalanced role distributions.

        Args:
            df: DataFrame with message data.

        Returns:
            List of recommendations for role distribution issues.
        """
        recommendations = []

        if "role" not in df.columns:
            return recommendations

        role_counts = df["role"].value_counts(normalize=True)

        for role, fraction in role_counts.items():
            if fraction > self.imbalance_threshold:
                recommendations.append(
                    Recommendation(
                        category=RecommendationCategory.INSIGHT,
                        severity=RecommendationSeverity.MEDIUM,
                        title=f"Role distribution imbalance: {role}",
                        description=(
                            f"The '{role}' role accounts for {fraction * 100:.1f}% of "
                            f"all messages. This may indicate an imbalanced dataset. "
                            f"For conversational fine-tuning, a more balanced "
                            f"distribution is typically preferred."
                        ),
                        affected_samples=int(role_counts[role] * len(df)),
                        metric_name="role",
                        threshold=self.imbalance_threshold,
                        details={
                            "role_distribution": {
                                str(r): round(f * 100, 2)
                                for r, f in role_counts.items()
                            }
                        },
                    )
                )

        return recommendations

    def _check_token_lengths(
        self,
        df: pd.DataFrame,
        analysis_summary: dict[str, Any],
    ) -> list[Recommendation]:
        """Check for token lengths that may exceed context windows.

        Args:
            df: DataFrame with message data.
            analysis_summary: Summary statistics.

        Returns:
            List of recommendations for token length issues.
        """
        recommendations = []

        # Look for token count column
        token_cols = [col for col in df.columns if "token_count" in col]

        for col in token_cols:
            if col not in df.columns:
                continue

            series = df[col].dropna()
            if len(series) == 0:
                continue

            for threshold in self.token_warn_thresholds:
                exceeds_mask = series > threshold
                exceeds = exceeds_mask.sum()
                if exceeds > 0:
                    exceeds_pct = (exceeds / len(series)) * 100
                    threshold_name = self._get_context_window_name(threshold)

                    severity = (
                        RecommendationSeverity.HIGH
                        if exceeds_pct > 10
                        else (
                            RecommendationSeverity.MEDIUM
                            if exceeds_pct > 5
                            else RecommendationSeverity.LOW
                        )
                    )

                    # Get indices of samples exceeding threshold (limit to 20)
                    exceeds_indices = series[exceeds_mask].index.tolist()[:20]

                    recommendations.append(
                        Recommendation(
                            category=RecommendationCategory.WARNING,
                            severity=severity,
                            title=f"Messages exceeding {threshold_name} tokens",
                            description=(
                                f"Found {exceeds} messages ({exceeds_pct:.1f}%) with "
                                f"more than {threshold:,} tokens. These may be "
                                f"truncated when using models with {threshold_name} "
                                f"context windows. Consider truncating or splitting "
                                f"long messages."
                            ),
                            affected_samples=int(exceeds),
                            metric_name=col,
                            threshold=float(threshold),
                            details={
                                "exceeds_count": int(exceeds),
                                "max_tokens": int(series.max()),
                            },
                            sample_indices=exceeds_indices,
                        )
                    )

            # Only check first token column found
            break

        return recommendations

    def _check_conversation_length_distribution(
        self, df: pd.DataFrame
    ) -> list[Recommendation]:
        """Check conversation length distribution for issues.

        Args:
            df: DataFrame with conversation-level data.

        Returns:
            List of recommendations for conversation length issues.
        """
        recommendations = []

        if "num_messages" not in df.columns or len(df) == 0:
            return recommendations

        num_messages = df["num_messages"]

        # Check for single-turn conversations
        single_turn = (num_messages <= 2).sum()
        single_turn_pct = (single_turn / len(df)) * 100

        if single_turn_pct > 50:
            recommendations.append(
                Recommendation(
                    category=RecommendationCategory.INSIGHT,
                    severity=RecommendationSeverity.LOW,
                    title="Many single-turn conversations",
                    description=(
                        f"Found {single_turn} conversations ({single_turn_pct:.1f}%) "
                        f"with 2 or fewer messages. If training for multi-turn "
                        f"dialogue, consider augmenting with longer conversations."
                    ),
                    affected_samples=int(single_turn),
                    metric_name="num_messages",
                    details={"single_turn_count": int(single_turn)},
                )
            )

        # Check for very long conversations
        very_long = (num_messages > 20).sum()
        if very_long > 0:
            very_long_pct = (very_long / len(df)) * 100
            recommendations.append(
                Recommendation(
                    category=RecommendationCategory.INSIGHT,
                    severity=RecommendationSeverity.LOW,
                    title="Very long conversations detected",
                    description=(
                        f"Found {very_long} conversations ({very_long_pct:.1f}%) "
                        f"with more than 20 messages. Very long conversations may "
                        f"exceed context windows when tokenized."
                    ),
                    affected_samples=int(very_long),
                    metric_name="num_messages",
                    details={
                        "very_long_count": int(very_long),
                        "max_messages": int(num_messages.max()),
                    },
                )
            )

        return recommendations

    def _get_context_window_name(self, tokens: int) -> str:
        """Get human-readable name for a context window size.

        Args:
            tokens: Number of tokens.

        Returns:
            Human-readable name like "4k" or "8k".
        """
        for name, size in self.CONTEXT_WINDOW_SIZES.items():
            if tokens == size:
                return name
        return f"{tokens:,}"

    def _check_language_consistency(self, df: pd.DataFrame) -> list[Recommendation]:
        """Check for language inconsistency across messages.

        Args:
            df: DataFrame with message data.

        Returns:
            List of recommendations for language consistency issues.
        """
        recommendations = []

        # Look for language detection columns from quality analyzer
        lang_cols = [col for col in df.columns if "detected_language" in col]

        for col in lang_cols:
            if col not in df.columns:
                continue

            # Filter out empty language detections
            languages = df[col].dropna()
            languages = languages[languages != ""]

            if len(languages) < 10:
                continue

            # Count language distribution
            lang_counts = Counter(languages)
            total = sum(lang_counts.values())

            if total == 0:
                continue

            # Find dominant language
            dominant_lang, dominant_count = lang_counts.most_common(1)[0]
            dominant_fraction = dominant_count / total

            # Check for mixed languages
            if dominant_fraction < self.language_consistency_threshold:
                other_langs = {
                    lang: count
                    for lang, count in lang_counts.items()
                    if lang != dominant_lang
                }
                other_count = sum(other_langs.values())

                severity = (
                    RecommendationSeverity.HIGH
                    if dominant_fraction < 0.7
                    else (
                        RecommendationSeverity.MEDIUM
                        if dominant_fraction < 0.85
                        else RecommendationSeverity.LOW
                    )
                )

                # Get indices of non-dominant language samples (limit to 20)
                non_dominant_mask = df[col] != dominant_lang
                non_dominant_indices = df[non_dominant_mask].index.tolist()[:20]

                recommendations.append(
                    Recommendation(
                        category=RecommendationCategory.WARNING,
                        severity=severity,
                        title="Mixed language content detected",
                        description=(
                            f"Only {dominant_fraction * 100:.1f}% of messages are in the "
                            f"dominant language ({dominant_lang}). Found {other_count} "
                            f"messages in other languages. Mixed-language datasets may "
                            f"cause inconsistent model behavior. Consider filtering to "
                            f"a single language or intentionally creating a multilingual "
                            f"dataset."
                        ),
                        affected_samples=int(other_count),
                        metric_name=col,
                        threshold=self.language_consistency_threshold,
                        details={
                            "dominant_language": dominant_lang,
                            "dominant_fraction": round(dominant_fraction, 3),
                            "language_distribution": dict(lang_counts.most_common(5)),
                        },
                        sample_indices=non_dominant_indices,
                    )
                )
            break  # Only check first language column

        return recommendations

    def _check_special_token_leakage(self, df: pd.DataFrame) -> list[Recommendation]:
        """Check for leaked special tokens in content.

        Args:
            df: DataFrame with message data.

        Returns:
            List of recommendations for special token leakage.
        """
        recommendations = []

        # Check for has_special_tokens column from quality analyzer
        special_token_cols = [col for col in df.columns if "has_special_tokens" in col]

        for col in special_token_cols:
            if col not in df.columns:
                continue

            leaked_mask = df[col] == True  # noqa: E712
            leaked_count = leaked_mask.sum()
            if leaked_count > 0:
                leaked_pct = (leaked_count / len(df)) * 100

                severity = (
                    RecommendationSeverity.HIGH
                    if leaked_pct > 5
                    else (
                        RecommendationSeverity.MEDIUM
                        if leaked_pct > 1
                        else RecommendationSeverity.LOW
                    )
                )

                # Get indices of leaked samples (limit to 20)
                leaked_indices = df[leaked_mask].index.tolist()[:20]

                recommendations.append(
                    Recommendation(
                        category=RecommendationCategory.WARNING,
                        severity=severity,
                        title="Special token leakage detected",
                        description=(
                            f"Found {leaked_count} messages ({leaked_pct:.1f}%) "
                            f"containing leaked special tokens (e.g., <|endoftext|>, "
                            f"[INST], <s>). These tokens may interfere with model "
                            f"training and cause unexpected behavior. Remove or escape "
                            f"these tokens before training."
                        ),
                        affected_samples=int(leaked_count),
                        metric_name=col,
                        details={"leaked_count": int(leaked_count)},
                        sample_indices=leaked_indices,
                    )
                )
            break  # Only report once

        # Also check raw text if no quality analyzer was used
        if not special_token_cols and "text_content" in df.columns:
            leaked_count = 0
            for _, row in df.iterrows():
                text = str(row.get("text_content", ""))
                for pattern in self.SPECIAL_TOKEN_PATTERNS:
                    if pattern.search(text):
                        leaked_count += 1
                        break

            if leaked_count > 0:
                leaked_pct = (leaked_count / len(df)) * 100

                severity = (
                    RecommendationSeverity.HIGH
                    if leaked_pct > 5
                    else (
                        RecommendationSeverity.MEDIUM
                        if leaked_pct > 1
                        else RecommendationSeverity.LOW
                    )
                )

                recommendations.append(
                    Recommendation(
                        category=RecommendationCategory.WARNING,
                        severity=severity,
                        title="Special token leakage detected",
                        description=(
                            f"Found {leaked_count} messages ({leaked_pct:.1f}%) "
                            f"containing leaked special tokens. These may interfere "
                            f"with model training."
                        ),
                        affected_samples=int(leaked_count),
                        metric_name="text_content",
                        details={"leaked_count": int(leaked_count)},
                    )
                )

        return recommendations

    def _check_instruction_format_consistency(
        self, df: pd.DataFrame
    ) -> list[Recommendation]:
        """Check for inconsistent instruction formatting.

        Args:
            df: DataFrame with message data.

        Returns:
            List of recommendations for format consistency issues.
        """
        recommendations = []

        if "text_content" not in df.columns:
            return recommendations

        # Detect format patterns in content
        format_counts: dict[str, int] = {}
        for _, row in df.iterrows():
            text = str(row.get("text_content", ""))
            for format_name, pattern in self.INSTRUCTION_FORMAT_PATTERNS.items():
                if pattern.search(text):
                    format_counts[format_name] = format_counts.get(format_name, 0) + 1

        # Check for multiple formats
        if len(format_counts) > 1:
            total_formatted = sum(format_counts.values())
            format_str = ", ".join(
                f"{name}: {count}" for name, count in format_counts.items()
            )

            recommendations.append(
                Recommendation(
                    category=RecommendationCategory.WARNING,
                    severity=RecommendationSeverity.MEDIUM,
                    title="Inconsistent instruction formatting detected",
                    description=(
                        f"Found multiple instruction format patterns in the dataset: "
                        f"{format_str}. Mixing formats may confuse the model and reduce "
                        f"training effectiveness. Consider standardizing to a single "
                        f"format."
                    ),
                    affected_samples=int(total_formatted),
                    metric_name="text_content",
                    details={"format_distribution": format_counts},
                )
            )

        return recommendations

    def _check_pii_detected(self, df: pd.DataFrame) -> list[Recommendation]:
        """Check for PII in the dataset.

        Args:
            df: DataFrame with message data.

        Returns:
            List of recommendations for PII detection.
        """
        recommendations = []

        # Look for PII columns from quality analyzer
        pii_cols = [col for col in df.columns if "has_pii" in col]

        for col in pii_cols:
            if col not in df.columns:
                continue

            pii_mask = df[col] == True  # noqa: E712
            pii_count = pii_mask.sum()
            if pii_count > 0:
                pii_pct = pii_count / len(df)

                if pii_pct >= self.pii_warn_threshold:
                    severity = (
                        RecommendationSeverity.HIGH
                        if pii_pct > 0.1
                        else (
                            RecommendationSeverity.MEDIUM
                            if pii_pct > 0.05
                            else RecommendationSeverity.LOW
                        )
                    )

                    # Get PII types if available
                    pii_types_col = col.replace("has_pii", "pii_types")
                    pii_types_detail = ""
                    if pii_types_col in df.columns:
                        all_types = df[pii_types_col].dropna()
                        all_types = all_types[all_types != ""]
                        type_counts = Counter(
                            t for types in all_types for t in types.split(",") if t
                        )
                        if type_counts:
                            pii_types_detail = (
                                f" Types detected: "
                                f"{', '.join(f'{t} ({c})' for t, c in type_counts.most_common(5))}."
                            )

                    # Get indices of PII samples (limit to 20)
                    pii_indices = df[pii_mask].index.tolist()[:20]

                    recommendations.append(
                        Recommendation(
                            category=RecommendationCategory.WARNING,
                            severity=severity,
                            title="PII (Personally Identifiable Information) detected",
                            description=(
                                f"Found {pii_count} messages ({pii_pct * 100:.1f}%) "
                                f"containing potential PII.{pii_types_detail} "
                                f"Consider redacting or removing PII before training "
                                f"to prevent the model from memorizing sensitive data."
                            ),
                            affected_samples=int(pii_count),
                            metric_name=col,
                            threshold=self.pii_warn_threshold,
                            details={"pii_count": int(pii_count)},
                            sample_indices=pii_indices,
                        )
                    )
            break  # Only report once

        return recommendations

    def _check_quality_scores(self, df: pd.DataFrame) -> list[Recommendation]:
        """Check for low quality scores.

        Args:
            df: DataFrame with message data.

        Returns:
            List of recommendations for quality score issues.
        """
        recommendations = []

        # Look for quality score columns
        quality_cols = [col for col in df.columns if "quality_score" in col]

        for col in quality_cols:
            if col not in df.columns:
                continue

            scores = df[col].dropna()
            if len(scores) == 0:
                continue

            low_quality_mask = scores < self.quality_score_threshold
            low_quality_count = low_quality_mask.sum()
            if low_quality_count > 0:
                low_quality_pct = (low_quality_count / len(scores)) * 100
                avg_score = scores.mean()

                severity = (
                    RecommendationSeverity.HIGH
                    if low_quality_pct > 20
                    else (
                        RecommendationSeverity.MEDIUM
                        if low_quality_pct > 10
                        else RecommendationSeverity.LOW
                    )
                )

                # Get indices of low quality samples (limit to 20)
                low_quality_indices = scores[low_quality_mask].index.tolist()[:20]

                recommendations.append(
                    Recommendation(
                        category=RecommendationCategory.WARNING,
                        severity=severity,
                        title="Low quality samples detected",
                        description=(
                            f"Found {low_quality_count} messages ({low_quality_pct:.1f}%) "
                            f"with quality scores below {self.quality_score_threshold}. "
                            f"Average quality score: {avg_score:.2f}. Consider filtering "
                            f"or reviewing low-quality samples before training."
                        ),
                        affected_samples=int(low_quality_count),
                        metric_name=col,
                        threshold=self.quality_score_threshold,
                        details={
                            "low_quality_count": int(low_quality_count),
                            "average_score": round(avg_score, 3),
                            "min_score": round(scores.min(), 3),
                            "max_score": round(scores.max(), 3),
                        },
                        sample_indices=low_quality_indices,
                    )
                )
            break  # Only report once

        return recommendations

    def _check_encoding_issues(self, df: pd.DataFrame) -> list[Recommendation]:
        """Check for encoding issues in content.

        Args:
            df: DataFrame with message data.

        Returns:
            List of recommendations for encoding issues.
        """
        recommendations = []

        # Look for encoding issue columns from quality analyzer
        encoding_cols = [col for col in df.columns if "has_encoding_issues" in col]

        for col in encoding_cols:
            if col not in df.columns:
                continue

            issue_mask = df[col] == True  # noqa: E712
            issue_count = issue_mask.sum()
            if issue_count > 0:
                issue_pct = (issue_count / len(df)) * 100

                severity = (
                    RecommendationSeverity.HIGH
                    if issue_pct > 5
                    else (
                        RecommendationSeverity.MEDIUM
                        if issue_pct > 1
                        else RecommendationSeverity.LOW
                    )
                )

                # Get indices of encoding issue samples (limit to 20)
                issue_indices = df[issue_mask].index.tolist()[:20]

                recommendations.append(
                    Recommendation(
                        category=RecommendationCategory.WARNING,
                        severity=severity,
                        title="Encoding issues detected",
                        description=(
                            f"Found {issue_count} messages ({issue_pct:.1f}%) with "
                            f"potential encoding issues (e.g., mojibake, invalid "
                            f"characters). These may indicate data corruption or "
                            f"incorrect character encoding. Consider re-encoding or "
                            f"cleaning affected samples."
                        ),
                        affected_samples=int(issue_count),
                        metric_name=col,
                        details={"issue_count": int(issue_count)},
                        sample_indices=issue_indices,
                    )
                )
            break  # Only report once

        return recommendations

    def _check_high_repetition(self, df: pd.DataFrame) -> list[Recommendation]:
        """Check for highly repetitive content.

        Args:
            df: DataFrame with message data.

        Returns:
            List of recommendations for repetition issues.
        """
        recommendations = []

        # Look for repetition columns from quality analyzer
        repetition_cols = [col for col in df.columns if "has_high_repetition" in col]

        for col in repetition_cols:
            if col not in df.columns:
                continue

            repetitive_mask = df[col] == True  # noqa: E712
            repetitive_count = repetitive_mask.sum()
            if repetitive_count > 0:
                repetitive_pct = (repetitive_count / len(df)) * 100

                severity = (
                    RecommendationSeverity.MEDIUM
                    if repetitive_pct > 10
                    else RecommendationSeverity.LOW
                )

                # Get indices of repetitive samples (limit to 20)
                repetitive_indices = df[repetitive_mask].index.tolist()[:20]

                recommendations.append(
                    Recommendation(
                        category=RecommendationCategory.INSIGHT,
                        severity=severity,
                        title="Highly repetitive content detected",
                        description=(
                            f"Found {repetitive_count} messages ({repetitive_pct:.1f}%) "
                            f"with high repetition ratios. Repetitive content may cause "
                            f"the model to learn repetitive patterns. Consider reviewing "
                            f"or filtering these samples."
                        ),
                        affected_samples=int(repetitive_count),
                        metric_name=col,
                        details={"repetitive_count": int(repetitive_count)},
                        sample_indices=repetitive_indices,
                    )
                )
            break  # Only report once

        return recommendations

    def _check_instruction_clarity(self, df: pd.DataFrame) -> list[Recommendation]:
        """Check for unclear instructions in user messages.

        Args:
            df: DataFrame with message data.

        Returns:
            List of recommendations for instruction clarity issues.
        """
        recommendations = []

        # Look for instruction clarity score columns from training quality analyzer
        clarity_cols = [col for col in df.columns if "instruction_clarity_score" in col]

        for col in clarity_cols:
            if col not in df.columns:
                continue

            # Only consider non-null values (user messages)
            scores = df[col].dropna()
            if len(scores) == 0:
                continue

            low_clarity_mask = scores < self.instruction_clarity_threshold
            low_clarity_count = low_clarity_mask.sum()

            if low_clarity_count > 0:
                low_clarity_pct = (low_clarity_count / len(scores)) * 100
                avg_score = scores.mean()

                severity = (
                    RecommendationSeverity.HIGH
                    if low_clarity_pct > 30
                    else (
                        RecommendationSeverity.MEDIUM
                        if low_clarity_pct > 15
                        else RecommendationSeverity.LOW
                    )
                )

                # Get indices of low clarity samples (limit to 20)
                low_clarity_indices = scores[low_clarity_mask].index.tolist()[:20]

                recommendations.append(
                    Recommendation(
                        category=RecommendationCategory.WARNING,
                        severity=severity,
                        title="Unclear instructions detected",
                        description=(
                            f"Found {low_clarity_count} user instructions "
                            f"({low_clarity_pct:.1f}%) with low clarity scores "
                            f"(below {self.instruction_clarity_threshold}). "
                            f"Average clarity score: {avg_score:.2f}. "
                            f"Unclear instructions may lead to inconsistent model "
                            f"behavior. Consider making instructions more specific "
                            f"with clear action verbs and concrete details."
                        ),
                        affected_samples=int(low_clarity_count),
                        metric_name=col,
                        threshold=self.instruction_clarity_threshold,
                        details={
                            "low_clarity_count": int(low_clarity_count),
                            "average_score": round(avg_score, 3),
                            "min_score": round(scores.min(), 3),
                        },
                        sample_indices=low_clarity_indices,
                    )
                )
            break  # Only report once

        return recommendations

    def _check_response_completeness(self, df: pd.DataFrame) -> list[Recommendation]:
        """Check for incomplete responses in assistant messages.

        Args:
            df: DataFrame with message data.

        Returns:
            List of recommendations for response completeness issues.
        """
        recommendations = []

        # Look for response completeness score columns
        completeness_cols = [
            col for col in df.columns if "response_completeness_score" in col
        ]

        for col in completeness_cols:
            if col not in df.columns:
                continue

            # Only consider non-null values (assistant messages)
            scores = df[col].dropna()
            if len(scores) == 0:
                continue

            incomplete_mask = scores < self.response_completeness_threshold
            incomplete_count = incomplete_mask.sum()

            if incomplete_count > 0:
                incomplete_pct = (incomplete_count / len(scores)) * 100
                avg_score = scores.mean()

                severity = (
                    RecommendationSeverity.HIGH
                    if incomplete_pct > 25
                    else (
                        RecommendationSeverity.MEDIUM
                        if incomplete_pct > 10
                        else RecommendationSeverity.LOW
                    )
                )

                # Get indices of incomplete samples (limit to 20)
                incomplete_indices = scores[incomplete_mask].index.tolist()[:20]

                recommendations.append(
                    Recommendation(
                        category=RecommendationCategory.WARNING,
                        severity=severity,
                        title="Incomplete responses detected",
                        description=(
                            f"Found {incomplete_count} assistant responses "
                            f"({incomplete_pct:.1f}%) with low completeness scores "
                            f"(below {self.response_completeness_threshold}). "
                            f"Average completeness score: {avg_score:.2f}. "
                            f"Incomplete responses may teach the model to generate "
                            f"truncated or minimal outputs. Consider expanding "
                            f"short responses or removing low-quality samples."
                        ),
                        affected_samples=int(incomplete_count),
                        metric_name=col,
                        threshold=self.response_completeness_threshold,
                        details={
                            "incomplete_count": int(incomplete_count),
                            "average_score": round(avg_score, 3),
                            "min_score": round(scores.min(), 3),
                        },
                        sample_indices=incomplete_indices,
                    )
                )
            break  # Only report once

        return recommendations

    def _check_truncated_responses(self, df: pd.DataFrame) -> list[Recommendation]:
        """Check for truncated responses (improper endings).

        Args:
            df: DataFrame with message data.

        Returns:
            List of recommendations for truncated response issues.
        """
        recommendations = []

        # Look for proper ending columns from training quality analyzer
        ending_cols = [col for col in df.columns if "has_proper_ending" in col]

        for col in ending_cols:
            if col not in df.columns:
                continue

            # Only consider non-null values (assistant messages)
            endings = df[col].dropna()
            if len(endings) == 0:
                continue

            truncated_mask = endings == False  # noqa: E712
            truncated_count = truncated_mask.sum()

            if truncated_count > 0:
                truncated_pct = (truncated_count / len(endings)) * 100

                severity = (
                    RecommendationSeverity.HIGH
                    if truncated_pct > 15
                    else (
                        RecommendationSeverity.MEDIUM
                        if truncated_pct > 5
                        else RecommendationSeverity.LOW
                    )
                )

                # Get indices of truncated samples (limit to 20)
                truncated_indices = endings[truncated_mask].index.tolist()[:20]

                recommendations.append(
                    Recommendation(
                        category=RecommendationCategory.WARNING,
                        severity=severity,
                        title="Truncated responses detected",
                        description=(
                            f"Found {truncated_count} assistant responses "
                            f"({truncated_pct:.1f}%) that appear to be truncated "
                            f"(ending mid-sentence or with incomplete punctuation). "
                            f"Training on truncated responses may cause the model "
                            f"to generate incomplete outputs. Consider completing "
                            f"or removing these samples."
                        ),
                        affected_samples=int(truncated_count),
                        metric_name=col,
                        details={"truncated_count": int(truncated_count)},
                        sample_indices=truncated_indices,
                    )
                )
            break  # Only report once

        return recommendations

    def _check_instruction_diversity(self, df: pd.DataFrame) -> list[Recommendation]:
        """Check for low question diversity based on clustering results.

        Args:
            df: DataFrame with message data.

        Returns:
            List of recommendations for question diversity issues.
        """
        recommendations = []

        # Look for diversity-related columns from QuestionDiversityAnalyzer
        # These would be in dataset-level metrics, but we can detect issues
        # from the cluster distribution in per-sample data

        # Check for cluster_id columns
        cluster_cols = [
            col for col in df.columns if "question_diversity_cluster_id" in col
        ]

        for col in cluster_cols:
            if col not in df.columns:
                continue

            # Get cluster IDs, excluding None (non-user rows)
            cluster_ids = df[col].dropna()
            if len(cluster_ids) < 10:
                continue

            # Count cluster sizes
            cluster_counts = cluster_ids.value_counts()
            total_samples = len(cluster_ids)

            # IMPORTANT: cluster_id == -1 means "noise" in DBSCAN
            # Noise points are DIVERSE (not similar to others) - this is GOOD
            noise_count = cluster_counts.get(-1, 0)
            noise_ratio = noise_count / total_samples if total_samples > 0 else 0

            # Exclude noise (-1) from cluster count
            n_clusters = len([c for c in cluster_counts.index if c != -1])

            # High noise ratio means HIGH diversity - don't warn
            if noise_ratio > 0.5:
                # More than half are unique/diverse - this is good!
                continue

            # Check for low diversity (few clusters relative to sample size)
            # If we have fewer than log2(n) clusters, diversity is low
            import math

            expected_min_clusters = max(5, int(math.log2(total_samples)))

            if n_clusters < expected_min_clusters:
                severity = (
                    RecommendationSeverity.HIGH
                    if n_clusters <= 3
                    else (
                        RecommendationSeverity.MEDIUM
                        if n_clusters <= 5
                        else RecommendationSeverity.LOW
                    )
                )

                recommendations.append(
                    Recommendation(
                        category=RecommendationCategory.WARNING,
                        severity=severity,
                        title="Low user question diversity detected",
                        description=(
                            f"Found only {n_clusters} question clusters for "
                            f"{total_samples} user questions (excluding "
                            f"{noise_count} unique questions). This suggests "
                            f"your clustered questions are narrowly distributed. "
                            f"Consider expanding question variety."
                        ),
                        affected_samples=int(total_samples - noise_count),
                        metric_name=col,
                        details={
                            "num_clusters": int(n_clusters),
                            "total_questions": int(total_samples),
                            "unique_questions": int(noise_count),
                            "expected_min_clusters": int(expected_min_clusters),
                        },
                    )
                )
            break  # Only report once

        return recommendations

    def _check_concentrated_clusters(self, df: pd.DataFrame) -> list[Recommendation]:
        """Check for highly concentrated question clusters.

        Args:
            df: DataFrame with message data.

        Returns:
            List of recommendations for concentrated cluster issues.
        """
        recommendations = []

        # Look for concentration flag columns
        concentration_cols = [
            col for col in df.columns if "question_diversity_is_concentrated" in col
        ]

        for col in concentration_cols:
            if col not in df.columns:
                continue

            # Count samples in concentrated clusters
            concentrated_mask = df[col] == True  # noqa: E712
            concentrated_count = concentrated_mask.sum()

            # Get total user messages (non-null concentration values)
            total_users = df[col].notna().sum()
            if total_users == 0:
                continue

            concentrated_pct = (concentrated_count / total_users) * 100

            # Only warn if a significant portion (>50%) is in concentrated clusters
            # Lower percentages are common with DBSCAN and don't indicate a problem
            if concentrated_pct <= 50:
                continue

            severity = (
                RecommendationSeverity.HIGH
                if concentrated_pct > 80
                else RecommendationSeverity.MEDIUM
            )

            # Get indices of concentrated samples (limit to 20)
            concentrated_indices = df[concentrated_mask].index.tolist()[:20]

            # Get cluster size info for better context
            cluster_col = col.replace("_is_concentrated", "_cluster_size")
            if cluster_col in df.columns:
                cluster_sizes = df[cluster_col].dropna()
                if len(cluster_sizes) > 0:
                    max_cluster = int(cluster_sizes.max())
                    max_cluster_pct = (max_cluster / total_users) * 100
                else:
                    max_cluster_pct = concentrated_pct
            else:
                max_cluster_pct = concentrated_pct

            recommendations.append(
                Recommendation(
                    category=RecommendationCategory.WARNING,
                    severity=severity,
                    title="User questions concentrated in few clusters",
                    description=(
                        f"Found {concentrated_count} questions ({concentrated_pct:.1f}%) "
                        f"in large clusters (largest: {max_cluster_pct:.1f}%). "
                        f"This may indicate similar question structure or phrasing. "
                        f"Review whether this reflects low diversity or just common "
                        f"patterns. Note: DBSCAN can group similar structures."
                    ),
                    affected_samples=int(concentrated_count),
                    metric_name=col,
                    details={
                        "concentrated_count": int(concentrated_count),
                        "total_questions": int(total_users),
                        "concentrated_percentage": round(concentrated_pct, 2),
                        "largest_cluster_percentage": round(max_cluster_pct, 2),
                    },
                    sample_indices=concentrated_indices,
                )
            )
            break  # Only report once

        return recommendations

    def _check_placeholder_text(self, df: pd.DataFrame) -> list[Recommendation]:
        """Check for placeholder text in content.

        Args:
            df: DataFrame with message data.

        Returns:
            List of recommendations for placeholder text issues.
        """
        recommendations = []

        # Look for placeholder columns from content pattern analyzer
        placeholder_cols = [col for col in df.columns if "has_placeholder" in col]

        for col in placeholder_cols:
            if col not in df.columns:
                continue

            placeholder_mask = df[col] == True  # noqa: E712
            placeholder_count = placeholder_mask.sum()
            if placeholder_count > 0:
                placeholder_pct = (placeholder_count / len(df)) * 100

                severity = (
                    RecommendationSeverity.HIGH
                    if placeholder_pct > 5
                    else (
                        RecommendationSeverity.MEDIUM
                        if placeholder_pct > 1
                        else RecommendationSeverity.LOW
                    )
                )

                # Get indices of placeholder samples (limit to 20)
                placeholder_indices = df[placeholder_mask].index.tolist()[:20]

                recommendations.append(
                    Recommendation(
                        category=RecommendationCategory.WARNING,
                        severity=severity,
                        title="Placeholder text detected",
                        description=(
                            f"Found {placeholder_count} messages ({placeholder_pct:.1f}%) "
                            f"containing placeholder text (e.g., [Name], [Company Name], "
                            f"[Your...], [Insert...]). These indicate incomplete or "
                            f"template responses that should be filled in or removed "
                            f"before training."
                        ),
                        affected_samples=int(placeholder_count),
                        metric_name=col,
                        details={"placeholder_count": int(placeholder_count)},
                        sample_indices=placeholder_indices,
                    )
                )
            break  # Only report once

        return recommendations

    def _check_hallucinated_experiences(
        self, df: pd.DataFrame
    ) -> list[Recommendation]:
        """Check for AI hallucinated personal experiences.

        Args:
            df: DataFrame with message data.

        Returns:
            List of recommendations for hallucinated experience issues.
        """
        recommendations = []

        # Look for hallucinated experience columns from content pattern analyzer
        hallucination_cols = [
            col for col in df.columns if "has_hallucinated_experience" in col
        ]

        for col in hallucination_cols:
            if col not in df.columns:
                continue

            hallucination_mask = df[col] == True  # noqa: E712
            hallucination_count = hallucination_mask.sum()
            if hallucination_count > 0:
                hallucination_pct = (hallucination_count / len(df)) * 100

                severity = (
                    RecommendationSeverity.HIGH
                    if hallucination_pct > 2
                    else (
                        RecommendationSeverity.MEDIUM
                        if hallucination_pct > 0.5
                        else RecommendationSeverity.LOW
                    )
                )

                # Get indices of hallucination samples (limit to 20)
                hallucination_indices = df[hallucination_mask].index.tolist()[:20]

                recommendations.append(
                    Recommendation(
                        category=RecommendationCategory.WARNING,
                        severity=severity,
                        title="AI hallucinated experiences detected",
                        description=(
                            f"Found {hallucination_count} messages ({hallucination_pct:.1f}%) "
                            f"containing fabricated first-person experiences "
                            f"(e.g., 'When I was working as a project manager...'). "
                            f"Training on these may cause the model to generate similar "
                            f"hallucinations. Consider removing or rewriting these samples."
                        ),
                        affected_samples=int(hallucination_count),
                        metric_name=col,
                        details={"hallucination_count": int(hallucination_count)},
                        sample_indices=hallucination_indices,
                    )
                )
            break  # Only report once

        return recommendations

    def _check_nooutput_markers(self, df: pd.DataFrame) -> list[Recommendation]:
        """Check for nooutput/NA markers in content.

        Args:
            df: DataFrame with message data.

        Returns:
            List of recommendations for nooutput marker issues.
        """
        recommendations = []

        # Look for nooutput columns from content pattern analyzer
        nooutput_cols = [col for col in df.columns if "has_nooutput" in col]

        for col in nooutput_cols:
            if col not in df.columns:
                continue

            nooutput_mask = df[col] == True  # noqa: E712
            nooutput_count = nooutput_mask.sum()
            if nooutput_count > 0:
                nooutput_pct = (nooutput_count / len(df)) * 100

                severity = (
                    RecommendationSeverity.HIGH
                    if nooutput_pct > 1
                    else (
                        RecommendationSeverity.MEDIUM
                        if nooutput_pct > 0.1
                        else RecommendationSeverity.LOW
                    )
                )

                # Get indices of nooutput samples (limit to 20)
                nooutput_indices = df[nooutput_mask].index.tolist()[:20]

                recommendations.append(
                    Recommendation(
                        category=RecommendationCategory.WARNING,
                        severity=severity,
                        title="Nooutput/NA markers detected",
                        description=(
                            f"Found {nooutput_count} messages ({nooutput_pct:.1f}%) "
                            f"containing nooutput markers (e.g., <nooutput>, N/A, None). "
                            f"These are unusable training samples and should be removed."
                        ),
                        affected_samples=int(nooutput_count),
                        metric_name=col,
                        details={"nooutput_count": int(nooutput_count)},
                        sample_indices=nooutput_indices,
                    )
                )
            break  # Only report once

        return recommendations

    def _check_refusal_patterns(self, df: pd.DataFrame) -> list[Recommendation]:
        """Check for AI refusal patterns in content.

        Args:
            df: DataFrame with message data.

        Returns:
            List of recommendations for refusal pattern issues.
        """
        recommendations = []

        # Look for refusal columns from content pattern analyzer
        refusal_cols = [col for col in df.columns if "has_refusal" in col]

        for col in refusal_cols:
            if col not in df.columns:
                continue

            refusal_mask = df[col] == True  # noqa: E712
            refusal_count = refusal_mask.sum()
            if refusal_count > 0:
                refusal_pct = (refusal_count / len(df)) * 100

                severity = (
                    RecommendationSeverity.HIGH
                    if refusal_pct > 1
                    else (
                        RecommendationSeverity.MEDIUM
                        if refusal_pct > 0.1
                        else RecommendationSeverity.LOW
                    )
                )

                # Get indices of refusal samples (limit to 20)
                refusal_indices = df[refusal_mask].index.tolist()[:20]

                recommendations.append(
                    Recommendation(
                        category=RecommendationCategory.WARNING,
                        severity=severity,
                        title="AI refusal patterns detected",
                        description=(
                            f"Found {refusal_count} messages ({refusal_pct:.1f}%) "
                            f"containing AI refusal patterns (e.g., 'I cannot provide...', "
                            f"'I'm unable to help...'). These indicate the model refused "
                            f"to complete the task. Consider reviewing or removing these "
                            f"samples unless training for appropriate refusals."
                        ),
                        affected_samples=int(refusal_count),
                        metric_name=col,
                        details={"refusal_count": int(refusal_count)},
                        sample_indices=refusal_indices,
                    )
                )
            break  # Only report once

        return recommendations

    def _check_ifd_scores(self, df: pd.DataFrame) -> list[Recommendation]:
        """Check for IFD (Instruction-Following Difficulty) score issues.

        IFD measures how much the instruction helps the model predict the response.
        IFD = PPL(response | no instruction) / PPL(response | with instruction)

        - IFD > 1: Instruction helps the model (valuable sample)
        - IFD ≈ 1: Instruction provides minimal guidance
        - IFD < 1: Instruction may be confusing or misaligned (potentially problematic)

        Args:
            df: DataFrame with message data.

        Returns:
            List of recommendations for IFD score issues.
        """
        recommendations = []

        # Look for IFD score columns
        ifd_cols = [col for col in df.columns if "ifd_score" in col]

        for col in ifd_cols:
            if col not in df.columns:
                continue

            scores = df[col].dropna()
            if len(scores) == 0:
                continue

            # Check for samples where IFD < 1.0 (instruction doesn't help)
            problematic_mask = scores < 1.0
            problematic_count = problematic_mask.sum()

            if problematic_count > 0:
                problematic_pct = (problematic_count / len(scores)) * 100
                avg_score = scores.mean()
                median_score = scores.median()

                severity = (
                    RecommendationSeverity.HIGH
                    if problematic_pct > 20
                    else (
                        RecommendationSeverity.MEDIUM
                        if problematic_pct > 10
                        else RecommendationSeverity.LOW
                    )
                )

                # Get indices of problematic samples (limit to 20)
                problematic_indices = scores[problematic_mask].index.tolist()[:20]

                recommendations.append(
                    Recommendation(
                        category=RecommendationCategory.WARNING,
                        severity=severity,
                        title="Low IFD scores detected (instruction not helping)",
                        description=(
                            f"Found {problematic_count} samples "
                            f"({problematic_pct:.1f}%) with IFD scores below 1.0, "
                            f"meaning the instruction does not help the model predict "
                            f"the response. These may indicate misaligned "
                            f"instruction-response pairs or confusing instructions. "
                            f"Consider reviewing or removing these samples."
                        ),
                        affected_samples=int(problematic_count),
                        metric_name=col,
                        threshold=1.0,
                        details={
                            "problematic_count": int(problematic_count),
                            "average_ifd": round(avg_score, 3),
                            "median_ifd": round(median_score, 3),
                            "min_ifd": round(scores.min(), 3),
                            "max_ifd": round(scores.max(), 3),
                        },
                        sample_indices=problematic_indices,
                    )
                )

            # Also provide insight about overall IFD distribution
            if len(scores) >= 10:
                avg_score = scores.mean()
                median_score = scores.median()

                # High IFD samples are valuable for training
                high_ifd_mask = scores > 2.0
                high_ifd_count = high_ifd_mask.sum()
                high_ifd_pct = (high_ifd_count / len(scores)) * 100

                # Very high IFD may indicate trivial responses
                very_high_ifd_mask = scores > 100.0
                very_high_ifd_count = very_high_ifd_mask.sum()

                if avg_score > 2.0:
                    # Good dataset - instructions are helpful
                    recommendations.append(
                        Recommendation(
                            category=RecommendationCategory.INSIGHT,
                            severity=RecommendationSeverity.LOW,
                            title="Good instruction quality (high IFD scores)",
                            description=(
                                f"Average IFD score is {avg_score:.2f}, indicating "
                                f"that instructions effectively guide the model. "
                                f"{high_ifd_count} samples ({high_ifd_pct:.1f}%) "
                                f"have IFD > 2.0, making them high-value examples."
                            ),
                            affected_samples=int(high_ifd_count),
                            metric_name=col,
                            details={
                                "average_ifd": round(avg_score, 3),
                                "median_ifd": round(median_score, 3),
                                "high_ifd_count": int(high_ifd_count),
                                "high_ifd_percentage": round(high_ifd_pct, 2),
                            },
                        )
                    )
                elif avg_score < 1.5:
                    # Concerning - instructions provide minimal value
                    recommendations.append(
                        Recommendation(
                            category=RecommendationCategory.WARNING,
                            severity=RecommendationSeverity.MEDIUM,
                            title="Low average IFD (minimal instruction value)",
                            description=(
                                f"Average IFD score is {avg_score:.2f}, suggesting "
                                f"that instructions provide limited guidance. "
                                f"Consider reviewing instruction quality or ensuring "
                                f"instruction-response pairs are well-aligned."
                            ),
                            affected_samples=len(scores),
                            metric_name=col,
                            details={
                                "average_ifd": round(avg_score, 3),
                                "median_ifd": round(median_score, 3),
                            },
                        )
                    )

                # Warn about very high IFD values
                if very_high_ifd_count > 0:
                    very_high_pct = (very_high_ifd_count / len(scores)) * 100
                    very_high_indices = scores[very_high_ifd_mask].index.tolist()[:20]

                    recommendations.append(
                        Recommendation(
                            category=RecommendationCategory.INSIGHT,
                            severity=RecommendationSeverity.LOW,
                            title="Very high IFD scores detected",
                            description=(
                                f"Found {very_high_ifd_count} samples "
                                f"({very_high_pct:.1f}%) with IFD > 100. While high "
                                f"IFD generally indicates valuable samples, extremely "
                                f"high values may indicate trivial responses or data "
                                f"issues. Consider reviewing these samples."
                            ),
                            affected_samples=int(very_high_ifd_count),
                            metric_name=col,
                            details={
                                "very_high_ifd_count": int(very_high_ifd_count),
                                "max_ifd": round(scores.max(), 3),
                            },
                            sample_indices=very_high_indices,
                        )
                    )

            break  # Only report once

        return recommendations

    def _check_task_category_distribution(
        self, df: pd.DataFrame
    ) -> list[Recommendation]:
        """Check for task category distribution issues.

        Args:
            df: DataFrame with message data.

        Returns:
            List of recommendations for task category issues.
        """
        recommendations = []

        # Look for task category columns
        category_cols = [col for col in df.columns if "task_category_category" in col]

        for col in category_cols:
            if col not in df.columns:
                continue

            # Get category distribution (excluding None)
            categories = df[col].dropna()
            if len(categories) < 10:
                continue

            category_counts = categories.value_counts()
            total = len(categories)

            # Check for imbalanced distribution
            largest_category = category_counts.iloc[0]
            largest_category_name = category_counts.index[0]
            largest_pct = (largest_category / total) * 100

            # Check if dominated by single category
            if largest_pct > 50:
                severity = (
                    RecommendationSeverity.HIGH
                    if largest_pct > 70
                    else RecommendationSeverity.MEDIUM
                )

                recommendations.append(
                    Recommendation(
                        category=RecommendationCategory.WARNING,
                        severity=severity,
                        title=f"Task distribution dominated by '{largest_category_name}'",
                        description=(
                            f"The '{largest_category_name}' category accounts for "
                            f"{largest_pct:.1f}% of all tasks. This imbalance may "
                            f"cause the model to underperform on less common tasks. "
                            f"Consider adding more diverse task types."
                        ),
                        affected_samples=int(largest_category),
                        metric_name=col,
                        details={
                            "dominant_category": largest_category_name,
                            "dominant_percentage": round(largest_pct, 2),
                            "category_distribution": {
                                str(k): int(v)
                                for k, v in category_counts.head(10).items()
                            },
                        },
                    )
                )

            # Check for missing important categories
            important_categories = {"math", "coding", "reasoning", "creative_writing"}
            missing = important_categories - set(category_counts.index)
            if missing and len(category_counts) >= 3:
                recommendations.append(
                    Recommendation(
                        category=RecommendationCategory.INSIGHT,
                        severity=RecommendationSeverity.LOW,
                        title="Some common task categories are missing",
                        description=(
                            f"The following task categories are not represented: "
                            f"{', '.join(missing)}. Consider whether your dataset "
                            f"needs coverage in these areas."
                        ),
                        affected_samples=0,
                        metric_name=col,
                        details={"missing_categories": list(missing)},
                    )
                )
            break

        return recommendations

    def _check_instruct_reward_scores(
        self, df: pd.DataFrame
    ) -> list[Recommendation]:
        """Check for response quality issues based on instruct reward scores.

        Args:
            df: DataFrame with message data.

        Returns:
            List of recommendations for reward score issues.
        """
        recommendations = []

        # Look for reward score columns
        reward_cols = [col for col in df.columns if "instruct_reward_score" in col]

        for col in reward_cols:
            if col not in df.columns:
                continue

            scores = df[col].dropna()
            if len(scores) < 10:
                continue

            avg_score = scores.mean()
            low_quality_mask = scores < 2.5  # Below 2.5 on 0-5 scale
            low_quality_count = low_quality_mask.sum()
            low_quality_pct = (low_quality_count / len(scores)) * 100

            if low_quality_count > 0 and low_quality_pct > 10:
                severity = (
                    RecommendationSeverity.HIGH
                    if low_quality_pct > 30
                    else (
                        RecommendationSeverity.MEDIUM
                        if low_quality_pct > 20
                        else RecommendationSeverity.LOW
                    )
                )

                low_quality_indices = scores[low_quality_mask].index.tolist()[:20]

                recommendations.append(
                    Recommendation(
                        category=RecommendationCategory.WARNING,
                        severity=severity,
                        title="Low-quality responses detected (instruct reward)",
                        description=(
                            f"Found {low_quality_count} responses ({low_quality_pct:.1f}%) "
                            f"with reward scores below 2.5 (out of 5). Average score: "
                            f"{avg_score:.2f}. Consider filtering or improving these "
                            f"low-quality responses before training."
                        ),
                        affected_samples=int(low_quality_count),
                        metric_name=col,
                        threshold=2.5,
                        details={
                            "low_quality_count": int(low_quality_count),
                            "average_score": round(avg_score, 2),
                            "min_score": round(scores.min(), 2),
                        },
                        sample_indices=low_quality_indices,
                    )
                )
            break

        return recommendations

    def _check_input_quality_issues(self, df: pd.DataFrame) -> list[Recommendation]:
        """Check for input quality issues.

        Args:
            df: DataFrame with message data.

        Returns:
            List of recommendations for input quality issues.
        """
        recommendations = []

        # Look for input quality tier columns
        tier_cols = [col for col in df.columns if "input_quality_tier" in col]

        for col in tier_cols:
            if col not in df.columns:
                continue

            tiers = df[col].dropna()
            if len(tiers) < 10:
                continue

            tier_counts = tiers.value_counts()
            total = len(tiers)

            # Count poor quality inputs
            poor_count = tier_counts.get("poor", 0) + tier_counts.get("very_poor", 0)
            poor_pct = (poor_count / total) * 100

            if poor_count > 0 and poor_pct > 10:
                severity = (
                    RecommendationSeverity.HIGH
                    if poor_pct > 25
                    else (
                        RecommendationSeverity.MEDIUM
                        if poor_pct > 15
                        else RecommendationSeverity.LOW
                    )
                )

                poor_mask = df[col].isin(["poor", "very_poor"])
                poor_indices = df[poor_mask].index.tolist()[:20]

                recommendations.append(
                    Recommendation(
                        category=RecommendationCategory.WARNING,
                        severity=severity,
                        title="Poor quality inputs detected",
                        description=(
                            f"Found {poor_count} inputs ({poor_pct:.1f}%) rated as "
                            f"'poor' or 'very_poor'. These may be ambiguous, incomplete, "
                            f"or unanswerable instructions that could confuse the model. "
                            f"Consider improving or filtering these samples."
                        ),
                        affected_samples=int(poor_count),
                        metric_name=col,
                        details={
                            "poor_count": int(poor_count),
                            "tier_distribution": {str(k): int(v) for k, v in tier_counts.items()},
                        },
                        sample_indices=poor_indices,
                    )
                )
            break

        return recommendations

    def _check_conversation_structure(self, df: pd.DataFrame) -> list[Recommendation]:
        """Check for conversation structure issues.

        Args:
            df: DataFrame with message data.

        Returns:
            List of recommendations for conversation structure issues.
        """
        recommendations = []

        # Look for conversation structure columns
        single_turn_cols = [
            col for col in df.columns if "conversation_structure_is_single_turn" in col
        ]

        for col in single_turn_cols:
            if col not in df.columns:
                continue

            single_turn = df[col].dropna()
            if len(single_turn) < 10:
                continue

            single_turn_count = single_turn.sum()
            total = len(single_turn)
            single_turn_pct = (single_turn_count / total) * 100

            # High single-turn ratio insight
            if single_turn_pct > 90:
                recommendations.append(
                    Recommendation(
                        category=RecommendationCategory.INSIGHT,
                        severity=RecommendationSeverity.LOW,
                        title="Dataset is predominantly single-turn",
                        description=(
                            f"{single_turn_pct:.1f}% of conversations are single-turn. "
                            f"This is similar to Tulu-3 (95% single-turn). If training "
                            f"for multi-turn dialogue capabilities, consider adding "
                            f"more multi-turn conversations."
                        ),
                        affected_samples=int(single_turn_count),
                        metric_name=col,
                        details={"single_turn_percentage": round(single_turn_pct, 2)},
                    )
                )
            break

        return recommendations

    def _check_safety_issues(self, df: pd.DataFrame) -> list[Recommendation]:
        """Check for safety issues in content.

        Args:
            df: DataFrame with message data.

        Returns:
            List of recommendations for safety issues.
        """
        recommendations = []

        # Look for safety columns
        safety_cols = [col for col in df.columns if "safety_is_safe" in col]

        for col in safety_cols:
            if col not in df.columns:
                continue

            safety = df[col].dropna()
            if len(safety) == 0:
                continue

            unsafe_mask = safety == False  # noqa: E712
            unsafe_count = unsafe_mask.sum()
            unsafe_pct = (unsafe_count / len(safety)) * 100

            if unsafe_count > 0:
                severity = (
                    RecommendationSeverity.HIGH
                    if unsafe_pct > 5
                    else (
                        RecommendationSeverity.MEDIUM
                        if unsafe_pct > 1
                        else RecommendationSeverity.LOW
                    )
                )

                unsafe_indices = safety[unsafe_mask].index.tolist()[:20]

                # Get category breakdown if available
                category_col = col.replace("_is_safe", "_categories")
                category_detail = ""
                if category_col in df.columns:
                    all_categories = df[category_col].dropna()
                    all_categories = all_categories[all_categories != ""]
                    if len(all_categories) > 0:
                        cat_counts = Counter(
                            c for cats in all_categories for c in cats.split(",") if c
                        )
                        if cat_counts:
                            category_detail = (
                                f" Categories: "
                                f"{', '.join(f'{c} ({n})' for c, n in cat_counts.most_common(5))}."
                            )

                recommendations.append(
                    Recommendation(
                        category=RecommendationCategory.WARNING,
                        severity=severity,
                        title="Potentially unsafe content detected",
                        description=(
                            f"Found {unsafe_count} samples ({unsafe_pct:.1f}%) flagged "
                            f"as potentially unsafe.{category_detail} Review these "
                            f"samples before training to ensure appropriate content."
                        ),
                        affected_samples=int(unsafe_count),
                        metric_name=col,
                        details={"unsafe_count": int(unsafe_count)},
                        sample_indices=unsafe_indices,
                    )
                )
            break

        return recommendations

    def _check_difficulty_distribution(self, df: pd.DataFrame) -> list[Recommendation]:
        """Check for difficulty distribution issues.

        Args:
            df: DataFrame with message data.

        Returns:
            List of recommendations for difficulty distribution issues.
        """
        recommendations = []

        # Look for difficulty tier columns
        tier_cols = [col for col in df.columns if "difficulty_tier" in col]

        for col in tier_cols:
            if col not in df.columns:
                continue

            tiers = df[col].dropna()
            if len(tiers) < 10:
                continue

            tier_counts = tiers.value_counts()
            total = len(tiers)

            # Check for skewed distribution
            easy_count = tier_counts.get("easy", 0)
            easy_pct = (easy_count / total) * 100

            hard_count = tier_counts.get("hard", 0) + tier_counts.get("expert", 0)
            hard_pct = (hard_count / total) * 100

            if easy_pct > 70:
                recommendations.append(
                    Recommendation(
                        category=RecommendationCategory.INSIGHT,
                        severity=RecommendationSeverity.LOW,
                        title="Dataset skewed toward easy samples",
                        description=(
                            f"{easy_pct:.1f}% of samples are rated as 'easy'. "
                            f"Consider adding more challenging samples for better "
                            f"model capability development."
                        ),
                        affected_samples=int(easy_count),
                        metric_name=col,
                        details={
                            "tier_distribution": {
                                str(k): int(v) for k, v in tier_counts.items()
                            }
                        },
                    )
                )
            elif hard_pct > 70:
                recommendations.append(
                    Recommendation(
                        category=RecommendationCategory.INSIGHT,
                        severity=RecommendationSeverity.MEDIUM,
                        title="Dataset skewed toward difficult samples",
                        description=(
                            f"{hard_pct:.1f}% of samples are rated as 'hard' or 'expert'. "
                            f"Consider adding easier samples for curriculum learning."
                        ),
                        affected_samples=int(hard_count),
                        metric_name=col,
                        details={
                            "tier_distribution": {
                                str(k): int(v) for k, v in tier_counts.items()
                            }
                        },
                    )
                )
            break

        return recommendations

    def _check_response_completeness_analyzer(
        self, df: pd.DataFrame
    ) -> list[Recommendation]:
        """Check for response completeness issues from the new analyzer.

        Args:
            df: DataFrame with message data.

        Returns:
            List of recommendations for response completeness issues.
        """
        recommendations = []

        # Look for completeness columns from response_completeness analyzer
        complete_cols = [
            col for col in df.columns if "response_completeness_is_complete" in col
        ]

        for col in complete_cols:
            if col not in df.columns:
                continue

            completeness = df[col].dropna()
            if len(completeness) == 0:
                continue

            incomplete_mask = completeness == False  # noqa: E712
            incomplete_count = incomplete_mask.sum()
            incomplete_pct = (incomplete_count / len(completeness)) * 100

            if incomplete_count > 0 and incomplete_pct > 5:
                severity = (
                    RecommendationSeverity.HIGH
                    if incomplete_pct > 20
                    else (
                        RecommendationSeverity.MEDIUM
                        if incomplete_pct > 10
                        else RecommendationSeverity.LOW
                    )
                )

                incomplete_indices = completeness[incomplete_mask].index.tolist()[:20]

                # Get truncation type breakdown if available
                truncation_col = col.replace("_is_complete", "_truncation_type")
                truncation_detail = ""
                if truncation_col in df.columns:
                    incomplete_rows = df.loc[incomplete_mask]
                    truncation_types = incomplete_rows[truncation_col].dropna()
                    truncation_types = truncation_types[truncation_types != ""]
                    if len(truncation_types) > 0:
                        type_counts = truncation_types.value_counts()
                        truncation_detail = (
                            f" Truncation types: "
                            f"{', '.join(f'{t} ({c})' for t, c in type_counts.items())}."
                        )

                recommendations.append(
                    Recommendation(
                        category=RecommendationCategory.WARNING,
                        severity=severity,
                        title="Incomplete responses detected",
                        description=(
                            f"Found {incomplete_count} responses ({incomplete_pct:.1f}%) "
                            f"that appear incomplete or truncated.{truncation_detail} "
                            f"Training on incomplete responses may cause the model to "
                            f"generate truncated outputs."
                        ),
                        affected_samples=int(incomplete_count),
                        metric_name=col,
                        details={"incomplete_count": int(incomplete_count)},
                        sample_indices=incomplete_indices,
                    )
                )
            break

        return recommendations
