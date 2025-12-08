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

"""Observations engine for generating summary insights from analysis results."""

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import pandas as pd


class ObservationCategory(str, Enum):
    """Category of observation."""

    DISTRIBUTION = "distribution"  # Distribution characteristics
    COMPOSITION = "composition"  # Dataset composition
    CONTENT = "content"  # Content characteristics
    QUALITY = "quality"  # Quality metrics
    STRUCTURE = "structure"  # Structural characteristics


@dataclass
class Observation:
    """A single observation summarizing a finding from analysis.

    Attributes:
        category: The type of observation.
        title: Short title summarizing the observation.
        description: Detailed description of the finding.
        metric_name: Optional name of the metric this observation relates to.
        details: Optional additional details as a dictionary.
    """

    category: ObservationCategory
    title: str
    description: str
    metric_name: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert observation to a dictionary.

        Returns:
            Dictionary representation of the observation.
        """
        return {
            "category": self.category.value,
            "title": self.title,
            "description": self.description,
            "metric_name": self.metric_name,
            "details": self.details,
        }


class ObservationsEngine:
    """Engine for generating observations from analysis results.

    This engine analyzes the results from dataset analysis and generates
    summary observations about the main findings.
    """

    def generate_observations(
        self,
        message_df: pd.DataFrame,
        conversation_df: pd.DataFrame,
        analysis_summary: dict[str, Any],
    ) -> list[Observation]:
        """Generate all observations from analysis results.

        Args:
            message_df: DataFrame with message-level analysis results.
            conversation_df: DataFrame with conversation-level analysis results.
            analysis_summary: Summary statistics from the analysis.

        Returns:
            List of Observation objects.
        """
        observations = []

        # Dataset composition observations
        observations.extend(self._observe_dataset_composition(analysis_summary))
        observations.extend(self._observe_role_distribution(message_df))

        # Content observations
        observations.extend(self._observe_length_statistics(message_df, analysis_summary))
        observations.extend(self._observe_format_content(message_df))
        observations.extend(self._observe_diversity_metrics(message_df, analysis_summary))

        # Language observations
        observations.extend(self._observe_language_distribution(message_df))

        # Quality observations
        observations.extend(self._observe_quality_metrics(message_df))

        # Structure observations
        observations.extend(self._observe_conversation_structure(conversation_df))

        # Distribution characteristics from summary
        observations.extend(self._observe_distribution_characteristics(analysis_summary))

        return observations

    def _observe_dataset_composition(
        self, analysis_summary: dict[str, Any]
    ) -> list[Observation]:
        """Observe dataset composition statistics.

        Args:
            analysis_summary: Summary statistics from the analysis.

        Returns:
            List of observations about dataset composition.
        """
        observations = []

        overview = analysis_summary.get("dataset_overview", {})
        if not overview:
            return observations

        total_conversations = overview.get("total_conversations", 0)
        conversations_analyzed = overview.get("conversations_analyzed", 0)
        total_messages = overview.get("total_messages", 0)
        analyzers_used = overview.get("analyzers_used", [])

        if total_conversations > 0 and total_messages > 0:
            avg_messages = total_messages / conversations_analyzed
            observations.append(
                Observation(
                    category=ObservationCategory.COMPOSITION,
                    title="Dataset size and structure",
                    description=(
                        f"Dataset contains {conversations_analyzed:,} conversations "
                        f"with {total_messages:,} total messages "
                        f"(~{avg_messages:.1f} messages per conversation on average)."
                    ),
                    details={
                        "total_conversations": total_conversations,
                        "conversations_analyzed": conversations_analyzed,
                        "total_messages": total_messages,
                        "avg_messages_per_conversation": round(avg_messages, 2),
                        "analyzers_used": analyzers_used,
                    },
                )
            )

        return observations

    def _observe_role_distribution(self, df: pd.DataFrame) -> list[Observation]:
        """Observe role distribution in messages.

        Args:
            df: DataFrame with message data.

        Returns:
            List of observations about role distribution.
        """
        observations = []

        if "role" not in df.columns or len(df) == 0:
            return observations

        role_counts = df["role"].value_counts()
        total = len(df)

        # Build role distribution description
        role_parts = []
        for role, count in role_counts.items():
            pct = (count / total) * 100
            role_parts.append(f"{role}: {pct:.1f}%")

        if role_parts:
            observations.append(
                Observation(
                    category=ObservationCategory.COMPOSITION,
                    title="Message role distribution",
                    description=f"Role distribution: {', '.join(role_parts)}.",
                    metric_name="role",
                    details={
                        "role_distribution": {
                            str(r): {"count": int(c), "percentage": round(c / total * 100, 1)}
                            for r, c in role_counts.items()
                        }
                    },
                )
            )

        return observations

    def _observe_length_statistics(
        self, df: pd.DataFrame, analysis_summary: dict[str, Any]
    ) -> list[Observation]:
        """Observe length statistics.

        Args:
            df: DataFrame with message data.
            analysis_summary: Summary statistics from the analysis.

        Returns:
            List of observations about content length.
        """
        observations = []

        # Look for token count columns
        token_cols = [col for col in df.columns if "token_count" in col]

        for col in token_cols:
            if col not in df.columns:
                continue

            series = df[col].dropna()
            if len(series) == 0:
                continue

            mean_val = series.mean()
            median_val = series.median()
            min_val = series.min()
            max_val = series.max()

            observations.append(
                Observation(
                    category=ObservationCategory.CONTENT,
                    title="Token length statistics",
                    description=(
                        f"Messages average {mean_val:.0f} tokens "
                        f"(median: {median_val:.0f}, range: {min_val:.0f}-{max_val:.0f})."
                    ),
                    metric_name=col,
                    details={
                        "mean": round(mean_val, 2),
                        "median": round(median_val, 2),
                        "min": int(min_val),
                        "max": int(max_val),
                    },
                )
            )
            break  # Only report first token count column

        return observations

    def _observe_format_content(self, df: pd.DataFrame) -> list[Observation]:
        """Observe format content characteristics.

        Args:
            df: DataFrame with message data.

        Returns:
            List of observations about content formatting.
        """
        observations = []

        # Check for code blocks
        code_block_cols = [col for col in df.columns if "has_code_blocks" in col]
        for col in code_block_cols:
            if col not in df.columns:
                continue

            has_code = df[col] == True  # noqa: E712
            code_count = has_code.sum()
            if code_count > 0:
                code_pct = (code_count / len(df)) * 100
                observations.append(
                    Observation(
                        category=ObservationCategory.CONTENT,
                        title="Code content",
                        description=(
                            f"{code_count:,} messages ({code_pct:.1f}%) contain code blocks."
                        ),
                        metric_name=col,
                        details={"count": int(code_count), "percentage": round(code_pct, 1)},
                    )
                )
            break

        # Check for markdown
        markdown_cols = [col for col in df.columns if "has_markdown" in col]
        for col in markdown_cols:
            if col not in df.columns:
                continue

            has_markdown = df[col] == True  # noqa: E712
            markdown_count = has_markdown.sum()
            if markdown_count > 0:
                markdown_pct = (markdown_count / len(df)) * 100
                observations.append(
                    Observation(
                        category=ObservationCategory.CONTENT,
                        title="Markdown formatting",
                        description=(
                            f"{markdown_count:,} messages ({markdown_pct:.1f}%) "
                            f"contain markdown formatting."
                        ),
                        metric_name=col,
                        details={
                            "count": int(markdown_count),
                            "percentage": round(markdown_pct, 1),
                        },
                    )
                )
            break

        # Check for JSON content
        json_cols = [col for col in df.columns if "has_json" in col]
        for col in json_cols:
            if col not in df.columns:
                continue

            has_json = df[col] == True  # noqa: E712
            json_count = has_json.sum()
            if json_count > 0:
                json_pct = (json_count / len(df)) * 100
                observations.append(
                    Observation(
                        category=ObservationCategory.CONTENT,
                        title="JSON content",
                        description=(
                            f"{json_count:,} messages ({json_pct:.1f}%) contain JSON content."
                        ),
                        metric_name=col,
                        details={"count": int(json_count), "percentage": round(json_pct, 1)},
                    )
                )
            break

        return observations

    def _observe_diversity_metrics(
        self, df: pd.DataFrame, analysis_summary: dict[str, Any]
    ) -> list[Observation]:
        """Observe diversity metrics.

        Args:
            df: DataFrame with message data.
            analysis_summary: Summary statistics from the analysis.

        Returns:
            List of observations about diversity.
        """
        observations = []

        # Look for unique words ratio
        diversity_cols = [col for col in df.columns if "unique_words_ratio" in col]

        for col in diversity_cols:
            if col not in df.columns:
                continue

            series = df[col].dropna()
            if len(series) == 0:
                continue

            mean_diversity = series.mean()

            # Interpret diversity level
            if mean_diversity >= 0.8:
                level = "high"
                interpretation = "Most content uses varied vocabulary."
            elif mean_diversity >= 0.6:
                level = "moderate"
                interpretation = "Content has reasonable vocabulary diversity."
            else:
                level = "low"
                interpretation = "Content may be repetitive or formulaic."

            observations.append(
                Observation(
                    category=ObservationCategory.QUALITY,
                    title="Vocabulary diversity",
                    description=(
                        f"Average vocabulary diversity is {mean_diversity:.2f} ({level}). "
                        f"{interpretation}"
                    ),
                    metric_name=col,
                    details={
                        "average_diversity": round(mean_diversity, 3),
                        "level": level,
                    },
                )
            )
            break

        return observations

    def _observe_language_distribution(self, df: pd.DataFrame) -> list[Observation]:
        """Observe language distribution in the dataset.

        Args:
            df: DataFrame with message data.

        Returns:
            List of observations about language distribution.
        """
        observations = []

        # Look for language detection columns
        lang_cols = [col for col in df.columns if "detected_language" in col]

        for col in lang_cols:
            if col not in df.columns:
                continue

            languages = df[col].dropna()
            languages = languages[languages != ""]

            if len(languages) < 10:
                continue

            lang_counts = Counter(languages)
            total = sum(lang_counts.values())

            # Get top languages
            top_langs = lang_counts.most_common(3)
            dominant_lang, dominant_count = top_langs[0]
            dominant_pct = (dominant_count / total) * 100

            if dominant_pct >= 95:
                description = (
                    f"Dataset is predominantly {dominant_lang} ({dominant_pct:.1f}%)."
                )
            elif dominant_pct >= 80:
                other_langs = ", ".join(f"{lang} ({(c/total)*100:.1f}%)" for lang, c in top_langs[1:])
                description = (
                    f"Primary language is {dominant_lang} ({dominant_pct:.1f}%), "
                    f"with some {other_langs}."
                )
            else:
                langs_str = ", ".join(f"{lang} ({(c/total)*100:.1f}%)" for lang, c in top_langs)
                description = f"Multilingual dataset: {langs_str}."

            observations.append(
                Observation(
                    category=ObservationCategory.COMPOSITION,
                    title="Language distribution",
                    description=description,
                    metric_name=col,
                    details={
                        "dominant_language": dominant_lang,
                        "dominant_percentage": round(dominant_pct, 1),
                        "language_distribution": {
                            lang: round((c / total) * 100, 1) for lang, c in top_langs
                        },
                    },
                )
            )
            break

        return observations

    def _observe_quality_metrics(self, df: pd.DataFrame) -> list[Observation]:
        """Observe quality-related metrics.

        Args:
            df: DataFrame with message data.

        Returns:
            List of observations about quality.
        """
        observations = []

        # Check for PII
        pii_cols = [col for col in df.columns if "has_pii" in col]
        for col in pii_cols:
            if col not in df.columns:
                continue

            has_pii = df[col] == True  # noqa: E712
            pii_count = has_pii.sum()
            pii_pct = (pii_count / len(df)) * 100 if len(df) > 0 else 0

            if pii_count == 0:
                observations.append(
                    Observation(
                        category=ObservationCategory.QUALITY,
                        title="PII detection",
                        description="No PII (personally identifiable information) detected.",
                        metric_name=col,
                        details={"pii_count": 0, "percentage": 0},
                    )
                )
            else:
                observations.append(
                    Observation(
                        category=ObservationCategory.QUALITY,
                        title="PII detection",
                        description=(
                            f"PII detected in {pii_count:,} messages ({pii_pct:.1f}%)."
                        ),
                        metric_name=col,
                        details={"pii_count": int(pii_count), "percentage": round(pii_pct, 1)},
                    )
                )
            break

        # Check for duplicates
        if "text_content" in df.columns and len(df) > 0:
            # Filter out system messages for duplicate detection
            if "role" in df.columns:
                non_system_df = df[df["role"] != "system"]
            else:
                non_system_df = df

            if len(non_system_df) > 0:
                duplicates = non_system_df["text_content"].duplicated(keep=False)
                duplicate_count = duplicates.sum()
                duplicate_pct = (duplicate_count / len(non_system_df)) * 100

                if duplicate_count == 0:
                    observations.append(
                        Observation(
                            category=ObservationCategory.QUALITY,
                            title="Duplicate content",
                            description="No duplicate messages detected (excluding system prompts).",
                            metric_name="text_content",
                            details={"duplicate_count": 0, "percentage": 0},
                        )
                    )
                else:
                    unique_duplicated = non_system_df[duplicates]["text_content"].nunique()
                    observations.append(
                        Observation(
                            category=ObservationCategory.QUALITY,
                            title="Duplicate content",
                            description=(
                                f"{duplicate_count:,} messages ({duplicate_pct:.1f}%) are "
                                f"duplicates ({unique_duplicated:,} unique repeated texts)."
                            ),
                            metric_name="text_content",
                            details={
                                "duplicate_count": int(duplicate_count),
                                "percentage": round(duplicate_pct, 1),
                                "unique_duplicated_texts": int(unique_duplicated),
                            },
                        )
                    )

        return observations

    def _observe_conversation_structure(
        self, df: pd.DataFrame
    ) -> list[Observation]:
        """Observe conversation structure characteristics.

        Args:
            df: DataFrame with conversation-level data.

        Returns:
            List of observations about conversation structure.
        """
        observations = []

        if "num_messages" not in df.columns or len(df) == 0:
            return observations

        num_messages = df["num_messages"]

        # Analyze turn distribution
        single_turn = (num_messages <= 2).sum()
        multi_turn = (num_messages > 2).sum()
        single_turn_pct = (single_turn / len(df)) * 100

        if single_turn_pct >= 90:
            description = (
                f"Dataset is predominantly single-turn ({single_turn_pct:.1f}% have "
                f"2 or fewer messages)."
            )
        elif single_turn_pct >= 50:
            description = (
                f"Mixed single/multi-turn: {single_turn:,} single-turn ({single_turn_pct:.1f}%), "
                f"{multi_turn:,} multi-turn ({100 - single_turn_pct:.1f}%)."
            )
        else:
            description = (
                f"Dataset is predominantly multi-turn ({100 - single_turn_pct:.1f}% have "
                f"more than 2 messages)."
            )

        observations.append(
            Observation(
                category=ObservationCategory.STRUCTURE,
                title="Conversation structure",
                description=description,
                metric_name="num_messages",
                details={
                    "single_turn_count": int(single_turn),
                    "multi_turn_count": int(multi_turn),
                    "single_turn_percentage": round(single_turn_pct, 1),
                    "mean_messages": round(num_messages.mean(), 2),
                    "max_messages": int(num_messages.max()),
                },
            )
        )

        return observations

    def _observe_distribution_characteristics(
        self, analysis_summary: dict[str, Any]
    ) -> list[Observation]:
        """Observe distribution characteristics from the summary.

        Args:
            analysis_summary: Summary statistics from the analysis.

        Returns:
            List of observations about distribution characteristics.
        """
        observations = []

        # Check message-level summary for distribution info
        msg_summary = analysis_summary.get("message_level_summary", {})

        for analyzer_name, metrics in msg_summary.items():
            for metric_name, stats in metrics.items():
                if not isinstance(stats, dict):
                    continue

                # Check if distribution analysis is available
                dist_type = stats.get("distribution_type")
                if dist_type and dist_type != "unimodal":
                    num_modes = stats.get("num_modes", 0)
                    if num_modes >= 2:
                        # Get mode statistics if available
                        mode_stats = stats.get("mode_statistics", [])
                        mode_description = ""
                        if mode_stats:
                            mode_parts = []
                            for mode in mode_stats:
                                mode_mean = mode.get("mean", 0)
                                mode_pct = mode.get("weight", 0) * 100
                                mode_parts.append(f"~{mode_mean:.0f} ({mode_pct:.0f}%)")
                            mode_description = f" Modes centered at: {', '.join(mode_parts)}."

                        clean_name = metric_name.replace("text_content_", "").replace("_", " ")
                        observations.append(
                            Observation(
                                category=ObservationCategory.DISTRIBUTION,
                                title=f"Multimodal distribution in {clean_name}",
                                description=(
                                    f"The {clean_name} distribution has {num_modes} distinct modes, "
                                    f"suggesting different types of content.{mode_description}"
                                ),
                                metric_name=metric_name,
                                details={
                                    "distribution_type": dist_type,
                                    "num_modes": num_modes,
                                    "mode_statistics": mode_stats,
                                },
                            )
                        )

        return observations
