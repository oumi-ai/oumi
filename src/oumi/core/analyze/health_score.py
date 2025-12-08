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

"""Dataset Health Score computation for comprehensive quality assessment."""

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from oumi.core.analyze.recommendations import Recommendation, RecommendationSeverity
from oumi.utils.analysis_utils import DistributionType, detect_distribution_type


@dataclass
class HealthScoreComponent:
    """A single component of the health score.

    Attributes:
        name: Human-readable name of the component.
        score: Score for this component (0-100).
        weight: Weight of this component in the overall score.
        description: Description of what this component measures.
        details: Additional details about the component score.
    """

    name: str
    score: float
    weight: float
    description: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetHealthScore:
    """Comprehensive dataset health score.

    The health score provides a single composite metric (0-100) that summarizes
    the overall quality of a dataset for training. Higher scores indicate
    better quality datasets.

    Attributes:
        overall: Overall health score (0-100).
        grade: Letter grade (A, B, C, D, F).
        components: Individual component scores.
        recommendations_count: Number of recommendations generated.
        high_severity_count: Number of high severity issues.
        summary: Human-readable summary of the health score.
    """

    overall: float
    grade: str
    components: list[HealthScoreComponent]
    recommendations_count: int
    high_severity_count: int
    summary: str

    def to_dict(self) -> dict[str, Any]:
        """Convert health score to a dictionary.

        Returns:
            Dictionary representation of the health score.
        """
        return {
            "overall": round(self.overall, 1),
            "grade": self.grade,
            "components": [
                {
                    "name": c.name,
                    "score": round(c.score, 1),
                    "weight": c.weight,
                    "description": c.description,
                    "details": c.details,
                }
                for c in self.components
            ],
            "recommendations_count": self.recommendations_count,
            "high_severity_count": self.high_severity_count,
            "summary": self.summary,
        }


class HealthScoreCalculator:
    """Calculator for dataset health scores.

    The health score is computed from multiple dimensions:
    - Diversity: Vocabulary richness and content variety
    - Balance: Role distribution and conversation length balance
    - Quality: Based on quality analyzer scores (PII, encoding, etc.)
    - Consistency: Format consistency and language consistency
    - Length Distribution: Token/word length distribution health

    Each dimension contributes to the overall score with configurable weights.
    """

    # Default weights for each component (must sum to 1.0)
    DEFAULT_WEIGHTS = {
        "diversity": 0.15,
        "balance": 0.10,
        "quality": 0.15,
        "consistency": 0.10,
        "length_distribution": 0.10,
        "training_quality": 0.25,
        "data_hygiene": 0.15,
    }

    # Grade thresholds
    GRADE_THRESHOLDS = [
        (90, "A"),
        (80, "B"),
        (70, "C"),
        (60, "D"),
        (0, "F"),
    ]

    def __init__(
        self,
        *,
        weights: Optional[dict[str, float]] = None,
        recommendation_penalty: float = 2.0,
        high_severity_penalty: float = 5.0,
    ):
        """Initialize the HealthScoreCalculator.

        Args:
            weights: Custom weights for each component. Keys should match
                DEFAULT_WEIGHTS keys. Values must sum to 1.0.
            recommendation_penalty: Points to deduct per recommendation.
            high_severity_penalty: Additional points to deduct per high severity issue.
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.recommendation_penalty = recommendation_penalty
        self.high_severity_penalty = high_severity_penalty

        # Validate weights sum to ~1.0
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            raise ValueError(
                f"Weights must sum to 1.0, got {weight_sum}. "
                f"Weights: {self.weights}"
            )

    def calculate_health_score(
        self,
        message_df: pd.DataFrame,
        conversation_df: pd.DataFrame,
        analysis_summary: dict[str, Any],
        recommendations: list[Recommendation],
    ) -> DatasetHealthScore:
        """Calculate the comprehensive health score for a dataset.

        Args:
            message_df: DataFrame with message-level analysis results.
            conversation_df: DataFrame with conversation-level analysis results.
            analysis_summary: Summary statistics from the analysis.
            recommendations: List of recommendations from the analysis.

        Returns:
            DatasetHealthScore with overall score and component breakdown.
        """
        components = []

        # Calculate each component
        diversity_score = self._calculate_diversity_score(message_df)
        components.append(diversity_score)

        balance_score = self._calculate_balance_score(message_df, conversation_df)
        components.append(balance_score)

        quality_score = self._calculate_quality_score(message_df)
        components.append(quality_score)

        consistency_score = self._calculate_consistency_score(message_df)
        components.append(consistency_score)

        length_score = self._calculate_length_distribution_score(
            message_df, analysis_summary
        )
        components.append(length_score)

        training_quality_score = self._calculate_training_quality_score(message_df)
        components.append(training_quality_score)

        data_hygiene_score = self._calculate_data_hygiene_score(message_df)
        components.append(data_hygiene_score)

        # Calculate weighted overall score
        overall = sum(
            c.score * self.weights.get(c.name.lower().replace(" ", "_"), 0.2)
            for c in components
        )

        # Apply penalties for recommendations
        high_severity_count = sum(
            1 for r in recommendations if r.severity == RecommendationSeverity.HIGH
        )
        penalty = (
            len(recommendations) * self.recommendation_penalty
            + high_severity_count * self.high_severity_penalty
        )
        overall = max(0, overall - penalty)

        # Determine grade
        grade = "F"
        for threshold, letter in self.GRADE_THRESHOLDS:
            if overall >= threshold:
                grade = letter
                break

        # Generate summary
        summary = self._generate_summary(overall, grade, components, recommendations)

        return DatasetHealthScore(
            overall=overall,
            grade=grade,
            components=components,
            recommendations_count=len(recommendations),
            high_severity_count=high_severity_count,
            summary=summary,
        )

    def _calculate_diversity_score(self, df: pd.DataFrame) -> HealthScoreComponent:
        """Calculate diversity component score.

        Args:
            df: Message-level DataFrame.

        Returns:
            HealthScoreComponent for diversity.
        """
        score = 70.0  # Default baseline
        details = {}

        # Check for diversity metrics
        diversity_cols = [col for col in df.columns if "diversity" in col.lower()]
        unique_ratio_cols = [
            col for col in df.columns if "unique_words_ratio" in col.lower()
        ]

        if unique_ratio_cols:
            col = unique_ratio_cols[0]
            avg_ratio = df[col].dropna().mean()
            if avg_ratio > 0:
                score = min(100, avg_ratio * 120)
                details["avg_unique_words_ratio"] = round(avg_ratio, 3)

        elif diversity_cols:
            # Generic diversity metric
            col = diversity_cols[0]
            values = df[col].dropna()
            if len(values) > 0:
                avg_val = values.mean()
                details["avg_diversity"] = round(avg_val, 3)
                # Assume normalized 0-1 scale
                score = min(100, avg_val * 100)

        # Check for duplicate content (penalize)
        if "text_content" in df.columns:
            duplicate_ratio = df["text_content"].duplicated().mean()
            if duplicate_ratio > 0.05:
                penalty = min(30, duplicate_ratio * 100)
                score = max(0, score - penalty)
                details["duplicate_ratio"] = round(duplicate_ratio, 3)

        return HealthScoreComponent(
            name="Diversity",
            score=score,
            weight=self.weights.get("diversity", 0.2),
            description="Measures vocabulary richness and content variety",
            details=details,
        )

    def _calculate_balance_score(
        self, message_df: pd.DataFrame, conversation_df: pd.DataFrame
    ) -> HealthScoreComponent:
        """Calculate balance component score.

        Args:
            message_df: Message-level DataFrame.
            conversation_df: Conversation-level DataFrame.

        Returns:
            HealthScoreComponent for balance.
        """
        score = 80.0  # Default baseline
        details = {}

        # Check role distribution
        if "role" in message_df.columns:
            role_counts = message_df["role"].value_counts(normalize=True)
            details["role_distribution"] = {
                str(k): round(v, 3) for k, v in role_counts.items()
            }

            # Ideal: user and assistant roughly balanced (system can be less)
            user_ratio = float(role_counts.get("user", 0) or 0)
            assistant_ratio = float(role_counts.get("assistant", 0) or 0)

            if user_ratio > 0 and assistant_ratio > 0:
                # Calculate balance between user and assistant
                min_ratio = min(user_ratio, assistant_ratio)
                max_ratio = max(user_ratio, assistant_ratio)
                balance = min_ratio / max_ratio if max_ratio > 0 else 0.0
                # Perfect balance = 1.0, scale to 0-100
                role_score = balance * 100
                details["user_assistant_balance"] = round(balance, 3)
            else:
                role_score = 50.0  # Missing roles

            score = (score + role_score) / 2

        # Check conversation length distribution
        if "num_messages" in conversation_df.columns:
            num_msgs = conversation_df["num_messages"]
            cv = num_msgs.std() / num_msgs.mean() if num_msgs.mean() > 0 else 0

            # Lower CV = more consistent lengths = better
            # CV of 0.5 is reasonable, CV > 1.5 is very inconsistent
            cv_score = max(0, 100 - (cv * 40))
            details["conversation_length_cv"] = round(cv, 3)

            score = (score + cv_score) / 2

        return HealthScoreComponent(
            name="Balance",
            score=score,
            weight=self.weights.get("balance", 0.15),
            description="Measures role distribution and conversation length balance",
            details=details,
        )

    def _calculate_quality_score(self, df: pd.DataFrame) -> HealthScoreComponent:
        """Calculate quality component score.

        Args:
            df: Message-level DataFrame.

        Returns:
            HealthScoreComponent for quality.
        """
        score = 85.0  # Default baseline (optimistic)
        details = {}

        # Penalize for issues
        issue_penalties = [
            ("has_pii", 15),
            ("has_encoding_issues", 10),
            ("has_high_repetition", 5),
        ]

        for issue_pattern, penalty in issue_penalties:
            issue_cols = [col for col in df.columns if issue_pattern in col]
            for col in issue_cols:
                issue_ratio = df[col].mean()
                if issue_ratio > 0:
                    actual_penalty = penalty * issue_ratio
                    score = max(0, score - actual_penalty)
                    details[f"{issue_pattern}_ratio"] = round(issue_ratio, 3)
                break

        return HealthScoreComponent(
            name="Quality",
            score=score,
            weight=self.weights.get("quality", 0.25),
            description="Measures content quality based on PII, encoding, and safety",
            details=details,
        )

    def _calculate_consistency_score(self, df: pd.DataFrame) -> HealthScoreComponent:
        """Calculate consistency component score.

        Args:
            df: Message-level DataFrame.

        Returns:
            HealthScoreComponent for consistency.
        """
        score = 80.0  # Default baseline
        details = {}

        # Check language consistency
        lang_cols = [col for col in df.columns if "detected_language" in col]
        if lang_cols:
            col = lang_cols[0]
            languages = df[col].dropna()
            languages = languages[languages != ""]

            if len(languages) > 10:
                from collections import Counter

                lang_counts = Counter(languages)
                total = sum(lang_counts.values())
                dominant_lang, dominant_count = lang_counts.most_common(1)[0]
                consistency = dominant_count / total

                # Scale: 100% same language = 100, 50% = 50
                lang_score = consistency * 100
                details["language_consistency"] = round(consistency, 3)
                details["dominant_language"] = dominant_lang

                score = (score + lang_score) / 2

        # Check format consistency (using format analyzer results)
        format_cols = [col for col in df.columns if "has_markdown" in col]
        if format_cols:
            # Calculate what fraction have consistent formatting
            col = format_cols[0]
            format_ratio = df[col].mean()
            # Either all have markdown or none - both are consistent
            format_consistency = max(format_ratio, 1 - format_ratio)
            format_score = format_consistency * 100
            details["format_consistency"] = round(format_consistency, 3)

            score = (score + format_score) / 2

        return HealthScoreComponent(
            name="Consistency",
            score=score,
            weight=self.weights.get("consistency", 0.2),
            description="Measures language and format consistency across samples",
            details=details,
        )

    def _calculate_length_distribution_score(
        self, df: pd.DataFrame, analysis_summary: dict[str, Any]
    ) -> HealthScoreComponent:
        """Calculate length distribution component score.

        For unimodal distributions, uses CV-based scoring.
        For multimodal distributions, scores based on within-mode consistency
        and mode separation. Well-separated modes with low within-mode variance
        indicate intentional content types (e.g., short questions vs long explanations).

        Args:
            df: Message-level DataFrame.
            analysis_summary: Summary statistics.

        Returns:
            HealthScoreComponent for length distribution.
        """
        score = 80.0  # Default baseline
        details = {}

        # Check for length metrics
        length_cols = [
            col
            for col in df.columns
            if any(m in col for m in ["char_count", "word_count", "token_count"])
        ]

        if length_cols:
            col = length_cols[0]
            lengths = df[col].dropna()

            if len(lengths) > 0:
                mean_len = lengths.mean()
                std_len = lengths.std()
                cv = std_len / mean_len if mean_len > 0 else 0

                details["mean_length"] = round(mean_len, 1)
                details["std_length"] = round(std_len, 1)
                details["cv"] = round(cv, 3)

                # Detect distribution type
                dist_result = detect_distribution_type(lengths)
                details["distribution_type"] = dist_result.distribution_type.value

                if dist_result.distribution_type in (
                    DistributionType.BIMODAL,
                    DistributionType.MULTIMODAL,
                ):
                    # For multimodal distributions, score based on:
                    # 1. Within-mode consistency (low CV per mode = good)
                    # 2. Mode separation confidence (well-separated = intentional)
                    details["num_modes"] = dist_result.num_modes
                    details["mode_separation_confidence"] = dist_result.confidence

                    # Calculate average within-mode CV
                    within_mode_cvs = []
                    for ms in dist_result.mode_statistics:
                        mode_cv = ms.std / ms.mean if ms.mean > 0 else 0
                        within_mode_cvs.append(mode_cv)

                    if within_mode_cvs:
                        avg_within_cv = sum(within_mode_cvs) / len(within_mode_cvs)
                        details["avg_within_mode_cv"] = round(avg_within_cv, 3)

                        # Score: Low within-mode CV + high separation = good
                        # Within-mode CV < 0.5 is excellent
                        cv_score = max(0, 100 - (avg_within_cv * 80))

                        # Bonus for well-separated modes (indicates intentional design)
                        separation_bonus = dist_result.confidence * 20

                        score = min(100, cv_score + separation_bonus)
                    else:
                        # Fallback to global CV if mode stats unavailable
                        cv_penalty = max(0, (cv - 1.0) * 20)
                        score = max(0, score - cv_penalty)

                    # Add mode details
                    details["mode_statistics"] = [
                        {
                            "mode_id": ms.mode_id,
                            "mean": ms.mean,
                            "std": ms.std,
                            "count": ms.count,
                            "weight": ms.weight,
                        }
                        for ms in dist_result.mode_statistics
                    ]
                else:
                    # Unimodal: use existing CV-based scoring
                    # Penalize high variation
                    # CV of 1.0 is acceptable, CV > 2.0 is problematic
                    cv_penalty = max(0, (cv - 1.0) * 20)
                    score = max(0, score - cv_penalty)

                # Penalize very short content (applies to all distributions)
                short_ratio = (lengths < 10).mean()
                if short_ratio > 0.1:
                    short_penalty = short_ratio * 30
                    score = max(0, score - short_penalty)
                    details["short_content_ratio"] = round(short_ratio, 3)

                # Penalize extremely long content (>4096 tokens)
                if "token_count" in col:
                    long_ratio = (lengths > 4096).mean()
                    if long_ratio > 0.05:
                        long_penalty = long_ratio * 40
                        score = max(0, score - long_penalty)
                        details["exceeds_4k_ratio"] = round(long_ratio, 3)

        return HealthScoreComponent(
            name="Length Distribution",
            score=score,
            weight=self.weights.get("length_distribution", 0.1),
            description="Measures length distribution health and outliers",
            details=details,
        )

    def _calculate_training_quality_score(
        self, df: pd.DataFrame
    ) -> HealthScoreComponent:
        """Calculate training quality component score.

        This component measures how well the dataset is structured for
        effective SFT training based on instruction clarity and response
        completeness metrics.

        Args:
            df: Message-level DataFrame.

        Returns:
            HealthScoreComponent for training quality.
        """
        score = 75.0  # Default baseline
        details = {}

        # Check response completeness score
        completeness_cols = [
            col for col in df.columns if "response_completeness_score" in col
        ]
        if completeness_cols:
            col = completeness_cols[0]
            scores = df[col].dropna()
            if len(scores) > 0:
                avg_completeness = scores.mean()
                completeness_score = avg_completeness * 100
                details["avg_response_completeness"] = round(avg_completeness, 3)
                details["incomplete_ratio"] = round((scores < 0.5).mean(), 3)
                score = completeness_score

        # Check for truncated responses (penalize)
        ending_cols = [col for col in df.columns if "has_proper_ending" in col]
        if ending_cols:
            col = ending_cols[0]
            endings = df[col].dropna()
            if len(endings) > 0:
                truncated_ratio = (~endings).mean()
                if truncated_ratio > 0.05:
                    penalty = truncated_ratio * 30
                    score = max(0, score - penalty)
                    details["truncated_ratio"] = round(truncated_ratio, 3)

        return HealthScoreComponent(
            name="Training Quality",
            score=score,
            weight=self.weights.get("training_quality", 0.25),
            description="Measures instruction clarity and response completeness for SFT",
            details=details,
        )

    def _calculate_data_hygiene_score(self, df: pd.DataFrame) -> HealthScoreComponent:
        """Calculate data hygiene component score.

        This component measures data cleanliness including duplicates,
        format consistency, and data integrity issues.

        Args:
            df: Message-level DataFrame.

        Returns:
            HealthScoreComponent for data hygiene.
        """
        score = 90.0  # Default baseline (optimistic - assume clean)
        details = {}

        # Check for duplicates
        if "text_content" in df.columns:
            duplicate_ratio = df["text_content"].duplicated().mean()
            if duplicate_ratio > 0:
                # Scale penalty: 5% duplicates = -10, 20% = -40
                dup_penalty = min(50, duplicate_ratio * 200)
                score = max(0, score - dup_penalty)
                details["duplicate_ratio"] = round(duplicate_ratio, 3)

        # Check for near-duplicates from embedding analyzer
        near_dup_cols = [col for col in df.columns if "near_duplicate" in col.lower()]
        if near_dup_cols:
            col = near_dup_cols[0]
            near_dup_ratio = df[col].mean()
            if near_dup_ratio > 0:
                near_dup_penalty = min(30, near_dup_ratio * 150)
                score = max(0, score - near_dup_penalty)
                details["near_duplicate_ratio"] = round(near_dup_ratio, 3)

        # Check for encoding issues
        encoding_cols = [col for col in df.columns if "has_encoding_issues" in col]
        if encoding_cols:
            col = encoding_cols[0]
            encoding_ratio = df[col].mean()
            if encoding_ratio > 0:
                encoding_penalty = min(25, encoding_ratio * 150)
                score = max(0, score - encoding_penalty)
                details["encoding_issues_ratio"] = round(encoding_ratio, 3)

        # Check for empty content
        if "text_content" in df.columns:
            empty_ratio = (df["text_content"].astype(str).str.len() <= 5).mean()
            if empty_ratio > 0.01:
                empty_penalty = min(20, empty_ratio * 100)
                score = max(0, score - empty_penalty)
                details["empty_content_ratio"] = round(empty_ratio, 3)

        return HealthScoreComponent(
            name="Data Hygiene",
            score=score,
            weight=self.weights.get("data_hygiene", 0.15),
            description="Measures data cleanliness including duplicates and integrity",
            details=details,
        )

    def _generate_summary(
        self,
        overall: float,
        grade: str,
        components: list[HealthScoreComponent],
        recommendations: list[Recommendation],
    ) -> str:
        """Generate a human-readable summary of the health score.

        Args:
            overall: Overall score.
            grade: Letter grade.
            components: Component scores.
            recommendations: List of recommendations.

        Returns:
            Summary string.
        """
        # Find weakest components
        sorted_components = sorted(components, key=lambda c: c.score)
        weakest = sorted_components[:2]

        summary_parts = [f"Dataset Health: {grade} ({overall:.0f}/100)."]

        if grade in ("A", "B"):
            summary_parts.append("Dataset quality is good for training.")
        elif grade == "C":
            summary_parts.append("Dataset has some issues that may affect training.")
        else:
            summary_parts.append("Dataset has significant issues requiring attention.")

        if weakest:
            weak_names = [c.name for c in weakest if c.score < 70]
            if weak_names:
                summary_parts.append(
                    f"Areas needing improvement: {', '.join(weak_names)}."
                )

        high_severity = [
            r for r in recommendations if r.severity == RecommendationSeverity.HIGH
        ]
        if high_severity:
            summary_parts.append(
                f"Found {len(high_severity)} high-severity issues to address."
            )

        return " ".join(summary_parts)
