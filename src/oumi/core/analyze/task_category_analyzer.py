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

"""Task category analyzer for classifying instruction types.

Based on the Magpie framework taxonomy from "Fixing It in Post" paper,
this analyzer classifies instructions into task categories to help
understand dataset composition and identify task distribution imbalances.
"""

import re
from typing import Any, Optional

import pandas as pd

from oumi.core.analyze.column_types import ColumnType, ContentType
from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.registry import register_sample_analyzer


@register_sample_analyzer("task_category")
class TaskCategoryAnalyzer(SampleAnalyzer):
    """Analyzer for classifying instructions into task categories.

    Based on the Magpie framework taxonomy, this analyzer classifies
    user instructions into categories such as:
        - math: Mathematical problems, calculations, proofs
        - coding: Programming tasks, debugging, code generation
        - information_seeking: Factual questions, definitions, explanations
        - creative_writing: Stories, poems, creative content
        - editing: Text editing, rewriting, grammar correction
        - advice: Personal advice, recommendations, suggestions
        - reasoning: Logical reasoning, analysis, problem-solving
        - brainstorming: Idea generation, planning, lists
        - role_play: Character personas, simulated scenarios
        - data_analysis: Data processing, statistics, visualization
        - translation: Language translation tasks
        - other: Tasks that don't fit other categories

    Metrics computed:
        - task_category: The primary category of the instruction
        - task_confidence: Confidence score (0-1) for the classification
        - is_stem: Whether the task is STEM-related (math, coding, data_analysis)
        - is_conversational: Whether task is conversational (advice, role_play)
    """

    # Task category patterns - based on Magpie framework taxonomy
    _TASK_PATTERNS: dict[str, list[re.Pattern]] = {
        "math": [
            re.compile(
                r"\b(?:calculate|compute|solve|find\s+(?:the\s+)?(?:value|"
                r"derivative|integral|sum|product|area|volume|perimeter))\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(?:equation|formula|mathematics|algebra|geometry|"
                r"calculus|trigonometry|probability|statistics)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(?:prove|theorem|lemma|proof|mathematical|"
                r"arithmetic|fraction|decimal|percentage)\b",
                re.IGNORECASE,
            ),
            re.compile(r"\b(?:sin|cos|tan|log|sqrt|integral|derivative)\b"),
            re.compile(r"[+\-*/=]\s*\d+|\d+\s*[+\-*/=]"),
            re.compile(r"\b\d+\s*(?:plus|minus|times|divided\s+by)\s*\d+\b"),
        ],
        "coding": [
            re.compile(
                r"\b(?:write|create|implement|code|program|function|class|"
                r"method|algorithm|script)\b.*\b(?:in\s+)?(?:python|java|"
                r"javascript|c\+\+|rust|go|typescript|ruby|php|swift)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(?:debug|fix|refactor|optimize|compile|execute|run)\s+"
                r"(?:the\s+)?(?:code|program|script|function)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(?:API|SDK|library|framework|module|package|import|"
                r"export|async|await|promise|callback)\b",
                re.IGNORECASE,
            ),
            re.compile(r"```\w*\n|def\s+\w+|function\s+\w+|class\s+\w+"),
            re.compile(
                r"\b(?:variable|array|list|dict|object|string|integer|"
                r"boolean|null|void|return|if|else|for|while|loop)\b"
            ),
        ],
        "information_seeking": [
            re.compile(
                r"^(?:what|who|when|where|why|how|which)\s+",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(?:explain|describe|define|tell\s+me|inform|clarify|"
                r"elaborate|what\s+is|what\s+are|who\s+is|how\s+does)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(?:meaning|definition|explanation|information|"
                r"facts|history|background|overview)\b",
                re.IGNORECASE,
            ),
        ],
        "creative_writing": [
            re.compile(
                r"\b(?:write|create|compose|draft)\s+(?:a\s+)?(?:story|poem|"
                r"essay|article|script|dialogue|narrative|novel|fiction)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(?:creative|imaginative|fictional|artistic|literary|"
                r"poetic|prose|verse|stanza|chapter)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(?:character|plot|setting|theme|conflict|protagonist|"
                r"antagonist|narrator|climax)\b",
                re.IGNORECASE,
            ),
        ],
        "editing": [
            re.compile(
                r"\b(?:edit|rewrite|revise|improve|fix|correct|proofread|"
                r"rephrase|paraphrase|reword)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(?:grammar|spelling|punctuation|syntax|style|tone|"
                r"clarity|concise|coherent|flow)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(?:make\s+(?:it\s+)?(?:shorter|longer|clearer|simpler|"
                r"more\s+formal|less\s+formal|better))\b",
                re.IGNORECASE,
            ),
        ],
        "advice": [
            re.compile(
                r"\b(?:advice|suggest|recommend|should\s+I|what\s+should|"
                r"how\s+should|tips|guidance|counsel)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(?:help\s+me\s+(?:decide|choose|figure\s+out)|"
                r"what\s+(?:do\s+you|would\s+you)\s+(?:think|suggest|recommend))\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(?:career|relationship|health|financial|personal|"
                r"life\s+advice|decision)\b",
                re.IGNORECASE,
            ),
        ],
        "reasoning": [
            re.compile(
                r"\b(?:analyze|evaluate|assess|compare|contrast|critique|"
                r"argue|justify|reason|conclude)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(?:pros\s+and\s+cons|advantages|disadvantages|"
                r"benefits|drawbacks|implications|consequences)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(?:logical|reasoning|argument|premise|conclusion|"
                r"inference|deduction|induction|fallacy)\b",
                re.IGNORECASE,
            ),
        ],
        "brainstorming": [
            re.compile(
                r"\b(?:brainstorm|generate|list|come\s+up\s+with|"
                r"think\s+of|ideas?\s+for|suggestions?\s+for)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(?:give\s+me|provide|suggest)\s+(?:\d+\s+)?(?:ideas?|"
                r"options?|alternatives?|examples?|ways?)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(?:name|list|enumerate)\s+(?:\d+\s+)?(?:things?|"
                r"items?|examples?|reasons?|ways?)\b",
                re.IGNORECASE,
            ),
        ],
        "role_play": [
            re.compile(
                r"\b(?:pretend|act\s+as|role\s*-?\s*play|simulate|"
                r"imagine\s+you(?:'re|\s+are)|you\s+are\s+(?:a|an|the))\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(?:character|persona|scenario|dialogue|conversation|"
                r"interview|debate|as\s+if)\b",
                re.IGNORECASE,
            ),
        ],
        "data_analysis": [
            re.compile(
                r"\b(?:analyze|process|parse|extract|transform|aggregate|"
                r"filter|sort|group|join)\s+(?:the\s+)?(?:data|dataset)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(?:statistics|mean|median|mode|standard\s+deviation|"
                r"variance|correlation|regression|distribution)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(?:CSV|JSON|XML|SQL|database|table|column|row|"
                r"dataframe|pandas|numpy)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(?:visualization|chart|graph|plot|histogram|"
                r"scatter|bar\s+chart|pie\s+chart)\b",
                re.IGNORECASE,
            ),
        ],
        "translation": [
            re.compile(
                r"\b(?:translate|translation|convert)\s+(?:this\s+|the\s+)?"
                r"(?:text\s+|sentence\s+|phrase\s+|word\s+)?(?:from|into|to)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(?:in\s+)?(?:english|spanish|french|german|chinese|"
                r"japanese|korean|arabic|portuguese|russian|italian)\b",
                re.IGNORECASE,
            ),
        ],
    }

    # STEM categories
    _STEM_CATEGORIES = {"math", "coding", "data_analysis"}

    # Conversational categories
    _CONVERSATIONAL_CATEGORIES = {"advice", "role_play", "brainstorming"}

    def __init__(
        self,
        *,
        min_confidence: float = 0.3,
        default_category: str = "other",
        analyze_user_only: bool = True,
    ):
        """Initialize the TaskCategoryAnalyzer.

        Args:
            min_confidence: Minimum confidence threshold for classification.
            default_category: Category to assign when no patterns match.
            analyze_user_only: If True, only analyze user messages.
        """
        self.min_confidence = min_confidence
        self.default_category = default_category
        self.analyze_user_only = analyze_user_only

    def _classify_text(self, text: str) -> dict[str, Any]:
        """Classify a text into task categories.

        Args:
            text: Text to classify.

        Returns:
            Dictionary with classification results.
        """
        if not text or not text.strip():
            return {
                "task_category": self.default_category,
                "task_confidence": 0.0,
                "is_stem": False,
                "is_conversational": False,
            }

        # Count pattern matches for each category
        category_scores: dict[str, int] = {}
        for category, patterns in self._TASK_PATTERNS.items():
            score = 0
            for pattern in patterns:
                matches = pattern.findall(text)
                score += len(matches)
            if score > 0:
                category_scores[category] = score

        # Determine best category
        if not category_scores:
            return {
                "task_category": self.default_category,
                "task_confidence": 0.0,
                "is_stem": False,
                "is_conversational": False,
            }

        # Get the category with highest score
        best_category = max(category_scores, key=lambda k: category_scores[k])
        best_score = category_scores[best_category]
        total_score = sum(category_scores.values())

        # Calculate confidence as proportion of matches
        confidence = best_score / total_score if total_score > 0 else 0.0

        # Apply minimum confidence threshold
        if confidence < self.min_confidence:
            return {
                "task_category": self.default_category,
                "task_confidence": confidence,
                "is_stem": False,
                "is_conversational": False,
            }

        return {
            "task_category": best_category,
            "task_confidence": round(confidence, 3),
            "is_stem": best_category in self._STEM_CATEGORIES,
            "is_conversational": best_category in self._CONVERSATIONAL_CATEGORIES,
        }

    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> tuple[pd.DataFrame, dict]:
        """Analyze text fields for task category classification.

        Args:
            df: Input DataFrame with text fields.
            schema: Column schema dict to identify text fields.

        Returns:
            Tuple of (DataFrame with added task category analysis columns.
            generated column schema dict).
        """
        result_df = df.copy()
        generated_schema = {}

        if not schema:
            raise ValueError(
                "schema is required to identify text fields for task category "
                "analysis. Please provide a column schema dict that specifies "
                "which columns contain text content."
            )

        text_columns = [
            col
            for col, config in schema.items()
            if config.get("content_type") == ContentType.TEXT and col in df.columns
        ]

        if not text_columns:
            return result_df, generated_schema

        # Find the role column if we need to filter
        role_column = None
        if self.analyze_user_only:
            for col, config in schema.items():
                if (
                    config.get("content_type") == ContentType.CATEGORICAL
                    and col in df.columns
                    and "role" in col.lower()
                ):
                    role_column = col
                    break

        analyzer_id = getattr(self, "analyzer_id", "task_category")

        for column in text_columns:
            if self.analyze_user_only and role_column is not None:
                # Only analyze user messages
                analysis_results = df.apply(
                    lambda row: (
                        self._classify_text(str(row[column]))
                        if str(row.get(role_column, "")).lower() == "user"
                        else {
                            "task_category": None,
                            "task_confidence": None,
                            "is_stem": None,
                            "is_conversational": None,
                        }
                    ),
                    axis=1,
                )
            else:
                analysis_results = df[column].astype(str).apply(self._classify_text)

            # Extract results to columns
            col_name = f"{column}_{analyzer_id}_category"
            result_df[col_name] = analysis_results.apply(
                lambda r: r.get("task_category")
            )
            generated_schema[col_name] = {
                "type": ColumnType.STRING,
                "content_type": ContentType.CATEGORICAL,
                "description": "Predicted task category",
            }
            
            col_name = f"{column}_{analyzer_id}_confidence"
            result_df[col_name] = analysis_results.apply(
                lambda r: r.get("task_confidence")
            )
            generated_schema[col_name] = {
                "type": ColumnType.FLOAT,
                "content_type": ContentType.NUMERIC,
                "description": "Confidence score for predicted category (0.0-1.0)",
            }
            
            col_name = f"{column}_{analyzer_id}_is_stem"
            result_df[col_name] = analysis_results.apply(
                lambda r: r.get("is_stem")
            )
            generated_schema[col_name] = {
                "type": ColumnType.BOOL,
                "content_type": ContentType.BOOLEAN,
                "description": "Whether task is in STEM category",
            }
            
            col_name = f"{column}_{analyzer_id}_is_conversational"
            result_df[col_name] = analysis_results.apply(lambda r: r.get("is_conversational"))
            generated_schema[col_name] = {
                "type": ColumnType.BOOL,
                "content_type": ContentType.BOOLEAN,
                "description": "Whether task is conversational in nature",
            }

        return result_df, generated_schema
