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

"""Content pattern analyzer for detecting AI-specific quality issues in datasets.

This analyzer identifies quality issues commonly found in AI-generated training data:
- Placeholder text ([Name], [Product Name], etc.)
- AI hallucinated personal experiences
- Nooutput tags and AI refusals
- Suspicious/potentially fake URLs
"""

import re
from typing import Any, Optional

import pandas as pd

from oumi.core.analyze.column_types import ContentType
from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.registry import register_sample_analyzer


@register_sample_analyzer("content_pattern")
class ContentPatternAnalyzer(SampleAnalyzer):
    """Analyzer for detecting AI-specific content quality issues.

    This analyzer identifies common quality problems in AI-generated training data:
        - Placeholder text: [Name], [Product Name], [Your...], [Insert...], etc.
        - AI hallucinated experiences: Fabricated first-person stories
        - Nooutput markers: <nooutput>, N/A, and similar non-responses
        - AI refusals: "I cannot provide...", "I'm unable to..."
        - Suspicious URLs: Potentially hallucinated/fake URLs

    Quality metrics computed:
        - has_placeholder: Boolean indicating placeholder text detected
        - placeholder_count: Number of placeholders found
        - placeholder_types: Types of placeholders detected
        - has_hallucinated_experience: Boolean for AI fabricated stories
        - has_nooutput: Boolean for nooutput/NA markers
        - has_refusal: Boolean for AI refusal patterns
        - has_suspicious_url: Boolean for potentially fake URLs
        - content_pattern_score: Composite quality score (0-1, higher is cleaner)
    """

    # Bracketed placeholders: [Name], [Product Name], [Your Name], etc.
    _BRACKET_PLACEHOLDER_PATTERN = re.compile(
        r"\[(?:Name|Product\s*Name|Company\s*Name|Your\s*\w*|Insert\s*\w*|"
        r"Add\s*\w*|Fill\s*(?:in\s*)?\w*|Replace\s*(?:with\s*)?\w*|"
        r"Enter\s*\w*|Specify\s*\w*|X{2,}|Placeholder|TBD|TODO|FIXME|"
        r"Person|Place|Date|Time|Location|Address|City|Country|"
        r"Email|Phone|Number|Amount|Price|Value|Title|Subject|Topic)\]",
        re.IGNORECASE,
    )

    # Angle bracket placeholders: <name>, <your_company>, etc.
    _ANGLE_PLACEHOLDER_PATTERN = re.compile(
        r"<(?:name|your[_\s]?\w*|company[_\s]?\w*|product[_\s]?\w*|"
        r"insert[_\s]?\w*|fill[_\s]?\w*|placeholder|tbd|todo|"
        r"person|place|date|time|email|phone)\s*/?>",
        re.IGNORECASE,
    )

    # AI hallucinated experience patterns
    _HALLUCINATED_EXPERIENCE_PATTERNS = [
        # "I had to... when I was working as..."
        re.compile(
            r"\bI\s+(?:had\s+to|needed\s+to|was\s+asked\s+to|decided\s+to|"
            r"chose\s+to)\s+\w+.*?(?:when\s+I\s+(?:was|worked)|"
            r"during\s+my\s+(?:time|tenure|career))",
            re.IGNORECASE,
        ),
        # "When I was a [profession]..."
        re.compile(
            r"\bWhen\s+I\s+was\s+(?:a\s+|an\s+)?(?:project\s+manager|"
            r"software\s+engineer|developer|doctor|nurse|teacher|lawyer|"
            r"manager|director|CEO|founder|consultant|analyst|designer|"
            r"researcher|scientist|professor|engineer|accountant|"
            r"marketing\s+manager|sales\s+representative|HR\s+manager)\b",
            re.IGNORECASE,
        ),
        # "In my experience as a..."
        re.compile(
            r"\b(?:In\s+my|From\s+my|Based\s+on\s+my)\s+"
            r"(?:experience|time|years?|work|career)\s+"
            r"(?:as\s+|at\s+|in\s+|with\s+|working\s+)?(?:a\s+|an\s+)?",
            re.IGNORECASE,
        ),
        # "During my career/time at..."
        re.compile(
            r"\bDuring\s+my\s+(?:career|time|tenure|work|years?)\s+"
            r"(?:at|with|in|as)\s+",
            re.IGNORECASE,
        ),
        # "I recently visited/attended..."
        re.compile(
            r"\bI\s+recently\s+(?:visited|attended|went\s+to|completed|"
            r"finished|started|joined)\s+",
            re.IGNORECASE,
        ),
    ]

    # Nooutput and non-response patterns
    _NOOUTPUT_PATTERNS = [
        re.compile(r"<\s*nooutput\s*>", re.IGNORECASE),
        re.compile(r"<\s*no[_\s-]?output\s*>", re.IGNORECASE),
        re.compile(r"\[no[_\s-]?output\]", re.IGNORECASE),
        re.compile(r"^\s*N/?A\s*$"),
        re.compile(r"^\s*\[N/?A\]\s*$"),
        re.compile(r"^\s*None\s*$", re.IGNORECASE),
        re.compile(r"^\s*-\s*$"),
    ]

    # AI refusal patterns
    _REFUSAL_PATTERNS = [
        re.compile(
            r"\bI(?:'m|\s+am)?\s*(?:cannot|can't|unable\s+to|won't|will\s+not|"
            r"refuse\s+to|not\s+able\s+to)\s+"
            r"(?:provide|generate|create|write|produce|help|"
            r"assist|complete|fulfill)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:This|That|The)\s+(?:request|question|prompt|task)\s+"
            r"(?:is|seems|appears)\s+(?:inappropriate|harmful|unethical|"
            r"impossible|beyond\s+my\s+capabilities)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:This|That)\s+(?:type\s+of\s+)?(?:instruction|request)\s+"
            r"cannot\s+be\s+(?:fulfilled|completed|processed)",
            re.IGNORECASE,
        ),
    ]

    # Suspicious URL patterns (potentially hallucinated)
    _SUSPICIOUS_URL_PATTERN = re.compile(
        r"https?://(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?",
        re.IGNORECASE,
    )

    # Known fake/commonly hallucinated domains
    _HALLUCINATED_DOMAINS = {
        "diversityintech.com",
        "aiethics.org",
        "techforgood.com",
        "example-company.com",
        "samplewebsite.com",
        "yourwebsite.com",
        "mycompany.com",
        "testsite.com",
    }

    def __init__(
        self,
        *,
        detect_placeholders: bool = True,
        detect_hallucinated_experiences: bool = True,
        detect_nooutput: bool = True,
        detect_refusals: bool = True,
        detect_suspicious_urls: bool = False,
        placeholder_whitelist: Optional[list[str]] = None,
        check_output_only: bool = False,
        compute_content_pattern_score: bool = True,
    ):
        """Initialize the ContentPatternAnalyzer.

        Args:
            detect_placeholders: Whether to detect placeholder text like [Name].
            detect_hallucinated_experiences: Whether to detect AI fabricated stories.
            detect_nooutput: Whether to detect nooutput/NA markers.
            detect_refusals: Whether to detect AI refusal patterns.
            detect_suspicious_urls: Whether to detect potentially fake URLs.
            placeholder_whitelist: List of placeholder patterns to ignore.
            check_output_only: If True, only analyze assistant/output messages.
            compute_content_pattern_score: Whether to compute composite score.
        """
        self.detect_placeholders = detect_placeholders
        self.detect_hallucinated_experiences = detect_hallucinated_experiences
        self.detect_nooutput = detect_nooutput
        self.detect_refusals = detect_refusals
        self.detect_suspicious_urls = detect_suspicious_urls
        self.placeholder_whitelist = set(placeholder_whitelist or [])
        self.check_output_only = check_output_only
        self.compute_content_pattern_score = compute_content_pattern_score

    def _detect_placeholders(self, text: str) -> dict[str, Any]:
        """Detect placeholder text in content.

        Args:
            text: Input text to analyze.

        Returns:
            Dictionary with placeholder detection results.
        """
        placeholder_types = []
        total_count = 0

        # Check bracketed placeholders
        bracket_matches = self._BRACKET_PLACEHOLDER_PATTERN.findall(text)
        for match in bracket_matches:
            if match not in self.placeholder_whitelist:
                placeholder_types.append("bracket")
                total_count += 1

        # Check angle bracket placeholders
        angle_matches = self._ANGLE_PLACEHOLDER_PATTERN.findall(text)
        for match in angle_matches:
            if match not in self.placeholder_whitelist:
                placeholder_types.append("angle")
                total_count += 1

        return {
            "has_placeholder": total_count > 0,
            "placeholder_count": total_count,
            "placeholder_types": ",".join(set(placeholder_types))
            if placeholder_types
            else "",
        }

    def _detect_hallucinated_experience(self, text: str) -> bool:
        """Detect AI hallucinated personal experiences.

        Args:
            text: Input text to analyze.

        Returns:
            True if hallucinated experience patterns are detected.
        """
        for pattern in self._HALLUCINATED_EXPERIENCE_PATTERNS:
            if pattern.search(text):
                return True
        return False

    def _detect_nooutput(self, text: str) -> bool:
        """Detect nooutput/NA markers.

        Args:
            text: Input text to analyze.

        Returns:
            True if nooutput patterns are detected.
        """
        text_stripped = text.strip()

        # Check if entire response is very short and matches nooutput
        if len(text_stripped) < 50:
            for pattern in self._NOOUTPUT_PATTERNS:
                if pattern.search(text_stripped):
                    return True

        # Also check for nooutput tags anywhere in text
        if re.search(r"<\s*nooutput\s*>", text, re.IGNORECASE):
            return True

        return False

    def _detect_refusal(self, text: str) -> bool:
        """Detect AI refusal patterns.

        Args:
            text: Input text to analyze.

        Returns:
            True if refusal patterns are detected.
        """
        for pattern in self._REFUSAL_PATTERNS:
            if pattern.search(text):
                return True
        return False

    def _detect_suspicious_urls(self, text: str) -> dict[str, Any]:
        """Detect potentially fake/hallucinated URLs.

        Args:
            text: Input text to analyze.

        Returns:
            Dictionary with URL detection results.
        """
        urls = self._SUSPICIOUS_URL_PATTERN.findall(text)
        suspicious_count = 0

        for url in urls:
            # Extract domain from URL
            domain_match = re.search(
                r"https?://(?:www\.)?([a-zA-Z0-9-]+\.[a-zA-Z]{2,})", url
            )
            if domain_match:
                domain = domain_match.group(1).lower()
                if domain in self._HALLUCINATED_DOMAINS:
                    suspicious_count += 1

        return {
            "has_suspicious_url": suspicious_count > 0,
            "suspicious_url_count": suspicious_count,
        }

    def _compute_content_pattern_score(
        self,
        has_placeholder: bool,
        placeholder_count: int,
        has_hallucinated_experience: bool,
        has_nooutput: bool,
        has_refusal: bool,
        has_suspicious_url: bool,
    ) -> float:
        """Compute a composite content quality score.

        Higher scores indicate cleaner content (fewer issues).

        Args:
            has_placeholder: Whether placeholders were detected.
            placeholder_count: Number of placeholders found.
            has_hallucinated_experience: Whether AI fabrication detected.
            has_nooutput: Whether nooutput marker detected.
            has_refusal: Whether AI refusal detected.
            has_suspicious_url: Whether suspicious URL detected.

        Returns:
            Quality score between 0 and 1.
        """
        score = 1.0

        # Deductions (severity-weighted)
        if has_nooutput:
            score -= 0.5  # Most severe - unusable sample
        if has_refusal:
            score -= 0.4  # Severe - indicates incomplete response
        if has_placeholder:
            score -= min(0.3, placeholder_count * 0.1)  # Scale by count
        if has_hallucinated_experience:
            score -= 0.2
        if has_suspicious_url:
            score -= 0.1

        return max(0.0, round(score, 3))

    def _analyze_text(self, text: str) -> dict[str, Any]:
        """Analyze a single text sample for content pattern issues.

        Args:
            text: Input text to analyze.

        Returns:
            Dictionary of content pattern metrics.
        """
        results = {}

        # Placeholder detection
        has_placeholder = False
        placeholder_count = 0
        if self.detect_placeholders:
            placeholder_results = self._detect_placeholders(text)
            results.update(placeholder_results)
            has_placeholder = placeholder_results["has_placeholder"]
            placeholder_count = placeholder_results["placeholder_count"]

        # Hallucinated experience detection
        has_hallucinated_experience = False
        if self.detect_hallucinated_experiences:
            has_hallucinated_experience = self._detect_hallucinated_experience(text)
            results["has_hallucinated_experience"] = has_hallucinated_experience

        # Nooutput detection
        has_nooutput = False
        if self.detect_nooutput:
            has_nooutput = self._detect_nooutput(text)
            results["has_nooutput"] = has_nooutput

        # Refusal detection
        has_refusal = False
        if self.detect_refusals:
            has_refusal = self._detect_refusal(text)
            results["has_refusal"] = has_refusal

        # Suspicious URL detection
        has_suspicious_url = False
        if self.detect_suspicious_urls:
            url_results = self._detect_suspicious_urls(text)
            results.update(url_results)
            has_suspicious_url = url_results["has_suspicious_url"]

        # Composite score
        if self.compute_content_pattern_score:
            results["content_pattern_score"] = self._compute_content_pattern_score(
                has_placeholder=has_placeholder,
                placeholder_count=placeholder_count,
                has_hallucinated_experience=has_hallucinated_experience,
                has_nooutput=has_nooutput,
                has_refusal=has_refusal,
                has_suspicious_url=has_suspicious_url,
            )

        return results

    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Analyze text fields for content pattern issues.

        Args:
            df: Input DataFrame with text fields.
            schema: Column schema dict to identify text fields.

        Returns:
            DataFrame with added content pattern analysis columns.
        """
        result_df = df.copy()

        if not schema:
            raise ValueError(
                "schema is required to identify text fields for content pattern "
                "analysis. Please provide a column schema dict that specifies "
                "which columns contain text content."
            )

        text_columns = [
            col
            for col, config in schema.items()
            if config.get("content_type") == ContentType.TEXT and col in df.columns
        ]

        if not text_columns:
            return result_df

        # Find the role column if we need to filter by role
        role_column = None
        if self.check_output_only:
            for col, config in schema.items():
                if (
                    config.get("content_type") == ContentType.CATEGORICAL
                    and col in df.columns
                    and "role" in col.lower()
                ):
                    role_column = col
                    break

        analyzer_id = getattr(self, "analyzer_id", "content_pattern")

        for column in text_columns:
            # Analyze all texts in the column
            if self.check_output_only and role_column is not None:
                # Only analyze assistant/output messages
                analysis_results = df.apply(
                    lambda row: (
                        self._analyze_text(str(row[column]))
                        if str(row.get(role_column, "")).lower() == "assistant"
                        else {}
                    ),
                    axis=1,
                )
            else:
                # Analyze all messages
                analysis_results = df[column].astype(str).apply(self._analyze_text)

            # Add columns for each metric
            if self.detect_placeholders:
                result_df[f"{column}_{analyzer_id}_has_placeholder"] = (
                    analysis_results.apply(lambda r: r.get("has_placeholder", None))
                )
                result_df[f"{column}_{analyzer_id}_placeholder_count"] = (
                    analysis_results.apply(lambda r: r.get("placeholder_count", None))
                )
                result_df[f"{column}_{analyzer_id}_placeholder_types"] = (
                    analysis_results.apply(lambda r: r.get("placeholder_types", None))
                )

            if self.detect_hallucinated_experiences:
                result_df[f"{column}_{analyzer_id}_has_hallucinated_experience"] = (
                    analysis_results.apply(
                        lambda r: r.get("has_hallucinated_experience", None)
                    )
                )

            if self.detect_nooutput:
                result_df[f"{column}_{analyzer_id}_has_nooutput"] = (
                    analysis_results.apply(lambda r: r.get("has_nooutput", None))
                )

            if self.detect_refusals:
                result_df[f"{column}_{analyzer_id}_has_refusal"] = (
                    analysis_results.apply(lambda r: r.get("has_refusal", None))
                )

            if self.detect_suspicious_urls:
                result_df[f"{column}_{analyzer_id}_has_suspicious_url"] = (
                    analysis_results.apply(lambda r: r.get("has_suspicious_url", None))
                )
                result_df[f"{column}_{analyzer_id}_suspicious_url_count"] = (
                    analysis_results.apply(
                        lambda r: r.get("suspicious_url_count", None)
                    )
                )

            if self.compute_content_pattern_score:
                result_df[f"{column}_{analyzer_id}_content_pattern_score"] = (
                    analysis_results.apply(
                        lambda r: r.get("content_pattern_score", None)
                    )
                )

        return result_df
