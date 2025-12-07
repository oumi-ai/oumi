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

"""FastText-based analyzer for language detection and text classification.

This analyzer uses FastText models for fast, accurate language detection
and optional text classification capabilities.
"""

import re
from typing import Any, Optional

import pandas as pd

from oumi.core.analyze.column_types import ContentType
from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.registry import register_sample_analyzer
from oumi.utils.logging import logger


@register_sample_analyzer("fasttext")
class FastTextAnalyzer(SampleAnalyzer):
    """Analyzer using FastText for language detection and text analysis.

    This analyzer provides fast, accurate language detection using FastText
    models. It's significantly faster than alternatives like langdetect while
    maintaining high accuracy (up to 95%).

    Features:
        - Language detection for 176+ languages
        - Language confidence scoring
        - Multi-language detection for mixed content
        - Script detection (Latin, Cyrillic, CJK, Arabic, etc.)
        - Offline operation (no API calls required)

    Output metrics:
        - detected_language: ISO 639-1 language code (e.g., "en", "es", "zh")
        - language_confidence: Confidence score (0-1)
        - is_multilingual: Whether text contains multiple languages
        - detected_script: Primary script used (latin, cyrillic, cjk, etc.)
        - language_name: Human-readable language name

    Note:
        Requires fast-langdetect package: pip install fast-langdetect
        Or full fasttext: pip install fasttext

    Example:
        >>> analyzer = FastTextAnalyzer(
        ...     detect_language=True,
        ...     detect_script=True,
        ...     min_confidence=0.5,
        ... )
        >>> result = analyzer.analyze_sample(df, schema)
    """

    # Script detection patterns
    _SCRIPT_PATTERNS = {
        "latin": re.compile(r"[a-zA-ZÀ-ÿ]"),
        "cyrillic": re.compile(r"[\u0400-\u04FF]"),
        "greek": re.compile(r"[\u0370-\u03FF]"),
        "arabic": re.compile(r"[\u0600-\u06FF]"),
        "hebrew": re.compile(r"[\u0590-\u05FF]"),
        "devanagari": re.compile(r"[\u0900-\u097F]"),
        "cjk": re.compile(r"[\u4E00-\u9FFF\u3400-\u4DBF]"),
        "hiragana": re.compile(r"[\u3040-\u309F]"),
        "katakana": re.compile(r"[\u30A0-\u30FF]"),
        "hangul": re.compile(r"[\uAC00-\uD7AF\u1100-\u11FF]"),
        "thai": re.compile(r"[\u0E00-\u0E7F]"),
    }

    # Language code to name mapping (common languages)
    _LANGUAGE_NAMES = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
        "ru": "Russian",
        "zh": "Chinese",
        "ja": "Japanese",
        "ko": "Korean",
        "ar": "Arabic",
        "hi": "Hindi",
        "nl": "Dutch",
        "pl": "Polish",
        "tr": "Turkish",
        "vi": "Vietnamese",
        "th": "Thai",
        "id": "Indonesian",
        "cs": "Czech",
        "sv": "Swedish",
        "da": "Danish",
        "fi": "Finnish",
        "no": "Norwegian",
        "uk": "Ukrainian",
        "el": "Greek",
        "he": "Hebrew",
        "ro": "Romanian",
        "hu": "Hungarian",
        "bg": "Bulgarian",
        "sk": "Slovak",
        "hr": "Croatian",
        "sr": "Serbian",
        "sl": "Slovenian",
        "lt": "Lithuanian",
        "lv": "Latvian",
        "et": "Estonian",
        "fa": "Persian",
        "bn": "Bengali",
        "ta": "Tamil",
        "te": "Telugu",
        "ml": "Malayalam",
        "mr": "Marathi",
        "gu": "Gujarati",
        "kn": "Kannada",
        "pa": "Punjabi",
        "ur": "Urdu",
        "sw": "Swahili",
        "tl": "Filipino",
        "ms": "Malay",
        "ca": "Catalan",
        "eu": "Basque",
        "gl": "Galician",
        "cy": "Welsh",
        "ga": "Irish",
        "af": "Afrikaans",
        "sq": "Albanian",
        "mk": "Macedonian",
        "be": "Belarusian",
        "ka": "Georgian",
        "hy": "Armenian",
        "az": "Azerbaijani",
        "kk": "Kazakh",
        "uz": "Uzbek",
        "mn": "Mongolian",
        "ne": "Nepali",
        "si": "Sinhala",
        "km": "Khmer",
        "lo": "Lao",
        "my": "Burmese",
    }

    def __init__(
        self,
        *,
        detect_language: bool = True,
        detect_script: bool = True,
        detect_multilingual: bool = True,
        min_confidence: float = 0.0,
        low_confidence_threshold: float = 0.5,
        use_fast_langdetect: bool = True,
    ):
        """Initialize the FastTextAnalyzer.

        Args:
            detect_language: Whether to detect the primary language.
            detect_script: Whether to detect the writing script.
            detect_multilingual: Whether to check for multiple languages.
            min_confidence: Minimum confidence to report a language.
                Languages below this threshold are reported as "unknown".
            low_confidence_threshold: Threshold for flagging low-confidence
                detections. Useful for identifying potentially problematic samples.
            use_fast_langdetect: Use fast-langdetect library (recommended).
                If False, uses the full fasttext library with HuggingFace model.
        """
        self.detect_language = detect_language
        self.detect_script = detect_script
        self.detect_multilingual = detect_multilingual
        self.min_confidence = min_confidence
        self.low_confidence_threshold = low_confidence_threshold
        self.use_fast_langdetect = use_fast_langdetect

        # Lazy-load the model
        self._model = None
        self._detector = None

    def _check_dependencies(self) -> None:
        """Check if required dependencies are installed."""
        if self.use_fast_langdetect:
            try:
                import fast_langdetect  # noqa: F401
            except ImportError:
                raise ImportError(
                    "FastTextAnalyzer requires fast-langdetect. "
                    "Install with: pip install fast-langdetect"
                )
        else:
            try:
                import fasttext  # noqa: F401
            except ImportError:
                raise ImportError(
                    "FastTextAnalyzer requires fasttext. "
                    "Install with: pip install fasttext-wheel"
                )

    def _init_detector(self) -> None:
        """Initialize the language detection model."""
        if self._detector is not None:
            return

        self._check_dependencies()

        if self.use_fast_langdetect:
            from fast_langdetect import LangDetector

            self._detector = LangDetector()
            logger.info("Initialized fast-langdetect for language detection")
        else:
            import fasttext
            from huggingface_hub import hf_hub_download

            model_path = hf_hub_download(
                repo_id="facebook/fasttext-language-identification",
                filename="model.bin",
            )
            self._model = fasttext.load_model(model_path)
            logger.info("Loaded fasttext-language-identification model")

    def _detect_language_fast(self, text: str) -> tuple[str, float]:
        """Detect language using fast-langdetect.

        Args:
            text: Text to analyze.

        Returns:
            Tuple of (language_code, confidence).
        """
        if not text or len(text.strip()) < 3:
            return "unknown", 0.0

        try:
            result = self._detector.detect(text)
            lang_code = result.lang
            confidence = result.score

            # Normalize language codes (remove script suffixes like "zh-Hans")
            if "-" in lang_code:
                lang_code = lang_code.split("-")[0]

            return lang_code, confidence
        except Exception:
            return "unknown", 0.0

    def _detect_language_fasttext(self, text: str) -> tuple[str, float]:
        """Detect language using full fasttext model.

        Args:
            text: Text to analyze.

        Returns:
            Tuple of (language_code, confidence).
        """
        if not text or len(text.strip()) < 3:
            return "unknown", 0.0

        try:
            # Clean text (fasttext expects single line)
            clean_text = " ".join(text.split())
            predictions = self._model.predict(clean_text, k=1)
            label = predictions[0][0]  # e.g., "__label__en"
            confidence = float(predictions[1][0])

            # Extract language code from label
            lang_code = label.replace("__label__", "")

            return lang_code, confidence
        except Exception:
            return "unknown", 0.0

    def _detect_language(self, text: str) -> tuple[str, float]:
        """Detect language using the configured method.

        Args:
            text: Text to analyze.

        Returns:
            Tuple of (language_code, confidence).
        """
        if self.use_fast_langdetect:
            return self._detect_language_fast(text)
        else:
            return self._detect_language_fasttext(text)

    def _detect_script(self, text: str) -> str:
        """Detect the primary writing script in the text.

        Args:
            text: Text to analyze.

        Returns:
            Primary script name (e.g., "latin", "cyrillic", "cjk").
        """
        if not text:
            return "unknown"

        script_counts: dict[str, int] = {}

        for script_name, pattern in self._SCRIPT_PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                script_counts[script_name] = len(matches)

        if not script_counts:
            return "unknown"

        # Return the script with the most characters
        return max(script_counts, key=script_counts.get)  # type: ignore

    def _detect_multilingual(self, text: str) -> bool:
        """Check if text contains multiple languages.

        Uses a simple heuristic: splits text into chunks and checks
        if different languages are detected in different chunks.

        Args:
            text: Text to analyze.

        Returns:
            True if multiple languages detected.
        """
        if not text or len(text) < 50:
            return False

        # Split into sentences or chunks
        sentences = re.split(r"[.!?\n]+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        if len(sentences) < 2:
            return False

        # Detect language for each chunk
        languages = set()
        for sentence in sentences[:10]:  # Limit to first 10 sentences
            lang, conf = self._detect_language(sentence)
            if conf >= self.low_confidence_threshold and lang != "unknown":
                languages.add(lang)

        return len(languages) > 1

    def _get_language_name(self, lang_code: str) -> str:
        """Get human-readable language name from code.

        Args:
            lang_code: ISO 639-1 language code.

        Returns:
            Human-readable language name.
        """
        return self._LANGUAGE_NAMES.get(lang_code, lang_code.capitalize())

    def _analyze_text(self, text: str) -> dict[str, Any]:
        """Analyze a single text for language and script.

        Args:
            text: Text to analyze.

        Returns:
            Dictionary with analysis results.
        """
        results: dict[str, Any] = {}

        if self.detect_language:
            lang_code, confidence = self._detect_language(text)

            # Apply minimum confidence threshold
            if confidence < self.min_confidence:
                lang_code = "unknown"
                confidence = 0.0

            results["detected_language"] = lang_code
            results["language_confidence"] = round(confidence, 4)
            results["language_name"] = self._get_language_name(lang_code)
            results["low_confidence"] = confidence < self.low_confidence_threshold

        if self.detect_script:
            results["detected_script"] = self._detect_script(text)

        if self.detect_multilingual:
            results["is_multilingual"] = self._detect_multilingual(text)

        return results

    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Analyze text fields for language and script detection.

        Args:
            df: Input DataFrame with text fields.
            schema: Column schema dict to identify text fields.

        Returns:
            DataFrame with added language analysis columns.
        """
        result_df = df.copy()

        if not schema:
            raise ValueError(
                "schema is required to identify text fields for FastText analysis. "
                "Please provide a column schema dict that specifies which "
                "columns contain text content."
            )

        text_columns = [
            col
            for col, config in schema.items()
            if config.get("content_type") == ContentType.TEXT and col in df.columns
        ]

        if not text_columns:
            return result_df

        # Initialize detector
        self._init_detector()

        analyzer_id = getattr(self, "analyzer_id", "fasttext")

        for column in text_columns:
            logger.info(f"Analyzing language for column: {column}")

            # Analyze each text
            analysis_results = df[column].astype(str).apply(self._analyze_text)

            # Extract results into columns
            if self.detect_language:
                result_df[f"{column}_{analyzer_id}_detected_language"] = (
                    analysis_results.apply(lambda r: r.get("detected_language"))
                )
                result_df[f"{column}_{analyzer_id}_language_confidence"] = (
                    analysis_results.apply(lambda r: r.get("language_confidence"))
                )
                result_df[f"{column}_{analyzer_id}_language_name"] = (
                    analysis_results.apply(lambda r: r.get("language_name"))
                )
                result_df[f"{column}_{analyzer_id}_low_confidence"] = (
                    analysis_results.apply(lambda r: r.get("low_confidence"))
                )

            if self.detect_script:
                result_df[f"{column}_{analyzer_id}_detected_script"] = (
                    analysis_results.apply(lambda r: r.get("detected_script"))
                )

            if self.detect_multilingual:
                result_df[f"{column}_{analyzer_id}_is_multilingual"] = (
                    analysis_results.apply(lambda r: r.get("is_multilingual"))
                )

        return result_df

    def compute_dataset_metrics(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Compute dataset-level language metrics.

        Args:
            df: DataFrame with language analysis columns.
            schema: Column schema dict.

        Returns:
            Dictionary with dataset-level language metrics.
        """
        metrics: dict[str, Any] = {}

        # Find language columns
        lang_cols = [col for col in df.columns if "_detected_language" in col]

        if not lang_cols:
            return metrics

        lang_col = lang_cols[0]
        conf_col = lang_col.replace("_detected_language", "_language_confidence")

        # Language distribution
        lang_counts = df[lang_col].value_counts().to_dict()
        metrics["language_distribution"] = lang_counts
        metrics["num_languages"] = len([l for l in lang_counts if l != "unknown"])
        metrics["primary_language"] = df[lang_col].mode().iloc[0] if len(df) > 0 else "unknown"

        # Confidence statistics
        if conf_col in df.columns:
            conf_values = df[conf_col].dropna()
            if len(conf_values) > 0:
                metrics["avg_language_confidence"] = round(conf_values.mean(), 4)
                metrics["low_confidence_ratio"] = round(
                    (conf_values < self.low_confidence_threshold).mean(), 4
                )

        # Multilingual ratio
        multi_cols = [col for col in df.columns if "_is_multilingual" in col]
        if multi_cols:
            multi_col = multi_cols[0]
            metrics["multilingual_ratio"] = round(df[multi_col].mean(), 4)

        return metrics
