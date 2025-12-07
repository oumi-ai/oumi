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

"""Data analysis utilities for customer onboarding.

This module provides tools to detect, parse, and analyze customer data
formats to enable automatic configuration generation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

FormatType = Literal["csv", "excel", "json", "jsonl", "word", "unknown"]


@dataclass
class ColumnInfo:
    """Information about a single column in the data."""

    name: str
    dtype: str
    sample_values: list[Any] = field(default_factory=list)
    unique_count: int = 0
    null_count: int = 0
    avg_length: Optional[float] = None
    is_conversation: bool = False
    is_text: bool = False
    is_categorical: bool = False


@dataclass
class DataSchema:
    """Inferred schema from customer data."""

    columns: list[ColumnInfo] = field(default_factory=list)
    row_count: int = 0
    sample_rows: list[dict[str, Any]] = field(default_factory=list)
    detected_format: FormatType = "unknown"
    source_path: str = ""

    # Convenience properties for quick access
    conversation_columns: list[str] = field(default_factory=list)
    text_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)

    # For Word documents
    raw_text: Optional[str] = None


class DataAnalyzer:
    """Analyze customer data to infer schema and suggest configurations.

    This class detects the format of customer data files and extracts
    schema information that can be used to generate Oumi configurations.

    Example:
        >>> analyzer = DataAnalyzer()
        >>> schema = analyzer.analyze("./customer_data.csv")
        >>> print(f"Found {schema.row_count} rows with {len(schema.columns)} columns")
        >>> for col in schema.columns:
        ...     print(f"  {col.name}: {col.dtype}")
    """

    # Pattern to detect conversation-like JSON structures
    CONVERSATION_PATTERNS = [
        r'\{"role":\s*"[^"]+",\s*"content"',
        r'\[\s*\{"role"',
        r"messages.*\[.*role",
    ]

    # Column names that typically indicate specific data types
    COLUMN_NAME_HINTS = {
        "conversation": ["conversation", "chat", "dialogue", "messages", "turns"],
        "question": ["question", "query", "prompt", "input", "request"],
        "answer": ["answer", "response", "output", "reply", "completion"],
        "context": ["context", "document", "passage", "text", "content"],
        "system": ["system", "instruction", "persona", "role"],
    }

    def __init__(
        self,
        max_sample_rows: int = 5,
        text_length_threshold: int = 100,
        categorical_threshold: float = 0.1,
    ):
        """Initialize the DataAnalyzer.

        Args:
            max_sample_rows: Maximum number of sample rows to store.
            text_length_threshold: Minimum average length to consider a column as text.
            categorical_threshold: Max ratio of unique values to consider categorical.
        """
        self.max_sample_rows = max_sample_rows
        self.text_length_threshold = text_length_threshold
        self.categorical_threshold = categorical_threshold

    def analyze(self, path: str | Path) -> DataSchema:
        """Analyze a data file and return its schema.

        Args:
            path: Path to the data file.

        Returns:
            DataSchema with inferred information about the data.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is not supported.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        format_type = self._detect_format(path)

        if format_type == "csv":
            return self._analyze_csv(path)
        elif format_type == "excel":
            return self._analyze_excel(path)
        elif format_type == "json":
            return self._analyze_json(path)
        elif format_type == "jsonl":
            return self._analyze_jsonl(path)
        elif format_type == "word":
            return self._analyze_word(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def _detect_format(self, path: Path) -> FormatType:
        """Detect the format of a file based on extension and content."""
        suffix = path.suffix.lower()

        format_map: dict[str, FormatType] = {
            ".csv": "csv",
            ".xlsx": "excel",
            ".xls": "excel",
            ".json": "json",
            ".jsonl": "jsonl",
            ".docx": "word",
            ".doc": "word",
        }

        if suffix in format_map:
            # For JSON files, check if it's actually JSONL
            if suffix == ".json":
                return self._check_json_vs_jsonl(path)
            return format_map[suffix]

        return "unknown"

    def _check_json_vs_jsonl(self, path: Path) -> FormatType:
        """Check if a .json file is actually JSONL format."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                second_line = f.readline().strip()

                # If both lines are valid JSON objects, it's JSONL
                if first_line and second_line:
                    try:
                        json.loads(first_line)
                        json.loads(second_line)
                        return "jsonl"
                    except json.JSONDecodeError:
                        pass

                # Otherwise, try parsing as regular JSON
                f.seek(0)
                json.load(f)
                return "json"
        except Exception:
            return "json"

    def _analyze_csv(self, path: Path) -> DataSchema:
        """Analyze a CSV file."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for CSV analysis. pip install pandas")

        # Try different encodings
        encodings = ["utf-8", "latin-1", "cp1252"]
        df = None

        for encoding in encodings:
            try:
                df = pd.read_csv(path, encoding=encoding, nrows=1000)
                break
            except UnicodeDecodeError:
                continue

        if df is None:
            raise ValueError(f"Could not read CSV file with any supported encoding")

        return self._analyze_dataframe(df, path, "csv")

    def _analyze_excel(self, path: Path) -> DataSchema:
        """Analyze an Excel file."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for Excel analysis. pip install pandas"
            )

        try:
            df = pd.read_excel(path, nrows=1000)
        except ImportError:
            raise ImportError(
                "openpyxl is required for Excel analysis. pip install openpyxl"
            )

        return self._analyze_dataframe(df, path, "excel")

    def _analyze_dataframe(
        self, df: "pd.DataFrame", path: Path, format_type: FormatType
    ) -> DataSchema:
        """Analyze a pandas DataFrame."""
        columns = []
        conversation_cols = []
        text_cols = []
        categorical_cols = []

        for col_name in df.columns:
            col_data = df[col_name]
            col_info = self._analyze_column(col_name, col_data, len(df))

            columns.append(col_info)

            if col_info.is_conversation:
                conversation_cols.append(col_name)
            if col_info.is_text:
                text_cols.append(col_name)
            if col_info.is_categorical:
                categorical_cols.append(col_name)

        # Get sample rows
        sample_rows = df.head(self.max_sample_rows).to_dict("records")

        return DataSchema(
            columns=columns,
            row_count=len(df),
            sample_rows=sample_rows,
            detected_format=format_type,
            source_path=str(path),
            conversation_columns=conversation_cols,
            text_columns=text_cols,
            categorical_columns=categorical_cols,
        )

    def _analyze_column(
        self, col_name: str, col_data: "pd.Series", total_rows: int
    ) -> ColumnInfo:
        """Analyze a single column."""
        import pandas as pd

        # Basic stats
        dtype = str(col_data.dtype)
        unique_count = col_data.nunique()
        null_count = col_data.isna().sum()

        # Sample values (non-null)
        non_null = col_data.dropna()
        sample_values = non_null.head(3).tolist() if len(non_null) > 0 else []

        # Calculate average length for string columns
        avg_length = None
        is_text = False
        is_conversation = False

        if col_data.dtype == object or pd.api.types.is_string_dtype(col_data):
            str_lengths = non_null.astype(str).str.len()
            if len(str_lengths) > 0:
                avg_length = float(str_lengths.mean())
                is_text = avg_length >= self.text_length_threshold

            # Check for conversation patterns
            for val in non_null.head(10):
                if self._is_conversation_text(str(val)):
                    is_conversation = True
                    break

        # Check column name hints
        col_lower = col_name.lower()
        for hint_type, hints in self.COLUMN_NAME_HINTS.items():
            if any(hint in col_lower for hint in hints):
                if hint_type == "conversation":
                    is_conversation = True
                elif hint_type in ("question", "answer", "context"):
                    is_text = True

        # Check if categorical
        is_categorical = False
        if total_rows > 0:
            unique_ratio = unique_count / total_rows
            is_categorical = unique_ratio <= self.categorical_threshold and unique_count < 100

        return ColumnInfo(
            name=col_name,
            dtype=dtype,
            sample_values=sample_values,
            unique_count=unique_count,
            null_count=null_count,
            avg_length=avg_length,
            is_conversation=is_conversation,
            is_text=is_text,
            is_categorical=is_categorical,
        )

    def _is_conversation_text(self, text: str) -> bool:
        """Check if text appears to be conversation data."""
        for pattern in self.CONVERSATION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _analyze_json(self, path: Path) -> DataSchema:
        """Analyze a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return self._analyze_json_list(data, path)
        elif isinstance(data, dict):
            return self._analyze_json_dict(data, path)
        else:
            raise ValueError(f"Unexpected JSON structure: {type(data)}")

    def _analyze_json_list(self, data: list, path: Path) -> DataSchema:
        """Analyze a JSON file containing a list of objects."""
        if not data:
            return DataSchema(
                detected_format="json",
                source_path=str(path),
            )

        # Treat as tabular data if list of dicts
        if isinstance(data[0], dict):
            try:
                import pandas as pd

                df = pd.DataFrame(data[:1000])
                return self._analyze_dataframe(df, path, "json")
            except ImportError:
                pass

        # Fallback: just store sample rows
        return DataSchema(
            row_count=len(data),
            sample_rows=data[: self.max_sample_rows],
            detected_format="json",
            source_path=str(path),
        )

    def _analyze_json_dict(self, data: dict, path: Path) -> DataSchema:
        """Analyze a JSON file containing a single object."""
        # This might be a configuration or synthesis spec
        columns = []
        for key, value in data.items():
            col_info = ColumnInfo(
                name=key,
                dtype=type(value).__name__,
                sample_values=[value] if not isinstance(value, (list, dict)) else [],
            )
            columns.append(col_info)

        return DataSchema(
            columns=columns,
            row_count=1,
            sample_rows=[data],
            detected_format="json",
            source_path=str(path),
        )

    def _analyze_jsonl(self, path: Path) -> DataSchema:
        """Analyze a JSONL file."""
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 1000:
                    break
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if data and isinstance(data[0], dict):
            try:
                import pandas as pd

                df = pd.DataFrame(data)
                schema = self._analyze_dataframe(df, path, "jsonl")
                # Count total lines for row_count
                with open(path, "r") as f:
                    schema.row_count = sum(1 for _ in f)
                return schema
            except ImportError:
                pass

        return DataSchema(
            row_count=len(data),
            sample_rows=data[: self.max_sample_rows],
            detected_format="jsonl",
            source_path=str(path),
        )

    def _analyze_word(self, path: Path) -> DataSchema:
        """Analyze a Word document."""
        try:
            from docx import Document
        except ImportError:
            raise ImportError(
                "python-docx is required for Word document analysis. "
                "pip install python-docx"
            )

        doc = Document(path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        full_text = "\n\n".join(paragraphs)

        return DataSchema(
            row_count=len(paragraphs),
            detected_format="word",
            source_path=str(path),
            raw_text=full_text,
        )

    def suggest_goal(self, schema: DataSchema) -> str:
        """Suggest a synthesis goal based on the data schema.

        Args:
            schema: The analyzed data schema.

        Returns:
            Suggested goal: "qa", "conversation", "augmentation", or "instruction".
        """
        # Word documents -> instruction following
        if schema.detected_format == "word":
            return "instruction"

        # Has conversation columns -> conversation augmentation
        if schema.conversation_columns:
            return "conversation"

        # Has question/answer-like columns -> Q&A
        col_names_lower = [c.name.lower() for c in schema.columns]
        qa_indicators = ["question", "answer", "query", "response", "prompt"]
        if any(ind in " ".join(col_names_lower) for ind in qa_indicators):
            return "qa"

        # Has context/document columns -> domain QA
        context_indicators = ["context", "document", "passage"]
        if any(ind in " ".join(col_names_lower) for ind in context_indicators):
            return "qa"

        # Default to augmentation
        return "augmentation"
