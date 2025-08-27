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

"""Utilities for importing conversations from various formats."""

import csv
import json
import re
from pathlib import Path
from typing import Any

from oumi.core.commands.command_context import CommandContext


class ImportUtilities:
    """Utility class for importing conversations from different formats."""

    def __init__(self, context: CommandContext):
        """Initialize import utilities.

        Args:
            context: Shared command context.
        """
        self.context = context
        self.console = context.console

    def import_conversation(
        self, file_path: str
    ) -> tuple[bool, str, list[dict[str, Any]]]:
        """Import conversation from the specified file.

        Args:
            file_path: Path to the file to import.

        Returns:
            Tuple of (success, message, imported_messages).
        """
        try:
            path_obj = Path(file_path)

            if not path_obj.exists():
                return False, f"File not found: {file_path}", []

            # Determine format from extension
            extension = path_obj.suffix.lower()

            if extension == ".json":
                return self._import_from_json(file_path)
            elif extension == ".csv":
                return self._import_from_csv(file_path)
            elif extension in [".xlsx", ".xls"]:
                return self._import_from_excel(file_path)
            elif extension == ".md":
                return self._import_from_markdown(file_path)
            elif extension == ".txt":
                return self._import_from_text(file_path)
            else:
                return False, f"Unsupported file format: {extension}", []

        except Exception as e:
            return False, f"Import failed: {str(e)}", []

    def _import_from_json(
        self, file_path: str
    ) -> tuple[bool, str, list[dict[str, Any]]]:
        """Import conversation from JSON format."""
        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)

            # Handle different JSON structures
            messages = []

            if isinstance(data, list):
                # Direct list of messages
                messages = data
            elif isinstance(data, dict):
                # Check for common structures
                if "conversation" in data:
                    messages = data["conversation"]
                elif "messages" in data:
                    messages = data["messages"]
                elif "conversation_history" in data:
                    messages = data["conversation_history"]
                elif "history" in data:
                    messages = data["history"]
                else:
                    # Assume the dict itself is a single message or contains message fields
                    if "role" in data and "content" in data:
                        messages = [data]
                    else:
                        return (
                            False,
                            "Unable to find conversation data in JSON file",
                            [],
                        )
            else:
                return False, "Invalid JSON structure for conversation import", []

            # Validate message format
            valid_messages = []
            for i, msg in enumerate(messages):
                if not isinstance(msg, dict):
                    continue

                # Ensure required fields
                if "role" not in msg or "content" not in msg:
                    continue

                # Validate role
                role = str(msg["role"]).lower()
                if role not in ["user", "assistant", "system"]:
                    continue

                valid_messages.append({"role": role, "content": str(msg["content"])})

            if not valid_messages:
                return False, "No valid messages found in JSON file", []

            return (
                True,
                f"Successfully imported {len(valid_messages)} messages from JSON",
                valid_messages,
            )

        except json.JSONDecodeError as e:
            return False, f"Invalid JSON format: {str(e)}", []
        except Exception as e:
            return False, f"Error reading JSON file: {str(e)}", []

    def _import_from_csv(
        self, file_path: str
    ) -> tuple[bool, str, list[dict[str, Any]]]:
        """Import conversation from CSV format."""
        try:
            messages = []

            with open(file_path, encoding="utf-8", newline="") as f:
                # Try to detect delimiter
                sample = f.read(1024)
                f.seek(0)

                delimiter = ","
                if "\\t" in sample and sample.count("\\t") > sample.count(","):
                    delimiter = "\\t"

                reader = csv.DictReader(f, delimiter=delimiter)

                # Look for common column names
                fieldnames = [name.lower().strip() for name in reader.fieldnames or []]

                # Map common column variations
                role_col = None
                content_col = None

                for field in fieldnames:
                    if field in ["role", "speaker", "author", "from", "type"]:
                        role_col = field
                    elif field in ["content", "message", "text", "body", "response"]:
                        content_col = field

                if not role_col or not content_col:
                    return (
                        False,
                        "CSV must contain 'role' and 'content' columns (or similar)",
                        [],
                    )

                for row in reader:
                    role = str(row.get(role_col, "")).lower().strip()
                    content = str(row.get(content_col, "")).strip()

                    if not role or not content:
                        continue

                    # Normalize role names
                    if role in ["user", "human", "person"]:
                        role = "user"
                    elif role in ["assistant", "ai", "bot", "model"]:
                        role = "assistant"
                    elif role in ["system"]:
                        role = "system"
                    else:
                        continue  # Skip unknown roles

                    messages.append({"role": role, "content": content})

            if not messages:
                return False, "No valid messages found in CSV file", []

            return (
                True,
                f"Successfully imported {len(messages)} messages from CSV",
                messages,
            )

        except Exception as e:
            return False, f"Error reading CSV file: {str(e)}", []

    def _import_from_excel(
        self, file_path: str
    ) -> tuple[bool, str, list[dict[str, Any]]]:
        """Import conversation from Excel format."""
        try:
            import pandas as pd
        except ImportError:
            return (
                False,
                "Excel import requires pandas. Install with: pip install pandas openpyxl",
                [],
            )

        try:
            # Read Excel file
            df = pd.read_excel(file_path)

            # Convert to list of dictionaries
            data = df.to_dict("records")

            # Process similar to CSV
            messages = []

            # Look for common column names (case-insensitive)
            columns = [col.lower().strip() for col in df.columns]
            role_col = None
            content_col = None

            for col in df.columns:
                col_lower = col.lower().strip()
                if col_lower in ["role", "speaker", "author", "from", "type"]:
                    role_col = col
                elif col_lower in ["content", "message", "text", "body", "response"]:
                    content_col = col

            if not role_col or not content_col:
                return (
                    False,
                    "Excel file must contain 'role' and 'content' columns (or similar)",
                    [],
                )

            for row in data:
                role = str(row.get(role_col, "")).lower().strip()
                content = str(row.get(content_col, "")).strip()

                if not role or not content or pd.isna(role) or pd.isna(content):
                    continue

                # Normalize role names
                if role in ["user", "human", "person"]:
                    role = "user"
                elif role in ["assistant", "ai", "bot", "model"]:
                    role = "assistant"
                elif role in ["system"]:
                    role = "system"
                else:
                    continue  # Skip unknown roles

                messages.append({"role": role, "content": content})

            if not messages:
                return False, "No valid messages found in Excel file", []

            return (
                True,
                f"Successfully imported {len(messages)} messages from Excel",
                messages,
            )

        except Exception as e:
            return False, f"Error reading Excel file: {str(e)}", []

    def _import_from_markdown(
        self, file_path: str
    ) -> tuple[bool, str, list[dict[str, Any]]]:
        """Import conversation from Markdown format."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            messages = []

            # Split by headers (## Role or # Role)
            parts = re.split(r"^(#+)\\s*([^\\n]+)", content, flags=re.MULTILINE)

            # Process parts
            current_role = None
            current_content = []

            for i in range(
                1, len(parts), 3
            ):  # Skip first empty part, then take header+content pairs
                header_level = parts[i] if i < len(parts) else ""
                header_text = parts[i + 1] if i + 1 < len(parts) else ""
                section_content = parts[i + 2] if i + 2 < len(parts) else ""

                # Extract role from header
                role_match = re.search(
                    r"(user|assistant|system|human|ai|bot)", header_text.lower()
                )
                if role_match:
                    # Save previous message if exists
                    if current_role and current_content:
                        messages.append(
                            {
                                "role": current_role,
                                "content": "\\n".join(current_content).strip(),
                            }
                        )

                    # Start new message
                    role = role_match.group(1)
                    if role in ["human"]:
                        role = "user"
                    elif role in ["ai", "bot"]:
                        role = "assistant"

                    current_role = role
                    current_content = (
                        [section_content.strip()] if section_content.strip() else []
                    )
                elif current_role:
                    # Continue current message
                    if section_content.strip():
                        current_content.append(section_content.strip())

            # Save last message
            if current_role and current_content:
                messages.append(
                    {
                        "role": current_role,
                        "content": "\\n".join(current_content).strip(),
                    }
                )

            if not messages:
                return False, "No valid messages found in Markdown file", []

            return (
                True,
                f"Successfully imported {len(messages)} messages from Markdown",
                messages,
            )

        except Exception as e:
            return False, f"Error reading Markdown file: {str(e)}", []

    def _import_from_text(
        self, file_path: str
    ) -> tuple[bool, str, list[dict[str, Any]]]:
        """Import conversation from plain text format."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            messages = []

            # Try different text formats

            # Format 1: "Role: content" pattern
            role_pattern = r"^(User|Assistant|System|Human|AI|Bot):\\s*(.+?)(?=^(?:User|Assistant|System|Human|AI|Bot):|$)"
            matches = re.findall(
                role_pattern, content, re.MULTILINE | re.DOTALL | re.IGNORECASE
            )

            if matches:
                for role, message_content in matches:
                    role = role.lower()
                    if role in ["human"]:
                        role = "user"
                    elif role in ["ai", "bot"]:
                        role = "assistant"

                    if role in ["user", "assistant", "system"]:
                        messages.append(
                            {"role": role, "content": message_content.strip()}
                        )
            else:
                # Format 2: Try splitting by common separators
                separators = ["=" * 40, "-" * 40, "=" * 20, "-" * 20]
                parts = [content]  # Start with whole content

                for sep in separators:
                    if sep in content:
                        parts = content.split(sep)
                        break

                # Look for role indicators in each part
                for part in parts:
                    part = part.strip()
                    if not part:
                        continue

                    # Look for role at start of part
                    role_match = re.match(
                        r"^(user|assistant|system|human|ai|bot)[:\\s]+(.+)",
                        part,
                        re.IGNORECASE | re.DOTALL,
                    )
                    if role_match:
                        role = role_match.group(1).lower()
                        content_text = role_match.group(2).strip()

                        if role in ["human"]:
                            role = "user"
                        elif role in ["ai", "bot"]:
                            role = "assistant"

                        if role in ["user", "assistant", "system"] and content_text:
                            messages.append({"role": role, "content": content_text})

            if not messages:
                return False, "No valid conversation structure found in text file", []

            return (
                True,
                f"Successfully imported {len(messages)} messages from text",
                messages,
            )

        except Exception as e:
            return False, f"Error reading text file: {str(e)}", []
