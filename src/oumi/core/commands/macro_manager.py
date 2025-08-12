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

"""Macro system for loading and executing Jinja template-based conversation macros."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:
    import jinja2
    from jinja2 import Environment, FileSystemLoader, Template, meta

    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False


@dataclass
class MacroField:
    """Represents a field that needs to be filled in a macro."""

    name: str
    description: Optional[str] = None
    placeholder: Optional[str] = None
    required: bool = True


@dataclass
class MacroInfo:
    """Information about a loaded macro."""

    name: str
    description: str
    path: Path
    fields: list[MacroField]
    turns: int
    template: Template


class MacroManager:
    """Manages macro loading, validation, and execution."""

    def __init__(self, macro_directories: Optional[list[Path]] = None):
        """Initialize macro manager.

        Args:
            macro_directories: List of directories to search for macros.
                             If None, uses default paths.
        """
        if not JINJA2_AVAILABLE:
            raise ImportError(
                "Jinja2 is required for macro functionality. "
                "Install with: pip install jinja2"
            )

        self.macro_directories = macro_directories or self._get_default_directories()
        self.jinja_env = Environment(
            loader=FileSystemLoader([str(d) for d in self.macro_directories]),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def _get_default_directories(self) -> list[Path]:
        """Get default macro directories."""
        # Look for macros in project root and user home
        directories = []

        # Project macros directory (4 levels up from macro_manager.py to project root)
        project_macros = Path(__file__).parent.parent.parent.parent / "macros"
        if project_macros.exists():
            directories.append(project_macros)

        # Also try relative to current working directory
        cwd_macros = Path.cwd() / "macros"
        if cwd_macros.exists() and cwd_macros not in directories:
            directories.append(cwd_macros)

        # User macros directory
        user_macros = Path.home() / ".oumi" / "macros"
        if user_macros.exists():
            directories.append(user_macros)

        # If no directories found, add expected path for better error messages
        if not directories:
            directories.append(project_macros)

        return directories

    def validate_macro_path(self, macro_path: str) -> tuple[bool, str, Optional[Path]]:
        """Validate that a macro path exists and is accessible.

        Args:
            macro_path: Path to the macro template file.

        Returns:
            Tuple of (is_valid, error_message, resolved_path).
        """
        # Handle both absolute and relative paths
        if Path(macro_path).is_absolute():
            full_path = Path(macro_path)
            if not full_path.exists():
                return False, f"Macro file not found: {macro_path}", None
        else:
            # Search in macro directories
            full_path = None
            for directory in self.macro_directories:
                candidate = directory / macro_path
                if candidate.exists():
                    full_path = candidate
                    break

            if full_path is None:
                searched_paths = [str(d / macro_path) for d in self.macro_directories]
                search_info = "\n  ".join(searched_paths)
                return (
                    False,
                    f"Macro not found in search paths:\n  {search_info}",
                    None,
                )

        # Validate file extension
        if full_path.suffix.lower() not in [".jinja", ".j2", ".jinja2"]:
            return (
                False,
                f"Invalid macro file extension. Expected .jinja, .j2, or .jinja2, "
                f"got {full_path.suffix}",
                None,
            )

        # Check if file is readable
        try:
            full_path.read_text(encoding="utf-8")
        except (PermissionError, UnicodeDecodeError) as e:
            return False, f"Cannot read macro file: {e}", None

        return True, "", full_path

    def load_macro(self, macro_path: str) -> tuple[bool, str, Optional[MacroInfo]]:
        """Load and parse a macro template.

        Args:
            macro_path: Path to the macro template file.

        Returns:
            Tuple of (success, error_message, macro_info).
        """
        # Validate path
        is_valid, error, full_path = self.validate_macro_path(macro_path)
        if not is_valid:
            return False, error, None

        try:
            # Load template content
            content = full_path.read_text(encoding="utf-8")

            # Parse template metadata (from header comments)
            metadata = self._parse_macro_metadata(content)

            # Create Jinja template
            template = self.jinja_env.from_string(content)

            # Extract template variables
            ast = self.jinja_env.parse(content)
            template_vars = meta.find_undeclared_variables(ast)

            # Create field definitions
            fields = []
            for var_name in sorted(template_vars):
                field_info = metadata.get("fields", {}).get(var_name, {})
                field = MacroField(
                    name=var_name,
                    description=field_info.get("description"),
                    placeholder=field_info.get("placeholder"),
                    required=field_info.get("required", True),
                )
                fields.append(field)

            # Count conversation turns (rough estimate)
            turns = self._estimate_conversation_turns(content)

            macro_info = MacroInfo(
                name=metadata.get("name", full_path.stem),
                description=metadata.get("description", "No description available"),
                path=full_path,
                fields=fields,
                turns=turns,
                template=template,
            )

            return True, "", macro_info

        except jinja2.TemplateSyntaxError as e:
            return False, f"Template syntax error: {e}", None
        except Exception as e:
            return False, f"Error loading macro: {e}", None

    def _parse_macro_metadata(self, content: str) -> dict[str, Any]:
        """Parse metadata from macro template header comments.

        Args:
            content: Template content.

        Returns:
            Dictionary of parsed metadata.
        """
        metadata = {}

        # Look for JSON metadata block in comments
        json_match = re.search(r"{#\s*METADATA\s*:\s*({.*?})\s*#}", content, re.DOTALL)
        if json_match:
            try:
                metadata = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Look for individual metadata fields in comments
        for line in content.split("\n")[:20]:  # Only check first 20 lines
            line = line.strip()
            if line.startswith("{#") and line.endswith("#}"):
                comment_content = line[2:-2].strip()

                # Parse key: value pairs
                if ":" in comment_content:
                    key, value = comment_content.split(":", 1)
                    key = key.strip().lower()
                    value = value.strip()

                    if key in ["name", "description"]:
                        metadata[key] = value

        return metadata

    def _estimate_conversation_turns(self, content: str) -> int:
        """Estimate the number of conversation turns in a template.

        Args:
            content: Template content.

        Returns:
            Estimated number of conversation turns.
        """
        # Count role indicators
        role_patterns = [
            r'"role"\s*:\s*["\'](?:user|assistant|system)["\']',
            r'role\s*=\s*["\'](?:user|assistant|system)["\']',
            r"<\|(?:user|assistant|system)\|>",
            r"Human:|Assistant:|System:",
        ]

        total_roles = 0
        for pattern in role_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            total_roles += len(matches)

        # Estimate turns (each turn is typically user + assistant)
        return max(1, (total_roles + 1) // 2)

    def render_macro(self, macro_info: MacroInfo, field_values: dict[str, str]) -> str:
        """Render a macro template with provided field values.

        Args:
            macro_info: Loaded macro information.
            field_values: Values for template fields.

        Returns:
            Rendered template content.
        """
        try:
            return macro_info.template.render(**field_values)
        except jinja2.TemplateError as e:
            raise ValueError(f"Template rendering error: {e}")
