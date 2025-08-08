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

"""Thinking content processor for multiple reasoning formats."""

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


@dataclass
class ThinkingContent:
    """Represents extracted thinking content with metadata."""

    format_type: str
    content: str
    compressed: str = ""
    word_count: int = 0
    start_position: int = 0
    end_position: int = 0

    def __post_init__(self):
        """Initialize computed fields."""
        if not self.word_count:
            self.word_count = len(self.content.split())
        if not self.compressed:
            self.compressed = self._create_compressed_view()

    def _create_compressed_view(self, max_length: int = 50) -> str:
        """Create a compressed view of the thinking content."""
        words = self.content.split()
        if len(words) <= 8:
            return self.content

        # Take first few words and add word count
        compressed_words = words[:6]
        compressed_text = " ".join(compressed_words)

        if len(compressed_text) > max_length:
            compressed_text = compressed_text[: max_length - 3] + "..."
        else:
            compressed_text += "..."

        return f"{compressed_text} ({self.word_count} words)"


@dataclass
class ThinkingResult:
    """Complete result of thinking processing."""

    thinking_parts: list[ThinkingContent] = field(default_factory=list)
    final_content: str = ""
    original_content: str = ""
    total_thinking_words: int = 0

    @property
    def has_thinking(self) -> bool:
        """Whether thinking content was found."""
        return len(self.thinking_parts) > 0

    def __post_init__(self):
        """Initialize computed fields."""
        self.total_thinking_words = sum(part.word_count for part in self.thinking_parts)


class ThinkingProcessor:
    """Processor for extracting and rendering thinking content from various formats."""

    # Pattern definitions for different thinking formats
    PATTERNS = {
        "gpt_oss": {
            # Updated pattern to handle full OpenAI Harmony format
            "pattern": r"<\|(?:start\|)?channel\|>(analysis|commentary|final)<\|message\|>(.*?)(?:<\|end\|>|$)",
            "flags": re.DOTALL,
            "groups": {"type": 1, "content": 2},
            "thinking_types": ["analysis", "commentary"],
            "final_types": ["final"],
        },
        "harmony_comprehensive": {
            # Comprehensive pattern for all OpenAI Harmony tags
            "pattern": r"<\|(?:start\|)?(?:channel\|)?(analysis|commentary|final|call|constrain|return)<\|(?:message\|)?(.*?)(?:<\|end\|>|$)",
            "flags": re.DOTALL,
            "groups": {"type": 1, "content": 2},
            "thinking_types": ["analysis", "commentary"],
            "final_types": ["final", "call", "constrain", "return"],
        },
        "simple_think": {
            "pattern": r"<think>(.*?)</think>",
            "flags": re.DOTALL,
            "groups": {"content": 1},
            "thinking_types": ["think"],
            "final_types": [],
        },
        "reasoning": {
            "pattern": r"<reasoning>(.*?)</reasoning>",
            "flags": re.DOTALL,
            "groups": {"content": 1},
            "thinking_types": ["reasoning"],
            "final_types": [],
        },
        "reflection": {
            "pattern": r"<reflection>(.*?)</reflection>",
            "flags": re.DOTALL,
            "groups": {"content": 1},
            "thinking_types": ["reflection"],
            "final_types": [],
        },
        "comment_thinking": {
            "pattern": r"<!-- thinking: (.*?) -->",
            "flags": re.DOTALL,
            "groups": {"content": 1},
            "thinking_types": ["comment"],
            "final_types": [],
        },
        "bracket_thinking": {
            "pattern": r"\[THINKING\](.*?)\[/THINKING\]",
            "flags": re.DOTALL,
            "groups": {"content": 1},
            "thinking_types": ["bracket"],
            "final_types": [],
        },
        "markdown_thinking": {
            "pattern": r"\*\*Thinking:\*\* (.*?)(?=\n\n|\n\*\*|\n[A-Z][a-z]*:|$)",
            "flags": re.DOTALL | re.MULTILINE,
            "groups": {"content": 1},
            "thinking_types": ["markdown"],
            "final_types": [],
        },
    }

    def __init__(self):
        """Initialize the thinking processor."""
        self._compiled_patterns = {}
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex patterns for better performance."""
        for format_name, pattern_info in self.PATTERNS.items():
            self._compiled_patterns[format_name] = re.compile(
                pattern_info["pattern"], pattern_info["flags"]
            )

    def extract_thinking(self, content: str) -> ThinkingResult:
        """Extract thinking content from text using all supported formats.

        Args:
            content: Raw text content to process

        Returns:
            ThinkingResult with extracted thinking parts and final content
        """
        result = ThinkingResult(original_content=content)
        processed_content = content
        all_matches = []

        # Find matches for all patterns
        for format_name, pattern_info in self.PATTERNS.items():
            compiled_pattern = self._compiled_patterns[format_name]
            matches = list(compiled_pattern.finditer(content))

            for match in matches:
                thinking_content = self._extract_thinking_from_match(
                    match, format_name, pattern_info
                )
                if thinking_content:
                    all_matches.append((match, thinking_content))

        # Sort matches by position to process in order
        all_matches.sort(key=lambda x: x[0].start())

        # Process matches and build final content
        last_end = 0
        final_parts = []

        for match, thinking_content in all_matches:
            # Add content before this match to final content
            if match.start() > last_end:
                final_parts.append(content[last_end : match.start()])

            result.thinking_parts.append(thinking_content)
            last_end = match.end()

        # Add any remaining content after last match
        if last_end < len(content):
            final_parts.append(content[last_end:])

        # Clean up and join final content
        result.final_content = "".join(final_parts).strip()

        # If no thinking found but content has special patterns, try fallback
        if not result.has_thinking:
            result.final_content = content

        return result

    def _extract_thinking_from_match(
        self, match: re.Match, format_name: str, pattern_info: dict
    ) -> Optional[ThinkingContent]:
        """Extract thinking content from a regex match."""
        groups = pattern_info["groups"]

        if "type" in groups:
            # Format with type information (like GPT-OSS)
            type_group = groups["type"]
            content_group = groups["content"]

            match_type = match.group(type_group)
            match_content = match.group(content_group)

            if match_type in pattern_info["thinking_types"]:
                return ThinkingContent(
                    format_type=format_name,
                    content=match_content.strip(),
                    start_position=match.start(),
                    end_position=match.end(),
                )
        else:
            # Simple format with just content
            content_group = groups["content"]
            match_content = match.group(content_group)

            return ThinkingContent(
                format_type=format_name,
                content=match_content.strip(),
                start_position=match.start(),
                end_position=match.end(),
            )

        return None

    def render_thinking(
        self,
        thinking_result: ThinkingResult,
        console: Console,
        style_params=None,
        compressed: bool = True,
    ) -> bool:
        """Render thinking content with appropriate formatting.

        Args:
            thinking_result: ThinkingResult to render
            console: Rich console for output
            style_params: Style parameters for formatting
            compressed: Whether to show compressed or full thinking

        Returns:
            True if thinking content was rendered, False otherwise
        """
        if not thinking_result.has_thinking:
            return False

        # Get style settings
        if style_params:
            analysis_text_style = style_params.analysis_text_style
            analysis_title_style = style_params.analysis_title_style
            analysis_border_style = style_params.analysis_border_style
            use_emoji = style_params.use_emoji
            expand_panels = style_params.expand_panels
        else:
            analysis_text_style = "dim cyan"
            analysis_title_style = "bold yellow"
            analysis_border_style = "yellow"
            use_emoji = True
            expand_panels = False

        # Render each thinking part
        for i, thinking_part in enumerate(thinking_result.thinking_parts):
            title = self._get_thinking_title(thinking_part.format_type, use_emoji)

            if compressed:
                content_text = thinking_part.compressed
                border_style = "dim " + analysis_border_style
                text_style = "dim " + analysis_text_style
            else:
                content_text = thinking_part.content
                border_style = analysis_border_style
                text_style = analysis_text_style

            console.print(
                Panel(
                    Text(content_text, style=text_style),
                    title=f"[{analysis_title_style}]{title}[/{analysis_title_style}]",
                    border_style=border_style,
                    padding=(0, 1),
                    expand=expand_panels,
                )
            )

        return True

    def _get_thinking_title(self, format_type: str, use_emoji: bool = True) -> str:
        """Get appropriate title for thinking content based on format."""
        emoji_map = {
            "gpt_oss": "🧠 ",
            "simple_think": "💭 ",
            "reasoning": "🤔 ",
            "reflection": "🪞 ",
            "comment_thinking": "💬 ",
            "bracket_thinking": "📝 ",
            "markdown_thinking": "✍️ ",
        }

        title_map = {
            "gpt_oss": "Analysis",
            "simple_think": "Thinking",
            "reasoning": "Reasoning",
            "reflection": "Reflection",
            "comment_thinking": "Internal Note",
            "bracket_thinking": "Thought Process",
            "markdown_thinking": "Thinking",
        }

        emoji = emoji_map.get(format_type, "🧠 ") if use_emoji else ""
        title = title_map.get(format_type, "Thinking")

        return f"{emoji}{title}"

    def convert_to_harmony_format(self, content: str) -> dict[str, Any]:
        """Convert content to Harmony format with separated thinking and content.

        Maintains backward compatibility with existing Harmony format expectations.

        Args:
            content: Raw content to process

        Returns:
            Dict with 'thinking', 'content', and metadata fields
        """
        result = self.extract_thinking(content)

        harmony_result = {}

        # Combine all thinking content
        if result.has_thinking:
            thinking_parts = []
            for part in result.thinking_parts:
                thinking_parts.append(f"[{part.format_type}] {part.content}")

            harmony_result["thinking"] = "\n\n".join(thinking_parts)
            harmony_result["thinking_metadata"] = {
                "total_words": result.total_thinking_words,
                "formats": [part.format_type for part in result.thinking_parts],
                "compressed_available": True,
            }

        # Set content (cleaned of thinking tags)
        harmony_result["content"] = (
            result.final_content if result.final_content else content
        )

        return harmony_result

    def clean_harmony_tags(self, content: str) -> str:
        """Remove any remaining OpenAI Harmony format tags from content.

        This is a safety net to catch any malformed or incomplete harmony tags
        that weren't processed by the main extraction patterns.

        Args:
            content: Content that may contain harmony tags

        Returns:
            Content with harmony tags removed
        """
        # Remove complete harmony tag patterns first (more specific)
        harmony_patterns = [
            r"<\|channel\|>(analysis|commentary|final)<\|message\|>",
            r"<\|start\|><\|channel\|>(analysis|commentary|final)<\|message\|>",
            r"<\|(call|constrain|return)\|>.*?<\|end\|>",
        ]

        cleaned_content = content
        for pattern in harmony_patterns:
            cleaned_content = re.sub(pattern, "", cleaned_content, flags=re.DOTALL)

        # Remove individual OpenAI Harmony special tokens
        harmony_tags = [
            r"<\|start\|>",
            r"<\|end\|>",
            r"<\|message\|>",
            r"<\|channel\|>",
            r"<\|constrain\|>",
            r"<\|return\|>",
            r"<\|call\|>",
        ]

        for tag_pattern in harmony_tags:
            cleaned_content = re.sub(tag_pattern, "", cleaned_content)

        # Clean up any leftover channel type markers that might be orphaned
        # Only remove if they appear at the very start of content after tag removal
        cleaned_content = re.sub(
            r"^(analysis|commentary|final)(?=\s)", "", cleaned_content
        )

        # Clean up multiple whitespaces and newlines left by tag removal
        cleaned_content = re.sub(r"\s+", " ", cleaned_content)
        cleaned_content = re.sub(r"\n\s*\n", "\n\n", cleaned_content)
        cleaned_content = re.sub(r"^\s+|\s+$", "", cleaned_content)

        return cleaned_content
