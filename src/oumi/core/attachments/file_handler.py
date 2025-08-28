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

"""File attachment handler with auto-detection and context management."""

import json
import mimetypes
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import ClassVar, Optional

from oumi.core.attachments.context_manager import ContextWindowManager

# Simplified attachment system uses plain text instead of ContentItems


class FileType(Enum):
    """Supported file types for attachment."""

    IMAGE = "image"
    PDF = "pdf"
    TEXT = "text"
    CSV = "csv"
    JSON = "json"
    MARKDOWN = "markdown"
    CODE = "code"
    UNKNOWN = "unknown"


class ProcessingStrategy(Enum):
    """Strategies for processing different file types."""

    FULL_CONTENT = "full_content"  # Include complete file content
    SUMMARIZED = "summarized"  # Provide summary + key excerpts
    CHUNKED = "chunked"  # Process in chunks
    PREVIEW_ONLY = "preview_only"  # Show preview + metadata only
    FAILED = "failed"  # Processing failed


@dataclass
class FileInfo:
    """Information about an attached file."""

    path: str
    name: str
    size_bytes: int
    file_type: FileType
    mime_type: Optional[str]
    processing_strategy: ProcessingStrategy
    error_message: Optional[str] = None


@dataclass
class AttachmentResult:
    """Result of file attachment processing."""

    file_info: FileInfo
    text_content: str  # Simplified to just text content
    success: bool
    warning_message: Optional[str] = None
    context_info: Optional[str] = None


class FileHandler:
    """Handles file attachment with intelligent processing and context management."""

    # File size thresholds (in bytes)
    SMALL_FILE_THRESHOLD = 10 * 1024  # 10KB
    MEDIUM_FILE_THRESHOLD = 100 * 1024  # 100KB
    LARGE_FILE_THRESHOLD = 1024 * 1024  # 1MB

    # File type mappings
    TEXT_EXTENSIONS: ClassVar[set[str]] = {
        ".txt",
        ".md",
        ".rst",
        ".log",
        ".cfg",
        ".ini",
        ".conf",
    }
    CODE_EXTENSIONS: ClassVar[set[str]] = {
        ".py",
        ".js",
        ".ts",
        ".html",
        ".css",
        ".java",
        ".cpp",
        ".c",
        ".h",
        ".go",
        ".rs",
        ".php",
        ".rb",
        ".swift",
        ".kt",
        ".scala",
        ".sh",
        ".yaml",
        ".yml",
    }
    IMAGE_EXTENSIONS: ClassVar[set[str]] = {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".tiff",
        ".webp",
        ".svg",
    }

    def __init__(self, context_manager: Optional[ContextWindowManager] = None):
        """Initialize the file handler.

        Args:
            context_manager: Context window manager for token budgeting.
        """
        self.context_manager = context_manager or ContextWindowManager()

    def attach_file(
        self, file_path: str, conversation_tokens: int = 0
    ) -> AttachmentResult:
        """Attach a file with intelligent processing.

        Args:
            file_path: Path to the file to attach.
            conversation_tokens: Current conversation length in tokens.

        Returns:
            AttachmentResult with processed file content.
        """
        try:
            # Validate file exists
            path = Path(file_path)
            if not path.exists():
                return self._create_error_result(
                    file_path, f"File not found: {file_path}"
                )

            if not path.is_file():
                return self._create_error_result(
                    file_path, f"Path is not a file: {file_path}"
                )

            # Get file info
            file_info = self._analyze_file(path)

            # Calculate context budget
            budget = self.context_manager.calculate_budget(conversation_tokens)

            # Process file based on type and size
            return self._process_file(file_info, budget)

        except Exception as e:
            import traceback

            error_details = traceback.format_exc()
            return self._create_error_result(
                file_path,
                f"Error processing file: {str(e)}\n\nTraceback:\n{error_details}",
            )

    def _analyze_file(self, path: Path) -> FileInfo:
        """Analyze file to determine type and processing strategy."""
        stat = path.stat()
        size_bytes = stat.st_size

        # Determine file type
        file_type = self._detect_file_type(path)

        # Determine MIME type
        mime_type, _ = mimetypes.guess_type(str(path))

        # Determine processing strategy based on size and type
        processing_strategy = self._determine_strategy(file_type, size_bytes)

        return FileInfo(
            path=str(path),
            name=path.name,
            size_bytes=size_bytes,
            file_type=file_type,
            mime_type=mime_type,
            processing_strategy=processing_strategy,
        )

    def _detect_file_type(self, path: Path) -> FileType:
        """Detect file type based on extension and content."""
        ext = path.suffix.lower()

        if ext in self.IMAGE_EXTENSIONS:
            return FileType.IMAGE
        elif ext == ".pdf":
            return FileType.PDF
        elif ext == ".csv":
            return FileType.CSV
        elif ext == ".json":
            return FileType.JSON
        elif ext == ".md":
            return FileType.MARKDOWN
        elif ext in self.CODE_EXTENSIONS:
            return FileType.CODE
        elif ext in self.TEXT_EXTENSIONS:
            return FileType.TEXT
        else:
            return FileType.UNKNOWN

    def _determine_strategy(
        self, file_type: FileType, size_bytes: int
    ) -> ProcessingStrategy:
        """Determine processing strategy based on file type and size."""
        if file_type == FileType.IMAGE:
            # Images are always processed as binary content
            return ProcessingStrategy.FULL_CONTENT

        if size_bytes <= self.SMALL_FILE_THRESHOLD:
            return ProcessingStrategy.FULL_CONTENT
        elif size_bytes <= self.MEDIUM_FILE_THRESHOLD:
            return ProcessingStrategy.SUMMARIZED
        elif size_bytes <= self.LARGE_FILE_THRESHOLD:
            return ProcessingStrategy.CHUNKED
        else:
            return ProcessingStrategy.PREVIEW_ONLY

    def _process_file(self, file_info: FileInfo, budget) -> AttachmentResult:
        """Process file according to its determined strategy."""
        try:
            if file_info.file_type == FileType.IMAGE:
                return self._process_image(file_info, budget)
            elif file_info.file_type == FileType.PDF:
                return self._process_pdf(file_info, budget)
            elif file_info.file_type in [
                FileType.TEXT,
                FileType.MARKDOWN,
                FileType.CODE,
            ]:
                return self._process_text_file(file_info, budget)
            elif file_info.file_type == FileType.CSV:
                return self._process_csv(file_info, budget)
            elif file_info.file_type == FileType.JSON:
                return self._process_json(file_info, budget)
            else:
                return self._process_unknown_file(file_info, budget)

        except Exception as e:
            file_info.error_message = str(e)
            file_info.processing_strategy = ProcessingStrategy.FAILED
            return AttachmentResult(
                file_info=file_info,
                text_content="",
                success=False,
                warning_message=f"Failed to process {file_info.name}: {str(e)}",
            )

    def _process_image(self, file_info: FileInfo, budget) -> AttachmentResult:
        """Process image file."""
        try:
            # Create descriptive text for the image
            size_mb = (file_info.size_bytes or 0) / (1024 * 1024)

            # Create a text description of the image
            image_description = f"""[ATTACHED IMAGE: {file_info.name}]
File type: {file_info.file_type.value}
Size: {size_mb:.2f} MB
Path: {file_info.path}

Note: This is an image file that has been attached to the conversation. The actual
image content is not displayed as text, but you can reference it in your response.
"""

            return AttachmentResult(
                file_info=file_info,
                text_content=image_description,
                success=True,
                context_info=f"Image attached ({size_mb:.1f} MB)",
            )

        except Exception as e:
            raise Exception(f"Failed to process image: {str(e)}")

    def _process_pdf(self, file_info: FileInfo, budget) -> AttachmentResult:
        """Process PDF file."""
        # For Phase 2, provide basic PDF info
        # Full PDF processing requires additional dependencies

        size_mb = (file_info.size_bytes or 0) / (1024 * 1024)
        text_content = f"📄 **Attached PDF: {file_info.name}**\n"
        text_content += f"Size: {size_mb:.1f} MB\n\n"

        if file_info.processing_strategy == ProcessingStrategy.PREVIEW_ONLY:
            text_content += (
                "⚠️ PDF is too large for full processing. Showing file info only.\n\n"
            )
            text_content += (
                "**Recommendation**: Use a PDF reader to extract specific sections "
                "you'd like to discuss."
            )
        else:
            text_content += "📝 PDF processing will be enhanced in a future update.\n"
            text_content += (
                "For now, please extract the text you'd like to discuss and "
                "paste it directly."
            )

        warning_message = None
        if file_info.processing_strategy == ProcessingStrategy.PREVIEW_ONLY:
            warning_message = f"PDF too large ({size_mb:.1f} MB) for full processing"

        return AttachmentResult(
            file_info=file_info,
            text_content=text_content,
            success=True,
            warning_message=warning_message,
            context_info=f"PDF metadata attached ({size_mb:.1f} MB)",
        )

    def _process_text_file(self, file_info: FileInfo, budget) -> AttachmentResult:
        """Process text-based files (txt, md, code)."""
        try:
            with open(file_info.path, encoding="utf-8") as f:
                content = f.read()

        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_info.path, encoding="latin-1") as f:
                    content = f.read()
            except Exception:
                raise Exception("Could not decode file as text")

        # Estimate tokens and handle context limits
        token_estimate = self.context_manager.check_content_fit(content, budget)

        # Prepare content based on strategy
        if (
            file_info.processing_strategy == ProcessingStrategy.FULL_CONTENT
            and token_estimate.fits_in_budget
        ):
            processed_content = self._format_text_content(file_info, content)
            context_info = (
                f"Full content included ({token_estimate.estimated_tokens:,} tokens)"
            )

        elif (
            file_info.processing_strategy == ProcessingStrategy.SUMMARIZED
            or not token_estimate.fits_in_budget
        ):
            # Truncate to fit budget
            truncated_content = self.context_manager.truncate_content(
                content, budget.available_for_content
            )
            processed_content = self._format_text_content(file_info, truncated_content)
            context_info = (
                f"Content truncated to fit context "
                f"({budget.available_for_content:,} tokens)"
            )

        else:
            # Preview only
            preview = content[:1000] + "..." if len(content) > 1000 else content
            processed_content = self._format_text_content(
                file_info, preview, is_preview=True
            )
            context_info = (
                f"Preview only - file too large "
                f"({token_estimate.estimated_tokens:,} tokens)"
            )

        # Prepare warning message if needed
        warning_message = None
        if not token_estimate.fits_in_budget:
            if file_info.processing_strategy == ProcessingStrategy.CHUNKED:
                chunk_info = self.context_manager.suggest_chunking_strategy(
                    token_estimate.estimated_tokens, budget.available_for_content
                )
                warning_message = chunk_info["recommendation"]
            else:
                warning_message = (
                    f"File truncated - original size: "
                    f"{token_estimate.estimated_tokens:,} tokens"
                )

        return AttachmentResult(
            file_info=file_info,
            text_content=processed_content,
            success=True,
            warning_message=warning_message,
            context_info=context_info,
        )

    def _process_csv(self, file_info: FileInfo, budget) -> AttachmentResult:
        """Process CSV file."""
        try:
            # For Phase 2, we'll provide basic CSV processing
            # Enhanced data analysis will come in later phases

            size_mb = (file_info.size_bytes or 0) / (1024 * 1024)

            # Read a preview of the CSV
            with open(file_info.path, encoding="utf-8") as f:
                lines = []
                for i, line in enumerate(f):
                    lines.append(line.strip())
                    if i >= 10:  # First 10 lines for preview
                        break

            preview_content = "\n".join(lines)
            if len(lines) > 10:
                preview_content += (
                    f"\n... ({(file_info.size_bytes or 0) // 1024} KB total)"
                )

            text_content = f"📊 **Attached CSV: {file_info.name}**\n"
            text_content += f"Size: {size_mb:.2f} MB\n\n"
            text_content += "**Preview (first 10 lines):**\n```csv\n"
            text_content += preview_content
            text_content += "\n```\n\n"

            if file_info.processing_strategy == ProcessingStrategy.PREVIEW_ONLY:
                text_content += "⚠️ CSV is large. Showing preview only.\n"
                text_content += (
                    "**Tip**: Ask specific questions about the data structure or "
                    "request analysis of particular columns."
                )

            warning_message = None
            if file_info.processing_strategy == ProcessingStrategy.PREVIEW_ONLY:
                warning_message = (
                    f"Large CSV file ({size_mb:.1f} MB) - showing preview only"
                )

            return AttachmentResult(
                file_info=file_info,
                text_content=text_content,
                success=True,
                warning_message=warning_message,
                context_info=f"CSV preview attached ({size_mb:.2f} MB)",
            )

        except Exception as e:
            raise Exception(f"Failed to process CSV: {str(e)}")

    def _process_json(self, file_info: FileInfo, budget) -> AttachmentResult:
        """Process JSON file."""
        try:
            with open(file_info.path, encoding="utf-8") as f:
                data = json.load(f)

            # Format JSON with proper indentation
            formatted_json = json.dumps(data, indent=2, ensure_ascii=False)

            # Check if it fits in context
            token_estimate = self.context_manager.check_content_fit(
                formatted_json, budget
            )

            if token_estimate.fits_in_budget:
                processed_content = self._format_json_content(file_info, formatted_json)
                context_info = (
                    f"Full JSON included ({token_estimate.estimated_tokens:,} tokens)"
                )
                warning_message = None
            else:
                # Provide structure summary instead of full content
                structure_info = self._analyze_json_structure(data)
                processed_content = f"🔧 **Attached JSON: {file_info.name}**\n\n"
                processed_content += f"**Structure Analysis:**\n{structure_info}\n\n"
                processed_content += "**Note**: JSON is too large for full inclusion. "
                processed_content += (
                    "Ask about specific keys or sections you're interested in."
                )

                context_info = (
                    f"JSON structure summary "
                    f"({token_estimate.estimated_tokens:,} tokens total)"
                )
                warning_message = (
                    "Large JSON file truncated - use specific queries for detailed data"
                )

            return AttachmentResult(
                file_info=file_info,
                text_content=processed_content,
                success=True,
                warning_message=warning_message,
                context_info=context_info,
            )

        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to process JSON: {str(e)}")

    def _process_unknown_file(self, file_info: FileInfo, budget) -> AttachmentResult:
        """Process unknown file type."""
        size_mb = (file_info.size_bytes or 0) / (1024 * 1024)

        text_content = f"❓ **Attached File: {file_info.name}**\n"
        text_content += f"Size: {size_mb:.2f} MB\n"
        text_content += f"Type: {file_info.mime_type or 'Unknown'}\n\n"
        text_content += "⚠️ File type not recognized for automatic processing.\n\n"
        text_content += (
            "**Supported formats**: Images (JPG, PNG, etc.), PDF, Text files, "
            "CSV, JSON, Markdown, Code files\n\n"
        )
        text_content += (
            "**Suggestion**: If this is a text file, try renaming with a "
            "`.txt` extension."
        )

        return AttachmentResult(
            file_info=file_info,
            text_content=text_content,
            success=True,
            warning_message=f"Unsupported file type: {file_info.name}",
            context_info=f"File metadata only ({size_mb:.2f} MB)",
        )

    def _format_text_content(
        self, file_info: FileInfo, content: str, is_preview: bool = False
    ) -> str:
        """Format text content for display."""
        icon = "📄"
        if file_info.file_type == FileType.CODE:
            icon = "💾"
        elif file_info.file_type == FileType.MARKDOWN:
            icon = "📝"

        preview_note = " (Preview)" if is_preview else ""

        result = (
            f"{icon} **Attached {file_info.file_type.value.title()}: "
            f"{file_info.name}**{preview_note}\n\n"
        )

        # Determine appropriate code block language
        lang = ""
        if file_info.file_type == FileType.CODE:
            ext = Path(file_info.name).suffix.lower()
            lang_map = {
                ".py": "python",
                ".js": "javascript",
                ".ts": "typescript",
                ".html": "html",
                ".css": "css",
                ".java": "java",
                ".cpp": "cpp",
                ".c": "c",
                ".go": "go",
                ".rs": "rust",
                ".php": "php",
                ".rb": "ruby",
                ".swift": "swift",
                ".yaml": "yaml",
                ".yml": "yaml",
                ".sh": "bash",
            }
            lang = lang_map.get(ext, "text")
        elif file_info.file_type == FileType.MARKDOWN:
            lang = "markdown"

        result += f"```{lang}\n{content}\n```"

        if is_preview:
            result += (
                "\n\n*Note: This is a preview. The full file may contain more content.*"
            )

        return result

    def _format_json_content(self, file_info: FileInfo, content: str) -> str:
        """Format JSON content for display."""
        return f"🔧 **Attached JSON: {file_info.name}**\n\n```json\n{content}\n```"

    def _analyze_json_structure(self, data) -> str:
        """Analyze JSON structure and provide summary."""

        def analyze_value(value, depth=0):
            indent = "  " * depth
            if isinstance(value, dict):
                result = f"{indent}Object with {len(value)} keys:\n"
                for key in list(value.keys())[:5]:  # Show first 5 keys
                    result += f"{indent}  - {key}: {type(value[key]).__name__}\n"
                if len(value) > 5:
                    result += f"{indent}  ... ({len(value) - 5} more keys)\n"
                return result
            elif isinstance(value, list):
                result = f"{indent}Array with {len(value)} items\n"
                if value and depth < 2:  # Don't go too deep
                    result += f"{indent}  Item type: {type(value[0]).__name__}\n"
                return result
            else:
                suffix = "..." if len(str(value)) > 50 else ""
                return f"{indent}{type(value).__name__}: {str(value)[:50]}{suffix}\n"

        return analyze_value(data)

    def _create_error_result(
        self, file_path: str, error_message: str
    ) -> AttachmentResult:
        """Create an error result for failed file processing."""
        file_info = FileInfo(
            path=file_path,
            name=Path(file_path).name,
            size_bytes=0,
            file_type=FileType.UNKNOWN,
            mime_type=None,
            processing_strategy=ProcessingStrategy.FAILED,
            error_message=error_message,
        )

        return AttachmentResult(
            file_info=file_info,
            text_content="",
            success=False,
            warning_message=error_message,
        )
