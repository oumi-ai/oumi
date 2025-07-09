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

from enum import Enum
from pathlib import Path

try:
    import pymupdf4llm  # pyright: ignore[reportMissingImports]
except ImportError:
    pymupdf4llm = None


class DocumentFormat(Enum):
    """Format of document (PDF, TXT, HTML, DOCX, MD)."""

    PDF = "pdf"
    """PDF file"""

    TXT = "txt"
    """Text file"""

    HTML = "html"
    """HTML file"""

    MD = "md"
    """Markdown file"""


class DocumentPath:
    """Path to a document in some storage location."""

    def __init__(self, path: str):
        """Initialize the document path.

        Args:
            path: The path to the document.
        """
        self._path = path
        self._document_format = self._get_document_format(path)

    def _get_document_format(self, path: str) -> DocumentFormat:
        """Get the document format from the path."""
        lower_path = path.lower()
        if lower_path.endswith(".pdf"):
            return DocumentFormat.PDF
        elif lower_path.endswith(".txt"):
            return DocumentFormat.TXT
        elif lower_path.endswith(".html"):
            return DocumentFormat.HTML
        elif lower_path.endswith(".md"):
            return DocumentFormat.MD
        else:
            raise ValueError(f"Unsupported document format: {path}")

    def get_document_format(self) -> DocumentFormat:
        """Get the document format."""
        return self._document_format

    def get_path_str(self) -> str:
        """Get the path to the document."""
        return self._path


class DocumentReader:
    """Reader for documents."""

    def __init__(self):
        """Initialize the document reader."""
        if pymupdf4llm is None:
            raise ImportError(
                "pymupdf4llm is not installed. Please install it with "
                "`pip install oumi[synthesis]`."
            )
        self._pymupdf4llm = pymupdf4llm

    def read(self, document_path: DocumentPath) -> list[str]:
        """Read the document."""
        document_format = document_path.get_document_format()
        path = Path(document_path.get_path_str())
        if "*" in str(path):
            return self._read_from_glob(path, document_format)
        else:
            return [self._read_from_document_format(path, document_format)]

    def _read_from_glob(self, path: Path, document_format: DocumentFormat) -> list[str]:
        """Read the document from the glob path."""
        documents = []
        for file in Path(path.parent).glob(path.name):
            documents.append(self._read_from_document_format(file, document_format))
        return documents

    def _read_from_document_format(
        self,
        document_path: Path,
        document_format: DocumentFormat,
    ) -> str:
        """Read the document from the document format."""
        path = str(document_path)
        if document_format == DocumentFormat.PDF:
            return self._read_from_pdf(path)
        elif (
            document_format == DocumentFormat.TXT
            or document_format == DocumentFormat.MD
            or document_format == DocumentFormat.HTML
        ):
            return self._read_from_text_file(path)
        else:
            raise NotImplementedError(f"Unsupported document format: {document_format}")

    def _read_from_pdf(self, document_path: str) -> str:
        """Read the document from the PDF format."""
        markdown_text = self._pymupdf4llm.to_markdown(document_path)
        return markdown_text

    def _read_from_text_file(self, document_path: str) -> str:
        """Read the document from the file."""
        with open(document_path) as file:
            return file.read()
