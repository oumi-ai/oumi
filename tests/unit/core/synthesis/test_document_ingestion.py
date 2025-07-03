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

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from oumi.core.synthesis.document_ingestion import (
    DocumentFormat,
    DocumentPath,
    DocumentReader,
)


@pytest.fixture
def reader():
    """Create a DocumentReader instance."""
    return DocumentReader()


@pytest.fixture
def sample_text_content():
    """Sample text content for testing."""
    return "This is sample text content for testing."


@pytest.fixture
def sample_pdf_content():
    """Sample PDF content converted to markdown."""
    return "# Sample PDF Content\n\nThis is a sample PDF converted to markdown."


def test_enum_values():
    """Test that DocumentFormat enum has correct values."""
    assert DocumentFormat.PDF.value == "pdf"
    assert DocumentFormat.TXT.value == "txt"
    assert DocumentFormat.HTML.value == "html"
    assert DocumentFormat.MD.value == "md"


def test_pdf_path():
    """Test initialization with PDF path."""
    path = DocumentPath("path/to/document.pdf")
    assert path.get_path_str() == "path/to/document.pdf"
    assert path.get_document_format() == DocumentFormat.PDF


def test_txt_path():
    """Test initialization with TXT path."""
    path = DocumentPath("path/to/document.txt")
    assert path.get_path_str() == "path/to/document.txt"
    assert path.get_document_format() == DocumentFormat.TXT


def test_html_path():
    """Test initialization with HTML path."""
    path = DocumentPath("path/to/document.html")
    assert path.get_path_str() == "path/to/document.html"
    assert path.get_document_format() == DocumentFormat.HTML


def test_md_path():
    """Test initialization with Markdown path."""
    path = DocumentPath("path/to/document.md")
    assert path.get_path_str() == "path/to/document.md"
    assert path.get_document_format() == DocumentFormat.MD


def test_glob_pattern_path():
    """Test initialization with glob pattern."""
    path = DocumentPath("path/to/*.pdf")
    assert path.get_path_str() == "path/to/*.pdf"
    assert path.get_document_format() == DocumentFormat.PDF


def test_unsupported_format():
    """Test initialization with unsupported format."""
    with pytest.raises(ValueError, match="Unsupported document format"):
        DocumentPath("path/to/document.docx")


def test_unsupported_format_unknown_extension():
    """Test initialization with unknown extension."""
    with pytest.raises(ValueError, match="Unsupported document format"):
        DocumentPath("path/to/document.unknown")


def test_no_extension():
    """Test initialization with no extension."""
    with pytest.raises(ValueError, match="Unsupported document format"):
        DocumentPath("path/to/document")


def test_read_single_pdf_document(reader, sample_pdf_content):
    """Test reading a single PDF document."""
    document_path = DocumentPath("path/to/document.pdf")

    with patch("pymupdf4llm.to_markdown", return_value=sample_pdf_content) as mock_pdf:
        result = reader.read(document_path)

        mock_pdf.assert_called_once_with("path/to/document.pdf")
        assert result == [sample_pdf_content]


def test_read_single_txt_document(reader, sample_text_content):
    """Test reading a single TXT document."""
    document_path = DocumentPath("path/to/document.txt")

    with patch("builtins.open", mock_open(read_data=sample_text_content)):
        result = reader.read(document_path)

        assert result == [sample_text_content]


def test_read_single_html_document(reader, sample_text_content):
    """Test reading a single HTML document."""
    document_path = DocumentPath("path/to/document.html")

    with patch("builtins.open", mock_open(read_data=sample_text_content)):
        result = reader.read(document_path)

        assert result == [sample_text_content]


def test_read_single_md_document(reader, sample_text_content):
    """Test reading a single Markdown document."""
    document_path = DocumentPath("path/to/document.md")

    with patch("builtins.open", mock_open(read_data=sample_text_content)):
        result = reader.read(document_path)

        assert result == [sample_text_content]


def test_read_multiple_documents_glob_pattern(reader, sample_text_content):
    """Test reading multiple documents using glob pattern."""
    document_path = DocumentPath("path/to/*.txt")

    # Mock Path.glob to return multiple files
    mock_files = [
        Path("path/to/file1.txt"),
        Path("path/to/file2.txt"),
        Path("path/to/file3.txt"),
    ]

    with patch("pathlib.Path.glob", return_value=mock_files):
        with patch("builtins.open", mock_open(read_data=sample_text_content)):
            result = reader.read(document_path)

            assert len(result) == 3
            assert all(content == sample_text_content for content in result)


def test_read_multiple_pdf_documents_glob_pattern(reader, sample_pdf_content):
    """Test reading multiple PDF documents using glob pattern."""
    document_path = DocumentPath("path/to/*.pdf")

    # Mock Path.glob to return multiple PDF files
    mock_files = [
        Path("path/to/file1.pdf"),
        Path("path/to/file2.pdf"),
    ]

    with patch("pathlib.Path.glob", return_value=mock_files):
        with patch(
            "pymupdf4llm.to_markdown", return_value=sample_pdf_content
        ) as mock_pdf:
            result = reader.read(document_path)

            assert len(result) == 2
            assert all(content == sample_pdf_content for content in result)
            assert mock_pdf.call_count == 2


def test_read_empty_glob_pattern(reader):
    """Test reading with glob pattern that matches no files."""
    document_path = DocumentPath("path/to/*.txt")

    with patch("pathlib.Path.glob", return_value=[]):
        result = reader.read(document_path)

        assert result == []


def test_read_from_document_format_unsupported(reader):
    """Test reading document with unsupported format."""
    # Create a mock DocumentFormat that's not implemented
    with patch.object(DocumentPath, "_get_document_format") as mock_format:
        # Create a new enum value that's not handled
        mock_format.return_value = MagicMock()
        mock_format.return_value.name = "UNSUPPORTED"

        document_path = DocumentPath.__new__(DocumentPath)
        document_path._path = "path/to/document.unsupported"
        document_path._document_format = mock_format.return_value

        with pytest.raises(NotImplementedError, match="Unsupported document format"):
            reader._read_from_document_format(
                Path("path/to/document.unsupported"), mock_format.return_value
            )


def test_read_from_pdf_calls_pymupdf4llm(reader, sample_pdf_content):
    """Test that reading PDF calls pymupdf4llm correctly."""
    with patch("pymupdf4llm.to_markdown", return_value=sample_pdf_content) as mock_pdf:
        result = reader._read_from_pdf("path/to/document.pdf")

        mock_pdf.assert_called_once_with("path/to/document.pdf")
        assert result == sample_pdf_content


def test_read_from_text_file_opens_file_correctly(reader, sample_text_content):
    """Test that reading text file opens file correctly."""
    with patch("builtins.open", mock_open(read_data=sample_text_content)) as mock_file:
        result = reader._read_from_text_file("path/to/document.txt")

        mock_file.assert_called_once_with("path/to/document.txt")
        assert result == sample_text_content


def test_read_from_glob_with_different_formats(reader, sample_text_content):
    """Test reading from glob with mixed document formats."""
    # Create mock files with different formats
    mock_files = [
        Path("path/to/file1.txt"),
        Path("path/to/file2.md"),
        Path("path/to/file3.html"),
    ]

    with patch("pathlib.Path.glob", return_value=mock_files):
        with patch("builtins.open", mock_open(read_data=sample_text_content)):
            result = reader._read_from_glob(Path("path/to/*.txt"), DocumentFormat.TXT)

            assert len(result) == 3
            assert all(content == sample_text_content for content in result)


def test_read_handles_file_read_error(reader):
    """Test that reading handles file read errors gracefully."""
    document_path = DocumentPath("path/to/nonexistent.txt")

    with patch("builtins.open", side_effect=FileNotFoundError("File not found")):
        with pytest.raises(FileNotFoundError):
            reader.read(document_path)


def test_read_handles_pdf_read_error(reader):
    """Test that reading handles PDF read errors gracefully."""
    document_path = DocumentPath("path/to/corrupted.pdf")

    with patch("pymupdf4llm.to_markdown", side_effect=Exception("PDF read error")):
        with pytest.raises(Exception, match="PDF read error"):
            reader.read(document_path)


def test_document_path_format_detection_edge_cases():
    """Test edge cases in document format detection."""
    # Test with multiple dots in filename
    path = DocumentPath("path/to/file.backup.pdf")
    assert path.get_document_format() == DocumentFormat.PDF

    # Test with uppercase extension
    path = DocumentPath("path/to/file.PDF")
    assert path.get_document_format() == DocumentFormat.PDF

    # Test with path containing spaces
    path = DocumentPath("path/to/my document.txt")
    assert path.get_document_format() == DocumentFormat.TXT


def test_read_real_pdf_document(reader, root_testdata_dir):
    """Test reading a real PDF document."""
    document_path = DocumentPath(f"{root_testdata_dir}/pdfs/mock.pdf")
    result = reader.read(document_path)

    # Verify the result
    assert len(result) == 1
    assert isinstance(result[0], str)
    assert len(result[0]) > 0

    assert "**Dummy PDF file**\n\n" == result[0]


def test_integration_read_mixed_documents(
    reader,
    sample_text_content,
    sample_pdf_content,
):
    """Integration test reading different document types."""
    # Test reading a mix of document types sequentially
    txt_path = DocumentPath("document.txt")
    pdf_path = DocumentPath("document.pdf")
    md_path = DocumentPath("document.md")

    with patch("builtins.open", mock_open(read_data=sample_text_content)):
        txt_result = reader.read(txt_path)
        md_result = reader.read(md_path)

    with patch("pymupdf4llm.to_markdown", return_value=sample_pdf_content):
        pdf_result = reader.read(pdf_path)

    assert txt_result == [sample_text_content]
    assert pdf_result == [sample_pdf_content]
    assert md_result == [sample_text_content]
