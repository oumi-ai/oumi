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

from oumi.core.configs.params.synthesis_params import (
    DocumentSegmentationParams,
    SegmentationStrategy,
)
from oumi.core.synthesis.document_ingestion import (
    DocumentReader,
    DocumentSegmenter,
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


def test_read_single_pdf_document(reader, sample_pdf_content):
    """Test reading a single PDF document."""
    document_path = "path/to/document.pdf"

    with patch("pymupdf4llm.to_markdown", return_value=sample_pdf_content) as mock_pdf:
        result = reader.read(document_path)

        mock_pdf.assert_called_once_with("path/to/document.pdf")
        assert result == [sample_pdf_content]


def test_read_single_txt_document(reader, sample_text_content):
    """Test reading a single TXT document."""
    document_path = "path/to/document.txt"

    with patch("builtins.open", mock_open(read_data=sample_text_content)):
        result = reader.read(document_path)

        assert result == [sample_text_content]


def test_read_single_html_document(reader, sample_text_content):
    """Test reading a single HTML document."""
    document_path = "path/to/document.html"

    with patch("builtins.open", mock_open(read_data=sample_text_content)):
        result = reader.read(document_path)

        assert result == [sample_text_content]


def test_read_single_md_document(reader, sample_text_content):
    """Test reading a single Markdown document."""
    document_path = "path/to/document.md"

    with patch("builtins.open", mock_open(read_data=sample_text_content)):
        result = reader.read(document_path)

        assert result == [sample_text_content]


def test_read_multiple_documents_glob_pattern(reader, sample_text_content):
    """Test reading multiple documents using glob pattern."""
    document_path = "path/to/*.txt"

    # Create mock Path objects with is_file() returning True
    mock_files = []
    for filename in ["file1.txt", "file2.txt", "file3.txt"]:
        mock_file = MagicMock(spec=Path)
        mock_file.is_file.return_value = True
        mock_file.suffix = ".txt"
        mock_file.__str__ = MagicMock(return_value=f"path/to/{filename}")
        mock_files.append(mock_file)

    with patch("pathlib.Path.glob", return_value=mock_files):
        with patch("builtins.open", mock_open(read_data=sample_text_content)):
            result = reader.read(document_path)

            assert len(result) == 3
            assert all(content == sample_text_content for content in result)


def test_read_multiple_directories_files_glob_pattern(reader, sample_text_content):
    """Test reading multiple documents using glob pattern."""
    document_path = "path/*/to/*.txt"

    # Create mock Path objects with is_file() returning True
    mock_files = []
    for path_str in [
        "path/subdir1/to/file1.txt",
        "path/subdir2/to/file2.txt",
        "path/subdir3/to/file3.txt",
    ]:
        mock_file = MagicMock(spec=Path)
        mock_file.is_file.return_value = True
        mock_file.suffix = ".txt"
        mock_file.__str__ = MagicMock(return_value=path_str)
        mock_files.append(mock_file)

    with patch("pathlib.Path.glob", return_value=mock_files):
        with patch("builtins.open", mock_open(read_data=sample_text_content)):
            result = reader.read(document_path)

            assert len(result) == 3
            assert all(content == sample_text_content for content in result)


def test_read_multiple_pdf_documents_glob_pattern(reader, sample_pdf_content):
    """Test reading multiple PDF documents using glob pattern."""
    document_path = "path/to/*.pdf"

    # Create mock Path objects with is_file() returning True
    mock_files = []
    for filename in ["file1.pdf", "file2.pdf"]:
        mock_file = MagicMock(spec=Path)
        mock_file.is_file.return_value = True
        mock_file.suffix = ".pdf"
        mock_file.__str__ = MagicMock(return_value=f"path/to/{filename}")
        mock_files.append(mock_file)

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
    document_path = "path/to/*.txt"

    with patch("pathlib.Path.glob", return_value=[]):
        result = reader.read(document_path)

        assert result == []


def test_read_from_document_format_unsupported(reader):
    """Test reading document with unsupported format."""

    with pytest.raises(NotImplementedError, match="Unsupported document format"):
        reader._read_from_document_format(Path("path/to/document.unsupported"))


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
    # Create mock Path objects with is_file() returning True
    mock_files = []
    formats = [("file1.txt", ".txt"), ("file2.md", ".md"), ("file3.html", ".html")]
    for filename, suffix in formats:
        mock_file = MagicMock(spec=Path)
        mock_file.is_file.return_value = True
        mock_file.suffix = suffix
        mock_file.__str__ = MagicMock(return_value=f"path/to/{filename}")
        mock_files.append(mock_file)

    with patch("pathlib.Path.glob", return_value=mock_files):
        with patch("builtins.open", mock_open(read_data=sample_text_content)):
            result = reader._read_from_glob(Path("path/to/*.txt"))

            assert len(result) == 3
            assert all(content == sample_text_content for content in result)


def test_read_handles_file_read_error(reader):
    """Test that reading handles file read errors gracefully."""
    document_path = "path/to/nonexistent.txt"

    with patch("builtins.open", side_effect=FileNotFoundError("File not found")):
        with pytest.raises(FileNotFoundError):
            reader.read(document_path)


def test_read_handles_pdf_read_error(reader):
    """Test that reading handles PDF read errors gracefully."""
    document_path = "path/to/corrupted.pdf"

    with patch("pymupdf4llm.to_markdown", side_effect=Exception("PDF read error")):
        with pytest.raises(Exception, match="PDF read error"):
            reader.read(document_path)


def test_read_real_pdf_document(reader, root_testdata_dir):
    """Test reading a real PDF document."""
    document_path = f"{root_testdata_dir}/pdfs/mock.pdf"
    result = reader.read(document_path)

    # Verify the result
    assert len(result) == 1
    assert isinstance(result[0], str)
    assert len(result[0]) > 0

    assert "**Dummy PDF file**" in result[0]


def test_read_mixed_documents(
    reader,
    sample_text_content,
    sample_pdf_content,
):
    """Integration test reading different document types."""
    # Test reading a mix of document types sequentially
    txt_path = "document.txt"
    pdf_path = "document.pdf"
    md_path = "document.md"

    with patch("builtins.open", mock_open(read_data=sample_text_content)):
        txt_result = reader.read(txt_path)
        md_result = reader.read(md_path)

    with patch("pymupdf4llm.to_markdown", return_value=sample_pdf_content):
        pdf_result = reader.read(pdf_path)

    assert txt_result == [sample_text_content]
    assert pdf_result == [sample_pdf_content]
    assert md_result == [sample_text_content]


# DocumentSegmenter tests
@pytest.fixture
def segmentation_params():
    """Create test default segmentation parameters."""
    return DocumentSegmentationParams(
        id="test_segments",
        segmentation_strategy=SegmentationStrategy.TOKENS,
        tokenizer="openai-community/gpt2",
        segment_length=10,
        segment_overlap=2,
    )


@pytest.fixture
def segmenter(segmentation_params):
    """Create a DocumentSegmenter instance."""
    with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer:
        mock_tokenizer.return_value = MagicMock()
        return DocumentSegmenter(segmentation_params)


@pytest.fixture
def sample_document():
    """Sample document for testing."""
    return "This is a sample document for testing segmentation functionality."


def test_document_segmenter_initialization(segmentation_params):
    """Test DocumentSegmenter initialization."""
    with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer:
        mock_tokenizer.return_value = MagicMock()

        segmenter = DocumentSegmenter(segmentation_params)

        assert segmenter._params == segmentation_params
        mock_tokenizer.assert_called_once_with(segmentation_params.tokenizer)


def test_segment_tokens_strategy(segmenter, sample_document):
    """Test segmentation with TOKENS strategy."""
    # Mock tokenizer behavior
    mock_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    segmenter._tokenizer.encode.return_value = mock_tokens

    # Mock decode behavior for segments
    def mock_decode(tokens):
        return f"decoded_{tokens[0]}_{tokens[-1]}"

    segmenter._tokenizer.decode.side_effect = mock_decode

    result = segmenter.segment(sample_document)

    # With segment_length=10, overlap=2, stride=8
    # First segment: tokens 0-9 (1-10)
    # Second segment: tokens 8-15 (9-15, limited by token count)
    expected_segments = [
        "decoded_1_10",  # tokens 1-10
        "decoded_9_15",  # tokens 9-15
    ]

    assert result == expected_segments
    segmenter._tokenizer.encode.assert_called_once_with(sample_document)


def test_segment_tokens_no_overlap(segmentation_params, sample_document):
    """Test segmentation with no overlap."""
    segmentation_params.segment_overlap = 0

    with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer:
        mock_tokenizer.return_value = MagicMock()
        segmenter = DocumentSegmenter(segmentation_params)

        # Mock tokenizer behavior
        mock_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        segmenter._tokenizer.encode.return_value = mock_tokens

        def mock_decode(tokens):
            return f"decoded_{tokens[0]}_{tokens[-1]}"

        segmenter._tokenizer.decode.side_effect = mock_decode

        result = segmenter.segment(sample_document)

        # With segment_length=10, overlap=0, stride=10
        # First segment: tokens 0-9 (1-10)
        # Second segment: tokens 10-14 (11-15)
        expected_segments = [
            "decoded_1_10",  # tokens 1-10
            "decoded_11_15",  # tokens 11-15
        ]

        assert result == expected_segments


def test_segment_tokens_single_segment(segmentation_params, sample_document):
    """Test segmentation when document fits in a single segment."""
    segmentation_params.segment_length = 20

    with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer:
        mock_tokenizer.return_value = MagicMock()
        segmenter = DocumentSegmenter(segmentation_params)

        # Mock tokenizer behavior - fewer tokens than segment_length
        mock_tokens = [1, 2, 3, 4, 5]
        segmenter._tokenizer.encode.return_value = mock_tokens
        segmenter._tokenizer.decode.return_value = "decoded_segment"

        result = segmenter.segment(sample_document)

        assert result == ["decoded_segment"]
        segmenter._tokenizer.encode.assert_called_once_with(sample_document)
        segmenter._tokenizer.decode.assert_called_once_with([1, 2, 3, 4, 5])


def test_segment_empty_document(segmenter):
    """Test segmentation with empty document."""
    segmenter._tokenizer.encode.return_value = []
    segmenter._tokenizer.decode.return_value = ""

    result = segmenter.segment("")

    assert result == []
    segmenter._tokenizer.encode.assert_called_once_with("")


def test_segment_with_large_overlap(segmentation_params, sample_document):
    """Test segmentation with large overlap."""
    segmentation_params.segment_overlap = 8  # stride = 10 - 8 = 2

    with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer:
        mock_tokenizer.return_value = MagicMock()
        segmenter = DocumentSegmenter(segmentation_params)

        # Mock tokenizer behavior
        mock_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        segmenter._tokenizer.encode.return_value = mock_tokens

        def mock_decode(tokens):
            return f"decoded_{tokens[0]}_{tokens[-1]}"

        segmenter._tokenizer.decode.side_effect = mock_decode

        result = segmenter.segment(sample_document)

        # With segment_length=10, overlap=8, stride=2
        # Many overlapping segments
        expected_segments = [
            "decoded_1_10",  # tokens 1-10
            "decoded_3_12",  # tokens 3-12
            "decoded_5_14",  # tokens 5-14
            "decoded_7_15",  # tokens 7-15 (limited by token count)
            "decoded_9_15",  # tokens 9-15
            "decoded_11_15",  # tokens 11-15
            "decoded_13_15",  # tokens 13-15
            "decoded_15_15",  # tokens 15-15
        ]

        assert result == expected_segments


def test_segment_unsupported_strategy(segmentation_params, sample_document):
    """Test segmentation with unsupported strategy."""
    # Mock an unsupported strategy
    segmentation_params.segmentation_strategy = "unsupported_strategy"

    with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer:
        mock_tokenizer.return_value = MagicMock()
        segmenter = DocumentSegmenter(segmentation_params)

        with pytest.raises(
            NotImplementedError, match="Unsupported segmentation strategy"
        ):
            segmenter.segment(sample_document)


def test_segment_batch(segmenter):
    """Test segmenting multiple documents in batch."""
    documents = [
        "First document content",
        "Second document content",
        "Third document content",
    ]

    # Mock the tokenizer to return predictable tokens for each document
    def mock_encode(text):
        if "First" in text:
            return [1, 2, 3, 4, 5]
        elif "Second" in text:
            return [6, 7, 8, 9, 10]
        elif "Third" in text:
            return [11, 12, 13, 14, 15]
        return []

    def mock_decode(tokens):
        return f"segment_{tokens}"

    with patch.object(segmenter._tokenizer, "encode", side_effect=mock_encode):
        with patch.object(segmenter._tokenizer, "decode", side_effect=mock_decode):
            segments = segmenter.segment_batch(documents)

            # Should return a flat list of segments from all documents
            assert len(segments) == 3  # One segment per document given our token counts
            assert segments[0] == "segment_[1, 2, 3, 4, 5]"  # First document
            assert segments[1] == "segment_[6, 7, 8, 9, 10]"  # Second document
            assert segments[2] == "segment_[11, 12, 13, 14, 15]"  # Third document
