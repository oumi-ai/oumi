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

from transformers import AutoTokenizer

from oumi.core.configs.params.synthesis_params import (
    DocumentSegmentationParams,
    SegmentationStrategy,
)

try:
    import pymupdf4llm  # pyright: ignore[reportMissingImports]
except ImportError:
    pymupdf4llm = None


class DocumentSegmenter:
    """Segmenter for documents."""

    def __init__(self, params: DocumentSegmentationParams):
        """Initialize the document segmenter."""
        self._params = params
        self._tokenizer = AutoTokenizer.from_pretrained(params.tokenizer)

    def segment(self, document: str) -> list[str]:
        """Segment the document."""
        segmentation_strategy = self._params.segmentation_strategy
        if segmentation_strategy == SegmentationStrategy.TOKENS:
            return self._segment_by_tokens(document)
        else:
            raise NotImplementedError(
                f"Unsupported segmentation strategy: {segmentation_strategy}"
            )

    def segment_batch(self, documents: list[str]) -> list[str]:
        """Segment multiple documents.

        Segments will be returned as a flat list of segments.
        """
        segments = []
        for document in documents:
            segments.extend(self.segment(document))
        return segments

    def _segment_by_tokens(self, document: str) -> list[str]:
        """Segment the document by tokens."""
        tokens = self._tokenizer.encode(document)
        segments = []
        stride = self._params.segment_length - self._params.segment_overlap
        for i in range(0, len(tokens), stride):
            segment = tokens[i : i + self._params.segment_length]
            decoded_segment = self._tokenizer.decode(segment)
            segments.append(decoded_segment)
        return segments


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

    def read(self, document_path: str) -> list[str]:
        """Read the document."""
        path = Path(document_path)
        if "*" in str(path):
            return self._read_from_glob(path)
        else:
            return [self._read_from_document_format(path)]

    def _read_from_glob(self, path: Path) -> list[str]:
        """Read the document from the glob path."""
        documents = []

        # Find the base directory (longest prefix without glob characters)
        parts = path.parts
        base_parts = []

        for part in parts:
            if "*" in part:
                break
            base_parts.append(part)

        # Determine base directory and pattern
        if base_parts:
            base_dir = Path(*base_parts)
            # Pattern is everything after the base directory
            pattern_parts = parts[len(base_parts) :]
            pattern = "/".join(pattern_parts)
        else:
            # If path starts with glob, use appropriate base
            if path.is_absolute():
                base_dir = Path("/")
                pattern = str(path)[1:]  # Remove leading slash
            else:
                base_dir = Path.cwd()
                pattern = str(path)

        # Use glob to find matching files
        for file in base_dir.glob(pattern):
            if file.is_file():
                documents.append(self._read_from_document_format(file))

        return documents

    def _read_from_document_format(
        self,
        document_path: Path,
    ) -> str:
        """Read the document from the document format."""
        path = str(document_path)
        suffix = document_path.suffix
        if suffix == ".pdf":
            return self._read_from_pdf(path)
        elif suffix == ".txt" or suffix == ".md" or suffix == ".html":
            return self._read_from_text_file(path)
        else:
            raise NotImplementedError(f"Unsupported document format: {suffix}")

    def _read_from_pdf(self, document_path: str) -> str:
        """Read the document from the PDF format."""
        markdown_text = self._pymupdf4llm.to_markdown(document_path)
        return markdown_text

    def _read_from_text_file(self, document_path: str) -> str:
        """Read the document from the file."""
        with open(document_path) as file:
            return file.read()
