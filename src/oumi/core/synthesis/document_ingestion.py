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
from oumi.utils.logging import logger

try:
    from pdftext.extraction import (  # pyright: ignore[reportMissingImports]
        plain_text_output,
    )
except ImportError:
    plain_text_output = None


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

    _SUPPORTED_FILE_TYPES = {"pdf", "txt", "md", "html"}

    def __init__(self):
        """Initialize the document reader."""
        if plain_text_output is None:
            raise ImportError(
                "pdftext is not installed. Please install it with "
                "`pip install oumi[synthesis]`."
            )
        self._extractor_method = plain_text_output

    def read(self, document_path: str) -> list[str]:
        """Read the document."""
        path = Path(document_path)
        if "*" in str(path):
            return self._read_from_glob(path)
        else:
            file_type = self._get_file_type(path)
            if file_type in self._SUPPORTED_FILE_TYPES:
                with open(path, "rb") as file:
                    file_bytes = file.read()
                return [self._read_from_document_format(file_bytes, file_type)]
            else:
                raise NotImplementedError(f"Unsupported document format: {file_type}")

    def _get_file_type(self, path: Path) -> str:
        """Get the file type of the document."""
        return path.suffix.lstrip(".").lower()

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
                file_type = self._get_file_type(file)
                if file_type in self._SUPPORTED_FILE_TYPES:
                    with open(file, "rb") as file:
                        file_bytes = file.read()
                        documents.append(
                            self._read_from_document_format(file_bytes, file_type)
                        )
                else:
                    logger.warning(
                        f"Unsupported document format, skipping file: {file}"
                    )

        return documents

    def _read_from_document_format(
        self,
        file_bytes: bytes,
        file_type: str,
    ) -> str:
        """Read the document from the document format."""
        if file_type == "pdf":
            return self._read_from_pdf(file_bytes)
        elif file_type == "txt" or file_type == "md" or file_type == "html":
            return self._read_from_text_file(file_bytes)
        else:
            raise NotImplementedError(f"Unsupported document format: {file_type}")

    def _read_from_pdf(self, file_bytes: bytes) -> str:
        """Read the document from the PDF format."""
        plain_text = self._extractor_method(file_bytes, sort=True, hyphens=True)
        return plain_text

    def _read_from_text_file(self, file_bytes: bytes) -> str:
        """Read the document from the file."""
        return file_bytes.decode("utf-8")
