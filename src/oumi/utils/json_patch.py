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

"""RFC 6902 JSON Patch utilities."""

from typing import Any


class JsonPatchError(Exception):
    """Raised when a JSON Patch is malformed or cannot be applied."""


class JsonPatchValidationError(Exception):
    """Raised when the patched document fails JSON Schema validation."""


def apply_json_patch(
    document: dict[str, Any],
    patch: list[dict[str, Any]],
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply an RFC 6902 JSON Patch to a document."""
    raise NotImplementedError


def parse_patch_response(text: str) -> list[dict[str, Any]] | None:
    """Extract a JSON Patch array from LLM-generated text."""
    raise NotImplementedError
