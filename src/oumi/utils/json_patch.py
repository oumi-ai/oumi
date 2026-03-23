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

import copy
from typing import Any

import jsonpatch
import jsonschema

from oumi.utils.str_utils import extract_json


class JsonPatchError(Exception):
    """Raised when a JSON Patch is malformed or cannot be applied."""


class JsonPatchValidationError(Exception):
    """Raised when the patched document fails JSON Schema validation."""


def apply_json_patch(
    document: dict[str, Any],
    patch: list[dict[str, Any]],
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply an RFC 6902 JSON Patch to a document.

    Args:
        document: The JSON document to patch (not mutated).
        patch: List of RFC 6902 patch operations.
        schema: Optional JSON Schema to validate the result against.

    Returns:
        A new dict with the patch applied.

    Raises:
        JsonPatchError: If the patch is malformed or application fails.
        JsonPatchValidationError: If the result fails schema validation.
    """
    doc_copy = copy.deepcopy(document)
    try:
        jp = jsonpatch.JsonPatch(patch)
        result = jp.apply(doc_copy)
    except (
        jsonpatch.JsonPatchException,
        jsonpatch.JsonPointerException,
        TypeError,
        KeyError,
        IndexError,
    ) as e:
        raise JsonPatchError(f"Failed to apply patch: {e}") from e

    if schema is not None:
        try:
            jsonschema.validate(instance=result, schema=schema)
        except jsonschema.ValidationError as e:
            raise JsonPatchValidationError(
                f"Patched document failed schema validation: {e.message}"
            ) from e

    return result


def parse_patch_response(text: str) -> list[dict[str, Any]] | None:
    """Extract a JSON Patch array from LLM-generated text.

    Handles markdown code fences and surrounding prose via extract_json.

    Args:
        text: Raw LLM response text.

    Returns:
        A list of patch operation dicts, or None if parsing fails.
    """
    result = extract_json(text, expected_type=list)
    if isinstance(result, list):
        return result
    return None
