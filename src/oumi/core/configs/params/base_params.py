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

import dataclasses
from collections.abc import Iterator
from typing import Any, Optional


@dataclasses.dataclass
class BaseParams:
    """Base class for all parameter classes.

    This class provides a common interface for all parameter classes,
    and provides a `finalize_and_validate` method to recursively validate the
    parameters.

    Subclasses should implement the `__finalize_and_validate__` method to perform
    custom validation logic.
    """

    #
    # Public methods
    #
    def finalize_and_validate(self) -> None:
        """Recursively finalizes and validates the parameters."""
        self._finalize_and_validate(set())

    def __finalize_and_validate__(self) -> None:
        """Finalizes and validates the parameters of this object.

        This method can be overridden by subclasses to implement custom
        validation logic.

        In case of validation errors, this method should raise a `ValueError`
        or other appropriate exception.
        """

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        """Returns an iterator over field names and values.

        Note: for an attribute to be a field, it must be declared in the
        dataclass definition and have a type annotation.
        """
        for param in dataclasses.fields(self):
            yield param.name, getattr(self, param.name)

    #
    # Private methods
    #
    def _finalize_and_validate(self, validated: Optional[set[int]]) -> None:
        """Recursively finalizes and validates the parameters."""
        if validated is None:
            validated = set()

        # If this object has already been validated, return immediately
        if id(self) in validated:
            return
        validated.add(id(self))

        # Finalize and validate the children of this object.
        # Note that we only support one level of nesting.
        # For example: `List[BaseParams]` is supported, but not `List[List[BaseParams]]`
        for _, attr_value in self:
            if isinstance(attr_value, BaseParams):
                attr_value._finalize_and_validate(validated)
            elif isinstance(attr_value, list):
                for item in attr_value:
                    if isinstance(item, BaseParams):
                        item._finalize_and_validate(validated)
            elif isinstance(attr_value, dict):
                for item in attr_value.values():
                    if isinstance(item, BaseParams):
                        item._finalize_and_validate(validated)

        # Validate this object itself
        self.__finalize_and_validate__()
