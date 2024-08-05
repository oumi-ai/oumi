import dataclasses
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Set, Tuple


@dataclass
class BaseParams:
    def validate(self, validated: Optional[Set[int]] = None) -> None:
        """Recursively validates the parameters."""
        if validated is None:
            validated = set()

        # If this object has already been validated, return immediately
        if id(self) in validated:
            return
        validated.add(id(self))

        # Validate the children of this object.
        # Note that we only support one level of nesting.
        # For example: `List[BaseParams]` is supported, but not `List[List[BaseParams]]`
        for _attr_name, attr_value in self:
            if isinstance(attr_value, BaseParams):
                attr_value.validate(validated)
            elif isinstance(attr_value, list):
                for item in attr_value:
                    if isinstance(item, BaseParams):
                        item.validate(validated)
            elif isinstance(attr_value, dict):
                for item in attr_value.values():
                    if isinstance(item, BaseParams):
                        item.validate(validated)

        # Validate this object itself
        self.__validate__()

    def __validate__(self) -> None:
        """Validates the parameters of this object.

        This method can be overridden by subclasses to implement custom
        validation logic.

        In case of validation errors, this method should raise a `ValueError`
        or other appropriate exception.
        """

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        """Returns an iterator over field names and values.

        Note: for an attribute to be a field, it must be declared in the
        dataclass definition and have a type annotation.
        """
        for param in dataclasses.fields(self):
            yield param.name, getattr(self, param.name)
