import dataclasses
from dataclasses import dataclass
from typing import Any, Iterator, Tuple


@dataclass
class BaseParams:
    def validate(self):
        """Validates final config params."""
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, BaseParams):
                attr_value.validate()
            elif isinstance(attr_value, list):
                for item in attr_value:
                    if isinstance(item, BaseParams):
                        item.validate()
            elif isinstance(attr_value, dict):
                for item in attr_value.values():
                    if isinstance(item, BaseParams):
                        item.validate()

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        """Returns an iterator over field names and values."""
        for param in dataclasses.fields(self):
            yield param.name, getattr(self, param.name)
