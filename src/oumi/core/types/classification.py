from typing import Optional

import pydantic


class Classification(pydantic.BaseModel):
    """Represents a conversation, which is a sequence of messages."""

    uid: Optional[str] = None
    """Optional unique identifier for the sample.

    This attribute can be used to assign a specific identifier to the conversation,
    which may be useful for tracking or referencing conversations in a larger context.
    """

    input: str
    """List of Message objects that make up the conversation."""

    label: int
    """Label ID corresponding to a particular class."""

    def to_dict(self):
        """Converts the conversation to a dictionary."""
        return self.model_dump(
            mode="json", exclude_unset=True, exclude_defaults=False, exclude_none=True
        )

    def append_id_to_string(self, s: str) -> str:
        """Appends conversation ID to a string.

        Can be useful for log or exception errors messages to allow users
        to identify relevant conversation.
        """
        if not self.uid:
            return s
        suffix = f"Unique id: {self.uid}."
        return (s.strip() + " " + suffix) if s else suffix

    @classmethod
    def from_dict(cls, data: dict) -> "Classification":
        """Converts a dictionary to a classification."""
        return cls.model_validate(data)

    def to_json(self) -> str:
        """Converts the classification to a JSON string."""
        return self.model_dump_json(
            exclude_unset=True, exclude_defaults=False, exclude_none=True
        )

    @classmethod
    def from_json(cls, data: str) -> "Classification":
        """Converts a JSON string to a classification."""
        return cls.model_validate_json(data)

    def __repr__(self) -> str:
        """Returns a string representation of the classification."""
        return str(self.to_dict())
