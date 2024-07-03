from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel


class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    id: Optional[str] = None
    content: str
    role: Role


class Conversation(BaseModel):
    conversation_id: Optional[str] = None
    messages: List[Message]
    metadata: Dict[str, str] = {}

    def __getitem__(self, item):
        """Get the message at the specified index.

        Args:
            item (int): The index of the message to retrieve.

        Returns:
            Any: The message at the specified index.
        """
        return self.messages[item]
