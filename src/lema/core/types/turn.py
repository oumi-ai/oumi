from enum import Enum
from typing import Dict, List, Optional

import pydantic


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(pydantic.BaseModel):
    id: Optional[str] = None
    content: str
    role: Role


class Conversation(pydantic.BaseModel):
    conversation_id: Optional[str] = None
    messages: List[Message]
    metadata: Dict[str, str] = {}

    def __getitem__(self, idx: int) -> Message:
        """Get the message at the specified index.

        Args:
            idx (int): The index of the message to retrieve.

        Returns:
            Any: The message at the specified index.
        """
        return self.messages[idx]
