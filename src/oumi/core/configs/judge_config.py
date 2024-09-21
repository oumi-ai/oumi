from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

import pydantic

from oumi.core.configs import (
    BaseConfig,
    GenerationConfig,
)
from oumi.core.configs.params.model_params import (
    ModelParams,
)
from oumi.core.types.turn import Conversation, Message, Role, TemplatedMessage


class JudgeAttributeValueType(str, Enum):
    """The type of the attribute."""

    BOOL = "bool"
    """The attribute is a boolean."""

    CATEGORICAL = "categorical"
    """The attribute is a categorical."""

    LIKERT_5 = "likert-5"
    """The attribute is a Likert scale."""


class JudgeAttribute(pydantic.BaseModel):
    """Configuration parameters for the judge."""

    name: str
    """The name of the attribute."""

    system_prompt: str

    examples: List[Union[TemplatedMessage, TemplatedMessage]] = field(
        default_factory=list
    )

    value_type: JudgeAttributeValueType = JudgeAttributeValueType.BOOL

    limit_examples: Optional[int] = 5

    @property
    def conversation(self) -> Conversation:
        """Returns the conversation in oumi format."""
        return Conversation(messages=self.messages)

    @property
    def messages(self) -> List[Message]:
        """Returns the messages in oumi format."""
        messages = [Message(content=self.system_prompt, role=Role.SYSTEM)]
        for example in self.examples:
            messages.append(example.message)
        return messages

    @classmethod
    def load(cls: Type, filename: str) -> "JudgeAttribute":
        """Loads the judge attribute from a file."""
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(path)
        return cls.model_validate_json(path.read_text())


@dataclass
class JudgeConfig(BaseConfig):
    attributes: Dict[str, JudgeAttribute] = field(default_factory=dict)
    """The attributes to judge."""

    model: ModelParams = field(default_factory=ModelParams)
    """Configuration parameters for the model used in inference."""

    generation: GenerationConfig = field(default_factory=GenerationConfig)
    """Configuration parameters for text generation during inference."""
