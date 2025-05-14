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

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Generic, Optional, TypeVar

import pydantic

from oumi.core.configs import BaseConfig
from oumi.core.configs.inference_engine_type import InferenceEngineType
from oumi.core.configs.params.generation_params import GenerationParams
from oumi.core.configs.params.model_params import ModelParams
from oumi.core.configs.params.remote_params import RemoteParams
from oumi.core.types.conversation import Conversation, Message, Role, TemplatedMessage


class JudgeAttributeValueType(str, Enum):
    """Enumeration of possible value types for judge attributes."""

    BOOL = "bool"
    """Boolean value type."""

    CATEGORICAL = "categorical"
    """Categorical value type."""

    LIKERT_5 = "likert-5"
    """Likert scale with 5 points value type."""


T = TypeVar("T", bound=TemplatedMessage)


class JudgeAttribute(pydantic.BaseModel, Generic[T]):
    """Attributes for the judge.

    Example:
        >>> attribute = JudgeAttribute( # doctest: +SKIP
        ...     name="helpful",
        ...     system_prompt="You are an impartial judge.",
        ...     examples=[
        ...         TemplatedMessage(
        ...             role=Role.USER,
        ...             request="What is the capital of France?",
        ...             response="The capital of France is Paris.",
        ...         ),
        ...         TemplatedMessage(
        ...             role=Role.ASSISTANT,
        ...             response="True",
        ...         ),
        ...     ],
        ...     value_type=JudgeAttributeValueType.BOOL,
        ...     limit_examples=5,
        ... )
        >>> print(attribute.name) # doctest: +SKIP
        helpful
    """

    name: str
    """The name of the attribute being judged."""

    system_prompt: str
    """The system prompt for the judge."""

    examples: list[T] = field(default_factory=list)
    """A list of few-shot example inputs and judgements."""

    value_type: JudgeAttributeValueType = JudgeAttributeValueType.BOOL
    """The type of value for the attribute."""

    limit_examples: Optional[int] = 5
    """The maximum number of examples to use.

    This is an optional parameter that limits the number of examples to be used for
    judging the attribute. If not specified, the default is 5.
    """

    @property
    def conversation(self) -> Conversation:
        """Returns the judgement conversation in oumi format.

        This will include the judge system prompt, and any few-shot examples.
        """
        return Conversation(messages=self.messages)

    @property
    def messages(self) -> list[Message]:
        """Returns the messages in oumi format.

        This will include the judge system prompt, and any few-shot examples.
        """
        messages = [Message(content=self.system_prompt, role=Role.SYSTEM)]
        return messages + [e.message for e in self.examples]

    @classmethod
    def load(cls: type, filename: str) -> "JudgeAttribute[T]":
        """Loads the judge attribute config from a file."""
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(path)
        return cls.model_validate_json(path.read_text())


@dataclass
class JudgeConfig(BaseConfig):
    """Configuration for the Judge.

    This class holds the configuration for the Judge,
      including the attributes to judge, the model parameters,
      and the text generation parameters.

    Examples:
        >>> attributes = {
        ...     "helpful": JudgeAttribute( # doctest: +SKIP
        ...         name="helpful",
        ...         system_prompt="Is this answer helpful?",
        ...         examples=[
        ...             TemplatedMessage(
        ...                 role=Role.USER,
        ...                 request="What is the capital of France?",
        ...                 response="The capital of France is Paris.",
        ...             ),
        ...             TemplatedMessage(
        ...                 role=Role.ASSISTANT,
        ...                 response="True",
        ...             ),
        ...         ],
        ...     ),
        ...     "honest": JudgeAttribute(
        ...         name="honest",
        ...         system_prompt="Is this answer honest?",
        ...         examples=[]
        ...     )
        ... }
        >>> model_params = ModelParams(model_name="example-model")
        >>> generation_params = GenerationParams(max_new_tokens=100) # doctest: +SKIP
        >>> judge_config = JudgeConfig( # doctest: +SKIP
        ...     attributes=attributes,
        ...     model=model_params,
        ...     generation=generation_params
        ... )
    """

    attributes: dict[str, JudgeAttribute] = field(default_factory=dict)
    """The attributes to judge."""

    model: ModelParams = field(default_factory=ModelParams)
    """Parameters for the model used in inference."""

    generation: GenerationParams = field(default_factory=GenerationParams)
    """Parameters for text generation during inference."""

    engine: InferenceEngineType = field(default=InferenceEngineType.NATIVE)
    """The inference engine to use for generation."""

    remote_params: Optional[RemoteParams] = None
    """Parameters for running inference against a remote API."""
