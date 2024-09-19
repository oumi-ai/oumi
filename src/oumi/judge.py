import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union

import pydantic
from jinja2 import Template
from omegaconf import MISSING

from oumi.core.configs import BaseConfig, GenerationConfig, ModelParams, RemoteParams
from oumi.core.inference import BaseInferenceEngine
from oumi.core.types.turn import Conversation, Message, Role
from oumi.inference import RemoteInferenceEngine
from oumi.utils.logging import logger
from oumi.utils.str_utils import str_to_bool


class BaseJudgeMessage(pydantic.BaseModel):
    template: str
    role: Role

    @property
    def content(self) -> str:
        """Renders the content of the message."""
        template = Template(self.template)

        fields = self.model_dump()
        fields.pop("template")  # remove the template from the fields

        return template.render(**fields).strip()

    @content.setter
    def content(self, value: str):
        raise RuntimeError("content is read-only")

    @property
    def message(self) -> Message:
        """Returns the message in oumi format."""
        content = str(self.content)
        return Message(content=content, role=self.role)


class JudgeInput(BaseJudgeMessage):
    role: Role = Role.USER
    request: str
    response: Optional[str] = None
    context: Optional[str] = None
    template: str = """<request>{{ request }}</request>
{% if context %}<context>{{ context }}</context>{% endif %}
{% if response %}<response>{{ response }}</response>{% endif %}
"""


class JudgeOutput(BaseJudgeMessage):
    role: Role = Role.ASSISTANT
    judgement: str
    explanation: Optional[str] = None
    template: str = (
        "<explanation>{{explanation}}</explanation><judgement>{{judgement}}</judgement>"
    )


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

    examples: List[Union[JudgeInput, JudgeOutput]] = field(default_factory=list)

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
    attributes: List[JudgeAttribute] = MISSING
    """The attributes to judge."""

    model: ModelParams = field(default_factory=ModelParams)
    """Configuration parameters for the model used in inference."""

    generation: GenerationConfig = field(default_factory=GenerationConfig)
    """Configuration parameters for text generation during inference."""


class Judge:
    def __init__(
        self,
        config: JudgeConfig,
        inference_engine: Optional[BaseInferenceEngine] = None,
    ):
        """Initialize the Judge."""
        self.config = config

        if inference_engine is None:
            self.inference_engine = self._create_inference_engine(config)
        else:
            self.inference_engine = inference_engine

    def judge(
        self, conversations: List[Conversation]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Judge a prompt."""
        response = self.inference_engine.infer(
            input=conversations, generation_config=self.config.generation
        )[0]
        return response.messages[-1].content, None

    def generate_prompts(self, judge_input: JudgeInput) -> Dict[str, Conversation]:
        """Generate judge prompts for a dataset."""
        prompts = {}

        for attribute in self.config.attributes:
            messages = attribute.messages.copy()
            messages.append(Message(content=judge_input.content, role=Role.USER))

            prompts[attribute.name] = Conversation(messages=messages)

        return prompts

    def _create_inference_engine(self, config: JudgeConfig) -> BaseInferenceEngine:
        """Create the inference engine."""
        # TODO: Initialize the appropriate inference engine based on the config
        # For now, we default to the remote inference engine
        # Users can override this method to provide their own inference engine
        # to the constructor of the Judge class.
        return RemoteInferenceEngine(self.config.model)

    @staticmethod
    def _extract_bool_answer(full_answer: str) -> Optional[bool]:
        explanation_match = re.search(
            r"<explanation>(.*?)</explanation>", full_answer, re.DOTALL
        )
        judgment_match = re.search(
            r"<judgement>(.*?)</judgement>", full_answer, re.DOTALL
        )

        explanation = explanation_match.group(1).strip() if explanation_match else None
        judgment = judgment_match.group(1).strip() if judgment_match else None
        print(explanation)
        return str_to_bool(judgment) if judgment else None


def _get_default_judge_config() -> JudgeConfig:
    oumi_top_dir = Path(__file__).parent.resolve()
    judges_directory = oumi_top_dir / "judges" / "oumi_v1"

    attribute_names = ["helpful", "honest", "safe", "valid"]
    attributes = [
        JudgeAttribute.load(str(judges_directory / f"{attribute}.json"))
        for attribute in attribute_names
    ]

    config = JudgeConfig(
        attributes=attributes,
        model=ModelParams(
            model_name="GPT-3.5-turbo",
        ),
        generation=GenerationConfig(
            max_new_tokens=100,
            remote_params=RemoteParams(
                api_url="http://localhost:1234/v1/chat/completions",
                max_retries=2,
            ),
        ),
    )
    return config


def test():
    """Tests the Judge class."""
    # Create a Judge instance
    judge = Judge(_get_default_judge_config())

    # Create a sample JudgeInput
    sample_input = JudgeInput(
        request="What is the capital of France?",
        response="The capital of France is Paris.",
        context="This is a geography question.",
    )

    # Test the to_message() method
    formatted_message = sample_input.content
    logger.info("Formatted message:")
    print(formatted_message)

    # Generate prompts
    prompts = judge.generate_prompts(sample_input)
    conversation = prompts["helpful"]

    print("Generated Prompts:")
    for message in conversation.messages:
        print(f"{message.role}: {message.content}")

    # Judge the prompts
    judgement, exception = judge.judge([conversation])
    print("\nRaw Judgement:")
    print(judgement)

    if exception:
        print("\nException:")
        print(exception)

    # Parsed judgment
    bool_result = judge._extract_bool_answer(judgement or "")
    print("\nExtracted Boolean Result:")
    print(bool_result)


# Example usage
if __name__ == "__main__":
    test()
