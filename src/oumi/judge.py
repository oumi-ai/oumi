import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pydantic
from jinja2 import Template
from omegaconf import MISSING

from oumi.core.configs import BaseConfig, GenerationConfig, ModelParams, RemoteParams
from oumi.core.configs.params.base_params import BaseParams
from oumi.core.inference import BaseInferenceEngine
from oumi.core.types.turn import Conversation, Message, Role
from oumi.inference import RemoteInferenceEngine
from oumi.utils.io_utils import load_file
from oumi.utils.logging import logger


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

    @property
    def message(self) -> Message:
        """Returns the message in oumi format."""
        return Message(content=self.content, role=self.role)


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


class JudgeSpec(pydantic.BaseModel):
    system_prompt: str
    examples: List[Union[JudgeInput, JudgeOutput]] = field(default_factory=list)

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


class JudgeAttributeValueType(str, Enum):
    """The type of the attribute."""

    BOOL = "bool"
    """The attribute is a boolean."""

    CATEGORICAL = "categorical"
    """The attribute is a categorical."""

    LIKERT = "likert"
    """The attribute is a Likert scale."""


@dataclass
class JudgeAttribute(BaseParams):
    """Configuration parameters for the judge."""

    name: str = MISSING
    """The name of the attribute."""

    value_type: JudgeAttributeValueType = JudgeAttributeValueType.BOOL
    """The type of the attribute."""

    few_shots: int = -1
    """The template to use for the judge."""

    spec_path: str = MISSING
    """The path to the specification file."""


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
            spec = JudgeSpec.model_validate_json(load_file(attribute.spec_path))
            messages = spec.messages
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
        MATCH_PATTERN = r"*<judgment>.*</judgment>*"

        if not full_answer:
            logger.error(f"Full Answer ERROR: {full_answer}")
            return None

        answer_match = re.search(MATCH_PATTERN, full_answer)
        if not answer_match:
            logger.error(f"Answer ERROR: {full_answer}")
            return None

        answer = answer_match.group(0).replace("<answer>", "").replace("</answer>", "")

        if answer[:3].lower() == "yes":
            return True
        elif answer[:2].lower() == "no":
            return False
        else:
            logger.error(f"Extraction ERROR: {full_answer}")
            return None


def _get_default_judge_config() -> JudgeConfig:
    oumi_top_dir = Path(__file__).parent.resolve()
    judges_directory = oumi_top_dir / "judges"

    config = JudgeConfig(
        attributes=[
            JudgeAttribute(
                name="helpful",
                spec_path=str(judges_directory / "helpful.json"),
                value_type=JudgeAttributeValueType.BOOL,
                few_shots=2,
            ),
        ],
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
    print("Formatted message:")
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
