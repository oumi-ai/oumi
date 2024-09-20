import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

import pydantic
from jinja2 import Template

from oumi.core.configs import BaseConfig, GenerationConfig, ModelParams, RemoteParams
from oumi.core.inference import BaseInferenceEngine
from oumi.core.types.turn import Conversation, Message, Role
from oumi.inference import (
    AnthropicInferenceEngine,
    LlamaCppInferenceEngine,
    RemoteInferenceEngine,
)
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

    def parse_label(self, raw_judgement: Optional[str]) -> Optional[bool]:
        """Parses the judgement."""
        if not raw_judgement:
            return None

        explanation_match = re.search(
            r"<explanation>(.*?)</explanation>", raw_judgement, re.DOTALL
        )
        judgment_match = re.search(
            r"<judgement>(.*?)</judgement>", raw_judgement, re.DOTALL
        )

        _explanation = explanation_match.group(1).strip() if explanation_match else None
        judgment = judgment_match.group(1).strip() if judgment_match else None

        return str_to_bool(judgment) if judgment else None


@dataclass
class JudgeConfig(BaseConfig):
    attributes: Dict[str, JudgeAttribute] = field(default_factory=dict)
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
        self, conversations: Union[List[Conversation], Dict[str, List[Conversation]]]
    ) -> Union[List[Conversation], Dict[str, List[Conversation]]]:
        """Judge a prompt."""
        if isinstance(conversations, list):
            return self.judge_attribute(conversations)
        else:
            return {
                attribute_name: self.judge_attribute(attribute_conversations)
                for attribute_name, attribute_conversations in conversations.items()
            }

    def judge_attribute(self, conversations: List[Conversation]) -> List[Conversation]:
        """Judge a single attribute."""
        metadatas = [convo.metadata for convo in conversations]

        responses = self.inference_engine.infer(
            input=conversations, generation_config=self.config.generation
        )

        assert len(responses) == len(metadatas)

        for response, metadata in zip(responses, metadatas):
            response.metadata.update(metadata)

        return responses

    def generate_prompts(self, judge_input: JudgeInput) -> Dict[str, Conversation]:
        """Generate judge prompts for a dataset."""
        prompts = {}

        for attribute_name, attribute in self.config.attributes.items():
            messages = attribute.messages.copy()
            messages.append(Message(content=judge_input.content, role=Role.USER))

            prompts[attribute.name] = Conversation(
                messages=messages,
                metadata={
                    "judge_attribute_name": attribute_name,
                    "judge_name": "oumi_v1",
                },
            )

        return prompts

    def parse_judgement(
        self, judgement: Optional[str], attribute_name: str
    ) -> Optional[bool]:
        """Parse the judgement."""
        return self.config.attributes[attribute_name].parse_label(judgement)

    def _create_inference_engine(self, config: JudgeConfig) -> BaseInferenceEngine:
        """Create the inference engine."""
        # TODO: Initialize the appropriate inference engine based on the config
        # For now, we default to the remote inference engine
        # Users can override this method to provide their own inference engine
        # to the constructor of the Judge class.
        if config.model.model_name.endswith(".gguf"):
            return LlamaCppInferenceEngine(config.model)
        elif config.model.model_name:
            return AnthropicInferenceEngine(config.model)
        return RemoteInferenceEngine(self.config.model)


def _get_default_judge_config() -> JudgeConfig:
    oumi_top_dir = Path(__file__).parent.resolve()
    judges_directory = oumi_top_dir / "judges" / "oumi_v1"

    attribute_names = ["helpful", "honest", "safe", "valid"]
    attributes = {
        attribute: JudgeAttribute.load(str(judges_directory / f"{attribute}.json"))
        for attribute in attribute_names
    }

    config = JudgeConfig(
        attributes=attributes,
        model=ModelParams(
            model_name="claude-3-5-sonnet-20240620",
        ),
        # generation=GenerationConfig(
        #     max_new_tokens=1024,
        #     remote_params=RemoteParams(
        #         api_url="http://localhost:1234/v1/chat/completions",
        #         max_retries=2,
        #     ),
        # ),
        generation=GenerationConfig(
            max_new_tokens=1024,
            remote_params=RemoteParams(
                api_url="https://api.anthropic.com/v1/messages",
                api_key_env_varname="ANTHROPIC_API_KEY",
                max_retries=0,
            ),
        ),
    )
    return config


def _get_default_local_judge_config() -> JudgeConfig:
    oumi_top_dir = Path(__file__).parent.resolve()
    judges_directory = oumi_top_dir / "judges" / "oumi_v1"

    attribute_names = ["helpful", "honest", "safe", "valid"]
    attributes = {
        attribute: JudgeAttribute.load(str(judges_directory / f"{attribute}.json"))
        for attribute in attribute_names
    }
    config = JudgeConfig(
        attributes=attributes,
        model=ModelParams(
            model_name=str(judges_directory / "Q4_K_M-00001-of-00001.gguf"),
        ),
        generation=GenerationConfig(
            max_new_tokens=1024,
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
    judgements = judge.judge_attribute([conversation])

    judgement = judgements[0].messages[-1].content
    print("\nRaw Judgement:")
    print(judgements[0])

    # Parsed judgment
    bool_result = judge.parse_judgement(judgement, "helpful")
    print("\nExtracted Boolean Result:")
    print(bool_result)


# Example usage
if __name__ == "__main__":
    test()
