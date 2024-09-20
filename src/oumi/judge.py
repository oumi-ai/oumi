from typing import Dict, List, Optional, Union

from oumi.core.configs import JudgeConfig
from oumi.core.inference import BaseInferenceEngine
from oumi.core.types.turn import Conversation, Message, Role, TemplatedMessage
from oumi.inference import (
    AnthropicInferenceEngine,
    LlamaCppInferenceEngine,
    RemoteInferenceEngine,
)
from oumi.judges.judge_zoo import _get_default_judge_config
from oumi.utils.logging import logger


class JudgeInput(TemplatedMessage):
    role: Role = Role.USER
    request: str
    response: Optional[str] = None
    context: Optional[str] = None
    template: str = """<request>{{ request }}</request>
{% if context %}<context>{{ context }}</context>{% endif %}
{% if response %}<response>{{ response }}</response>{% endif %}
"""


class JudgeOutput(TemplatedMessage):
    role: Role = Role.ASSISTANT
    judgement: str
    explanation: Optional[str] = None
    template: str = (
        "<explanation>{{explanation}}</explanation><judgement>{{judgement}}</judgement>"
    )


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


if __name__ == "__main__":
    test()
