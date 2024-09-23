import json
import re
from typing import Any, Dict, List, Optional, TypeVar, Union

from typing_extensions import Self

from oumi.core.configs import JudgeConfig
from oumi.core.inference import BaseInferenceEngine
from oumi.core.types.turn import Conversation, Message, Role, TemplatedMessage
from oumi.inference import (
    AnthropicInferenceEngine,
    LlamaCppInferenceEngine,
    RemoteInferenceEngine,
)
from oumi.utils.str_utils import str_to_bool

T = TypeVar("T", bound=TemplatedMessage)


class JudgeInput(TemplatedMessage):
    role: Role = Role.USER
    template: str = """<request>{{ request }}</request>
{% if context %}<context>{{ context }}</context>{% endif %}
{% if response %}<response>{{ response }}</response>{% endif %}
"""

    request: str
    response: Optional[str] = None
    context: Optional[str] = None


class JudgeOutput(TemplatedMessage):
    role: Role = Role.ASSISTANT
    template: str = (
        "<explanation>{{explanation}}</explanation><judgement>{{judgement}}</judgement>"
    )

    judgement: Optional[str]
    explanation: Optional[str] = None

    @classmethod
    def from_model_output(cls, raw_judgement: Optional[str]) -> Optional[Self]:
        """Parses the judgement."""
        if not raw_judgement:
            return None

        explanation_match = re.search(
            r"<explanation>(.*?)</explanation>", raw_judgement, re.DOTALL
        )
        judgment_match = re.search(
            r"<judgement>(.*?)</judgement>", raw_judgement, re.DOTALL
        )

        explanation = explanation_match.group(1).strip() if explanation_match else None
        judgment = judgment_match.group(1).strip() if judgment_match else None

        return cls(explanation=explanation, judgement=judgment)

    @classmethod
    def from_json_output(cls, raw_judgement: Optional[str]) -> Optional[Self]:
        """Parses the judgement from JSON."""
        if not raw_judgement:
            return None

        try:
            judgement_data = json.loads(raw_judgement)
            explanation = judgement_data.get("explanation")
            judgement = judgement_data.get("judgement")
            return cls(explanation=explanation, judgement=judgement)
        except json.JSONDecodeError:
            return None

    @property
    def label(self):
        """Convert the judgement to a boolean or Likert scale label.

        Returns:
            bool or int or None: The boolean or Likert scale interpretation of the
                judgement if present, otherwise None.
        """
        if self.judgement:
            if self.judgement.lower() in ["true", "false"]:
                return str_to_bool(self.judgement)
            elif self.judgement.isdigit():
                return int(self.judgement)
        return None


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
        self,
        conversations: Union[List[Conversation], List[Dict[str, Any]], List[T]],
    ) -> List[Conversation]:
        """Judge the given conversations."""
        judge_inputs = []
        for conversation in conversations:
            if isinstance(conversation, dict):
                judge_inputs.append(JudgeInput(**conversation))
            elif isinstance(conversation, TemplatedMessage):
                judge_inputs.append(conversation)
            elif isinstance(conversation, Conversation):
                judge_inputs.append(self._verify_conversation(conversation))
            else:
                raise ValueError(f"Unsupported conversation type: {type(conversation)}")

        all_prompts = {}
        for attribute_name in self.config.attributes:
            all_prompts[attribute_name] = []

        for judge_input in judge_inputs:
            prompts = self.generate_prompts(judge_input)
            for attribute_name, prompt in prompts.items():
                all_prompts[attribute_name].append(prompt)

        judged_conversations = self._infer_attributes(all_prompts)

        results = []
        for attribute_name, conversations in judged_conversations.items():
            for conversation in conversations:
                judgement = conversation.messages[-1].content
                parsed_judgement = JudgeOutput.from_json_output(judgement)
                conversation.metadata["parsed_judgement"] = (
                    str(parsed_judgement.label) if parsed_judgement else None
                )
                results.append(conversation)

        return results

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
        output = JudgeOutput.from_json_output(judgement)
        return output.label if output else None

    def _verify_conversation(self, conversation: Conversation) -> JudgeInput:
        judgement_conversation = [
            conversation.first_message(Role.SYSTEM),
            conversation.last_message(Role.USER),
            conversation.last_message(Role.ASSISTANT),
        ]

        request = (
            judgement_conversation[1].content or "" if judgement_conversation[1] else ""
        )

        return JudgeInput(
            request=request,
            response=judgement_conversation[2].content
            if judgement_conversation[2]
            else "",
            context=judgement_conversation[0].content
            if judgement_conversation[0]
            else "",
        )

    def _infer(self, conversations: List[Conversation]) -> List[Conversation]:
        """Judge a single attribute."""
        metadatas = [convo.metadata for convo in conversations]

        responses = self.inference_engine.infer(
            input=conversations, generation_config=self.config.generation
        )

        assert len(responses) == len(metadatas)

        for response, metadata in zip(responses, metadatas):
            response.metadata.update(metadata)

        return responses

    def _infer_attributes(
        self, conversations: Dict[str, List[Conversation]]
    ) -> Dict[str, List[Conversation]]:
        """Judge a prompt."""
        return {
            attribute_name: self._infer(attribute_conversations)
            for attribute_name, attribute_conversations in conversations.items()
        }

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
