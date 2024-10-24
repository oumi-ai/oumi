from typing import Any, Optional

from typing_extensions import override

from oumi.core.types.conversation import Conversation, Role, TemplatedMessage
from oumi.judges.base_judge import BaseJudge, BaseJudgeOutput
from oumi.utils.str_utils import str_to_bool


class OumiJudgeInput(TemplatedMessage):
    role: Role = Role.USER
    template: str = """<request>{{ request }}</request>
{% if context %}<context>{{ context }}</context>{% endif %}
{% if response %}<response>{{ response }}</response>{% endif %}
"""

    request: str
    response: Optional[str] = None
    context: Optional[str] = None


class OumiJudgeOutput(BaseJudgeOutput):
    role: Role = Role.ASSISTANT
    template: str = (
        "<explanation>{{explanation}}</explanation><judgement>{{judgement}}</judgement>"
    )

    judgement: Optional[str] = None
    explanation: Optional[str] = None

    @property
    @override
    def label(self):
        """Convert the judgement to a boolean or Likert scale label."""
        if self.judgement:
            if self.judgement.isdigit():
                return int(self.judgement)

            try:
                return str_to_bool(self.judgement)
            except ValueError:
                return None
        return None


class OumiXmlJudge(BaseJudge):
    def _transform_conversation_input(
        self, conversation: Conversation
    ) -> OumiJudgeInput:
        user_prompt = conversation.last_message(Role.USER)
        assistant_prompt = conversation.last_message(Role.ASSISTANT)

        if user_prompt and user_prompt.content:
            request = user_prompt.content
        else:
            raise ValueError("No user prompt found in conversation")

        response = None
        if assistant_prompt:
            response = assistant_prompt.content
        else:
            response = None

        return OumiJudgeInput(request=request, response=response)

    def _transform_dict_input(self, raw_input: dict[str, Any]) -> OumiJudgeInput:
        return OumiJudgeInput(**raw_input)

    def _transform_model_output(self, model_output) -> Optional[OumiJudgeOutput]:
        return OumiJudgeOutput.from_xml_output(model_output)
