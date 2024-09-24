import re
from typing import Any, Dict, Optional

from oumi.core.types.turn import Conversation, Role, TemplatedMessage
from oumi.judges.base_judge import BaseJudge, BaseJudgeOutput


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

    judgement: Optional[str]
    explanation: Optional[str] = None

    def custom_from_model_output(self, raw_judgement: Optional[str]):
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

        return OumiJudgeOutput(
            explanation=explanation, judgement=judgment, raw_judgement=raw_judgement
        )


class OumiXmlJudge(BaseJudge):
    def _transform_conversation_input(
        self, conversation: Conversation
    ) -> OumiJudgeInput:
        judgement_conversation = [
            conversation.first_message(Role.SYSTEM),
            conversation.last_message(Role.USER),
            conversation.last_message(Role.ASSISTANT),
        ]

        request = (
            judgement_conversation[1].content or "" if judgement_conversation[1] else ""
        )

        return OumiJudgeInput(
            request=request,
            response=judgement_conversation[2].content
            if judgement_conversation[2]
            else "",
            context=judgement_conversation[0].content
            if judgement_conversation[0]
            else "",
        )

    def _transform_dict_input(self, raw_input: Dict[str, Any]) -> OumiJudgeInput:
        return OumiJudgeInput(**raw_input)

    def _transfrorm_model_output(self, model_output) -> Optional[OumiJudgeOutput]:
        return OumiJudgeOutput.from_model_output(model_output)
