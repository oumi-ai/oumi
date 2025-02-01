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

from typing import Any, Optional

from typing_extensions import override

from oumi.core.types.conversation import Conversation, Message, Role, TemplatedMessage
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
        user_prompt: Optional[Message] = conversation.last_message(Role.USER)
        assistant_prompt: Optional[Message] = conversation.last_message(Role.ASSISTANT)

        if user_prompt is not None:
            if not user_prompt.contains_text_content_items_only():
                raise ValueError("User message contains non-text content!")
            request: str = user_prompt.compute_flattened_text_content()
        else:
            raise ValueError("No user prompt found in conversation")

        response: Optional[str] = None
        if assistant_prompt is not None:
            if not assistant_prompt.contains_text_content_items_only():
                raise ValueError("Assistant message contains non-text content!")
            response = assistant_prompt.compute_flattened_text_content()
        else:
            response = None

        return OumiJudgeInput(request=request, response=response)

    def _transform_dict_input(self, raw_input: dict[str, Any]) -> OumiJudgeInput:
        return OumiJudgeInput(**raw_input)

    def _transform_model_output(self, model_output) -> Optional[OumiJudgeOutput]:
        return OumiJudgeOutput.from_xml_output(model_output)
