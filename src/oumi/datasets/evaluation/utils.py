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

from oumi.core.types.conversation import Conversation, Role

_DEFAULT_INSTRUCTION_FIELD_NAME = "instruction"
_DEFAULT_OUTPUT_FIELD_NAME = "output"


def conversation_to_alpaca_format(
    conversation: Conversation,
    instruction_field_name: str = _DEFAULT_INSTRUCTION_FIELD_NAME,
    output_field_name: str = _DEFAULT_OUTPUT_FIELD_NAME,
) -> dict:
    """Converts an Oumi `Conversation` to Alpaca format.

    Converts an Oumi single-turn conversation to a dictionary with keys `instruction`
    and `output`. If the first message is a System Instruction, it is ignored. Any
    fields in the conversation's metadata are also retained as dict entries.

    Alpaca format (see: https://huggingface.co/datasets/tatsu-lab/alpaca_eval):
        {
            "instruction": <`str` prompt/request to a Large Language Model (LLM)>,
            "output": <`str` response by a Large Language Model (LLM)>,
        }
    """
    # Ensure the number of messages is correct.
    if len(conversation.messages) not in (2, 3):
        raise ValueError("Only single-turn conversations are currently supported.")

    # Ensure that the first message is an SI (if we have 3 messages).
    if len(conversation.messages) == 3:
        if conversation.messages[0].role != Role.SYSTEM:
            raise ValueError(
                f"Role of first message is `{conversation.messages[0].role}`, "
                "while `Role.SYSTEM` is expected for conversations of 3 messages."
            )

    # Extract the instruction and output.
    instruction = conversation.messages[-2]
    output = conversation.messages[-1]

    # Ensure that the roles for {instruction, output} are correct.
    if instruction.role != Role.USER:
        raise ValueError("Role of `instruction` should be `Role.USER`")
    if output.role != Role.ASSISTANT:
        raise ValueError("Role of `output` should be `Role.ASSISTANT`")

    # Extract metadata to add.
    metadata = {}
    if conversation.conversation_id is not None:
        metadata["conversation_id"] = conversation.conversation_id
    metadata.update(conversation.metadata)
    metadata.pop(instruction_field_name, None)
    metadata.pop(output_field_name, None)

    # Create a dictionary with the instruction, output, metadata.
    conversations_dict = {
        instruction_field_name: instruction.content,
        output_field_name: output.content,
    }
    conversations_dict.update(metadata)
    return conversations_dict


def conversations_to_alpaca_format(
    conversations: list[Conversation],
    instruction_field_name: str = _DEFAULT_INSTRUCTION_FIELD_NAME,
    output_field_name: str = _DEFAULT_OUTPUT_FIELD_NAME,
) -> list[dict]:
    """Converts a list of conversations to Alpaca format (list of dictionaries)."""
    return [
        conversation_to_alpaca_format(
            conversation=conversation,
            instruction_field_name=instruction_field_name,
            output_field_name=output_field_name,
        )
        for conversation in conversations
    ]
