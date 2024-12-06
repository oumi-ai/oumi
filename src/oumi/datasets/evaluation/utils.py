from oumi.core.types.conversation import Conversation, Role

DEFAULT_INSTRUCTION_FIELD_NAME = "instruction"
DEFAULT_OUTPUT_FIELD_NAME = "output"


def conversation_to_openai_format(
    conversation: Conversation,
    instruction_field_name: str = DEFAULT_INSTRUCTION_FIELD_NAME,
    output_field_name: str = DEFAULT_OUTPUT_FIELD_NAME,
) -> dict:
    """Converts an Oumi `Conversation` to Open AI format.

    Converts an Oumi single-turn conversation to a dictionary with keys `instruction`
    and `output`. If the first message is a System Instruction, it is ignored. Any
    fields in the conversation's metadata are also retained as dict entries.
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


def list_conversations_to_openai_format(
    list_conversations: list[Conversation],
    instruction_field_name: str = DEFAULT_INSTRUCTION_FIELD_NAME,
    output_field_name: str = DEFAULT_OUTPUT_FIELD_NAME,
) -> list[dict]:
    """Converts a list of conversations to openai format (list of dictionaries)."""
    return [
        conversation_to_openai_format(
            conversation=conversation,
            instruction_field_name=instruction_field_name,
            output_field_name=output_field_name,
        )
        for conversation in list_conversations
    ]
