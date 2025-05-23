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


def extract_question_images_answer_from_single_turn_conversation(
    example: dict,
) -> tuple[str, list, str]:
    """Finds question, answer, and optional images in a single-turn conversation.

    Args:
        example: A dictionary containing the conversation JSON.

    Returns:
        A tuple containing the question, images, and answer.
        The list of images is empty for text-only conversations.
    """
    if "conversation_json" not in example:
        raise ValueError(
            f"Example doesn't contain 'conversation_json' key. "
            f"Available keys: {example.keys()}"
        )

    conversation_json = example["conversation_json"]
    conversation = Conversation.from_json(conversation_json)

    user_messages = conversation.filter_messages(role=Role.USER)
    if len(user_messages) != 1:
        raise ValueError(f"Expected 1 user message, but got {len(user_messages)}.")

    assistant_messages = conversation.filter_messages(role=Role.ASSISTANT)
    if len(assistant_messages) != 1:
        raise ValueError(
            f"Expected 1 assistant message, but got {len(assistant_messages)}."
        )

    user_message = user_messages[0]
    assistant_message = assistant_messages[0]
    prompt: str = user_message.text_content_items[-1].content or ""
    images = [{"bytes": item.binary} for item in user_message.image_content_items]
    answer: str = assistant_message.text_content_items[-1].content or ""

    if len(images) > 0:
        # TODO: Generalize. This only works for QwenVL 2.5, which is the only
        # VLM supported by verl as of 2025-05-15.
        if not prompt.startswith("<image>"):
            prompt = "<image>" + prompt
    return (prompt, images, answer)


def try_prepare_grpo_example(
    example: dict,
) -> dict:
    """Prepares an example for GRPO_TRL processing.

    This function checks if the input example is one of known special cases
    e.g., SFT example, and transforms it into a GRPO compatible format.
    Otherwise, it returns the original example.

    Args:
        example (dict): The input example.

    Returns:
        GRPO compatible example, or an original example.
    """
    if not isinstance(example, dict):
        return example

    if "conversation_json" in example:
        prompt, images, answer = (
            extract_question_images_answer_from_single_turn_conversation(example)
        )
        if len(images) > 0:
            raise ValueError(
                f"Image content is not supported in GRPO_TRL yet. "
                f"Found {len(images)} image(s) in an example."
            )
        return {
            "prompt": prompt,
            "completion": answer,
        }

    return example
