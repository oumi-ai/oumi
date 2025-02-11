# A tool to save Oumi Conversation-s from SFT datasets to a file.
#
# Sample usage:
#
# python save_conversations.py --name "HuggingFaceH4/ultrachat_200k" \
#   --split train_sft --max-conversations 100 -o conversations.jsonl
#
# python save_conversations.py --name "HuggingFaceM4/Docmatix" \
#   --subset zero-shot-exp --split train --max-conversations 100 \
#   -o conversations.jsonl

import argparse
import copy
from pathlib import Path
from typing import Any, Optional

import jsonlines
from tqdm import tqdm

from oumi.builders import build_tokenizer
from oumi.core.configs import ModelParams
from oumi.core.datasets import BaseSftDataset
from oumi.core.registry import REGISTRY
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer
from oumi.core.types import ContentItem, Conversation, Message, Role
from oumi.utils.logging import logger, update_logger_level


def _load_sft_dataset(
    dataset_name: str,
    *,
    dataset_path: Optional[str],
    dataset_subset: Optional[str],
    dataset_split: Optional[str],
    tokenizer: Optional[BaseTokenizer] = None,
    processor_name: Optional[str] = None,
    trust_remote_code: bool = False,
    dataset_kwargs: Optional[dict[str, Any]] = None,
) -> BaseSftDataset:
    """Loads a custom SFT dataset with the specified name and subset."""
    dataset_class = REGISTRY.get_dataset(dataset_name, subset=dataset_subset)

    if dataset_class is None:
        raise ValueError(
            f"Unrecognized dataset: '{dataset_name}' (subset: {dataset_subset})"
        )

    dataset_kwargs = copy.deepcopy(dataset_kwargs) if dataset_kwargs is not None else {}
    if processor_name:
        dataset_kwargs["processor_name"] = processor_name

    dataset = dataset_class(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        split=dataset_split,
        subset=dataset_subset,
        tokenizer=tokenizer,
        trust_remote_code=trust_remote_code,
        **dataset_kwargs,
    )
    if not isinstance(dataset, BaseSftDataset):
        raise ValueError(
            f"Dataset '{dataset_name}' is not a subclass of BaseSftDataset. "
            f"Actual type: {type(dataset)}"
        )
    return dataset


def _split_message_list_on_user(messages: list[Message]) -> list[list[Message]]:
    if len(messages) == 0:
        return []
    turn_start_indices: list[int] = []
    for idx, m in enumerate(messages):
        if m.role == Role.USER:
            turn_start_indices.append(idx)

    num_turns = len(turn_start_indices)
    if num_turns == 0:
        return [messages]

    # SYSTEM messages
    lead_slice = messages[0 : turn_start_indices[0]]
    result = []

    image_content_items: list[ContentItem] = []

    for idx, start_index in enumerate(turn_start_indices):
        turn_messages = copy.copy(lead_slice)

        user_message = messages[start_index]
        if idx == 0:
            image_content_items = user_message.image_content_items
        elif len(image_content_items) > 0 and not user_message.contains_images():
            # Copy images from the first turn
            user_message = Message(
                role=Role.USER,
                content=(image_content_items + user_message.text_content_items),
            )
        turn_messages.append(user_message)

        if idx + 1 >= num_turns:
            turn_messages.extend(messages[(start_index + 1) :])
        else:
            turn_messages.extend(
                messages[(start_index + 1) : turn_start_indices[idx + 1]]
            )
        result.append(turn_messages)

    return result


def _not_assistant_fn(m: Message):
    return m.role != Role.ASSISTANT


def _process_conversation(
    input_conversation: Conversation,
    split_multi_turn_to_single_turn: bool,
    drop_assistant_messages: bool,
) -> list[Conversation]:
    result: list[Conversation] = []
    if split_multi_turn_to_single_turn:
        proto_conversation = copy.copy(input_conversation)
        proto_conversation.messages = []
        for turn_messages in _split_message_list_on_user(input_conversation.messages):
            new_conversation = copy.copy(proto_conversation)
            new_conversation.messages = turn_messages
            result.append(new_conversation)
    else:
        result = [input_conversation]

    if drop_assistant_messages:
        for convo in result:
            convo.messages = convo.filter_messages(filter_fn=_not_assistant_fn)

    return result


def main(args):
    """The script's entry point."""
    dataset_name: str = args.name
    dataset_path: Optional[str] = args.path
    dataset_subset: Optional[str] = args.subset
    dataset_split: Optional[str] = args.split
    trust_remote_code: bool = args.trust_remote_code
    split_multi_turn_to_single_turn: bool = args.split_multi_turn_to_single_turn
    drop_assistant_messages: bool = args.drop_assistant_messages
    model_name: str = args.model_name
    max_conversations: int = args.max_conversations
    output_file = args.output_file

    if not output_file:
        raise ValueError("Unspecified output file.")
    output_file = Path(output_file).resolve()
    if output_file.suffix.lower() != ".jsonl":
        raise ValueError(f"Output file must be .jsonl. Got: '{output_file}'")

    # Make the directory if it doesn't exist.
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Tokenizer is not used to generate conversations
    # but generally required by SFT dataset classes (for other methods).
    tokenizer = build_tokenizer(
        ModelParams(model_name=model_name, trust_remote_code=trust_remote_code)
    )

    dataset = _load_sft_dataset(
        dataset_name,
        dataset_path=dataset_path,
        dataset_subset=dataset_subset,
        dataset_split=dataset_split,
        tokenizer=tokenizer,
        processor_name=model_name,
        trust_remote_code=trust_remote_code,
    )

    num_records = len(dataset)
    max_conversations = (
        min(num_records, max_conversations) if max_conversations > 0 else num_records
    )
    logger.info(
        f"Writing {max_conversations} conversations (of {num_records}) "
        f"to '{output_file}'..."
    )

    num_conversations_written = 0
    num_messages_written = 0
    with jsonlines.open(output_file, mode="w") as writer:
        for idx in tqdm(range(max_conversations)):
            for conversation in _process_conversation(
                dataset.conversation(idx),
                split_multi_turn_to_single_turn,
                drop_assistant_messages,
            ):
                num_conversations_written += 1
                num_messages_written += len(conversation.messages)

                json_obj = conversation.to_dict()
                writer.write(json_obj)

    logger.info(
        f"Finished processing {max_conversations} input conversations, and "
        f"wrote {num_conversations_written} output conversations with "
        f"{num_messages_written} messages!"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Saves SFT conversations ")
    parser.add_argument("--name", type=str, required=True, help="Dataset name.")
    parser.add_argument("--path", type=str, required=False, help="Dataset path.")
    parser.add_argument("--subset", type=str, required=False, help="Dataset subset.")
    parser.add_argument("--split", type=str, required=False, help="Dataset split.")
    parser.add_argument(
        "--trust-remote-code",
        type=bool,
        default=True,
        required=False,
        help="Whether to trust remote code.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        required=False,
        help="Tokenizer name.",
    )
    parser.add_argument(
        "--split-multi-turn-to-single-turn",
        action="store_true",
        required=False,
        help=(
            "Whether to split multi-turn conversations "
            "into N single-turn conversations."
        ),
    )
    parser.add_argument(
        "--drop-assistant-messages",
        action="store_true",
        required=False,
        help=("Whether to remove all assistant responses."),
    )

    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        required=True,
        default="conversations.jsonl",
        help="Output JSONL file",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        help="Log level.",
    )
    parser.add_argument(
        "--max-conversations",
        type=int,
        default=-1,
        help=(
            "Maximum number of conversations to save. "
            "Non-positive value means `unlimited` i.e., all dataset records."
        ),
    )
    args = parser.parse_args()

    update_logger_level("oumi", level=args.log_level)

    main(args)
