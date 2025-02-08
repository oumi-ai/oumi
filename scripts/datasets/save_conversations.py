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


def main(args):
    """The script's entry point."""
    dataset_name: str = args.name
    dataset_path: Optional[str] = args.path
    dataset_subset: Optional[str] = args.subset
    dataset_split: Optional[str] = args.split
    trust_remote_code: bool = args.trust_remote_code
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

    with jsonlines.open(output_file, mode="w") as writer:
        for idx in tqdm(range(max_conversations)):
            convo = dataset.conversation(idx)
            json_obj = convo.to_dict()
            writer.write(json_obj)

    logger.info("Finished writing!")


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
