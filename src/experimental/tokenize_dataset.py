import argparse
import functools
import os
import pathlib
import time
from typing import Any, Dict, Iterator, List, NamedTuple, Tuple, Union

from datasets import Dataset, disable_caching
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from lema.builders import (
    build_tokenizer,
)
from lema.core.types import TrainingConfig
from lema.logging import logger

_TOKEN_IDS_COLUMN_NAME = "input_ids"  # The common convention.


def _list_input_files(
    input_paths: List[str],
    input_format: str,
) -> Iterator[pathlib.Path]:
    for path_str in input_paths:
        path = pathlib.Path(path_str)
        if not path.exists():
            logger.warning(f"{path} not found and skipped")
            continue
        yield from path.glob(f"*.{input_format}") if path.is_dir() else [path]


def _tokenize_examples(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    target_col: str,
    examples: Dict[str, Any],
) -> Dict[str, Any]:
    batch = tokenizer(examples[target_col])
    token_ids: List[List[int]] = batch.input_ids
    result = examples.copy()
    result[_TOKEN_IDS_COLUMN_NAME] = token_ids
    return result


def _tokenize_file(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    target_col: str,
    input_file: pathlib.Path,
    input_format: str,
    output_file: pathlib.Path,
    num_proc: int,
) -> None:
    logger.info(f"Loading {input_file}.")
    if input_format == "jsonl":
        dataset = Dataset.from_json(str(input_file), keep_in_memory=True)
    else:
        assert input_format == "parquet"
        dataset = Dataset.from_parquet(str(input_file), keep_in_memory=True)
    logger.info("Tokenizing the dataset.")
    dataset = dataset.map(
        functools.partial(_tokenize_examples, tokenizer, target_col),
        batched=True,
        batch_size=128,
        keep_in_memory=True,
        num_proc=num_proc,
    )
    logger.info("Finished tokenizing the dataset.")

    logger.info(f"Writing the tokenized data to {output_file}.")
    dataset.to_parquet(output_file)
    logger.info(f"Finished writing the tokenized to {output_file}.")


ParsedArgs = NamedTuple(
    "ParsedArgs",
    [
        ("config_path", str),
        ("verbose", bool),
        ("input_paths", List[str]),
        ("input_format", str),
        ("target_col", str),
        ("output_dir", str),
        ("overwrite", bool),
        ("num_proc", int),
    ],
)


def parse_cli() -> Tuple[ParsedArgs, List[str]]:
    """Parses command line arguments and returns the configuration filename."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", default=None, help="Path to the configuration file"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        nargs="+",
        help="Path(s) to the input data directory or file.",
    )
    parser.add_argument(
        "--input_format",
        type=str,
        default="parquet",
        choices=["jsonl", "parquet"],
        help="Input format.",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="",
        help="Target text column to tokenize.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite the output directory.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=-1,
        help=(
            "Number of processes for parallel execution. "
            "If -1, then use all available CPU cores."
        ),
    )

    args, unknown = parser.parse_known_args()
    return (
        ParsedArgs(
            config_path=args.config,
            verbose=args.verbose,
            input_paths=args.input_path,
            input_format=args.input_format,
            target_col=args.target_col,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            num_proc=args.num_proc,
        ),
        unknown,
    )


def main() -> None:
    """Main function."""
    disable_caching()
    parsed_args, arg_list = parse_cli()

    config: TrainingConfig = TrainingConfig.from_yaml_and_arg_list(
        parsed_args.config_path, arg_list, logger=logger
    )

    # Find first non-empty value as target column name.
    target_col = next(
        s
        for s in [
            parsed_args.target_col,
            config.data.train.target_col,
            config.data.validation.target_col,
            config.data.test.target_col,
            "text",
        ]
        if s
    )

    output_dir: pathlib.Path = pathlib.Path(parsed_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    logger.info("Initializing the tokenizer...")

    tokenizer = build_tokenizer(config.model)

    input_files: list[pathlib.Path] = sorted(
        _list_input_files(parsed_args.input_paths, parsed_args.input_format)
    )
    if not input_files:
        return

    logger.info("Loading the dataset...")
    for input_file in tqdm(input_files):
        output_file: pathlib.Path = output_dir / f"{input_file.stem}.parquet"
        if output_file.exists() and not parsed_args.overwrite:
            logger.error(f"{output_file} already exists. Specify --overwrite.")
            continue
        _tokenize_file(
            tokenizer,
            target_col,
            input_file,
            parsed_args.input_format,
            output_file,
            num_proc=(
                (os.cpu_count() or 1)
                if parsed_args.num_proc == -1
                else parsed_args.num_proc
            ),
        )

    end_time = time.time()
    logger.info(
        f"Finished tokenizing the dataset. Elapsed time: {end_time - start_time} sec!"
    )


if __name__ == "__main__":
    main()
