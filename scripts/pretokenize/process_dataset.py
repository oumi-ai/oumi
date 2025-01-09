import argparse
import functools
import os
import pathlib
import time
from collections.abc import Iterator
from typing import (
    Any,
    NamedTuple,
    Optional,
    TypeVar,
    Union,
    cast,
)

import datasets
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from oumi.builders import build_tokenizer
from oumi.core.configs import TrainingConfig
from oumi.utils.logging import logger

_TOKEN_IDS_COLUMN_NAME = "input_ids"  # The common convention.

DatasetType = TypeVar("DatasetType", datasets.Dataset, datasets.IterableDataset)


def _list_input_files(
    input_paths: list[str],
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
    examples: dict[str, Any],
) -> dict[str, Any]:
    batch = tokenizer(examples[target_col])
    token_ids: list[list[int]] = batch.input_ids
    result = examples.copy()
    result[_TOKEN_IDS_COLUMN_NAME] = token_ids
    return result


def _tokenize_dataset_impl(
    dataset: DatasetType,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    target_col: str,
    num_proc: int,
    keep_in_memory: bool,
) -> DatasetType:
    logger.info("Tokenizing the dataset...")
    dataset = dataset.map(
        functools.partial(_tokenize_examples, tokenizer, target_col),
        batched=True,
        batch_size=128,
        keep_in_memory=keep_in_memory,
        num_proc=num_proc,
    )
    logger.info("Finished tokenizing the dataset.")
    return dataset


def _process_file(
    tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]],
    target_col: str,
    input_file: pathlib.Path,
    input_format: str,
    output_parquet_file: pathlib.Path,
    num_proc: int,
) -> None:
    logger.info(f"Loading {input_file}.")
    if input_format == "jsonl":
        dataset = datasets.Dataset.from_json(str(input_file), keep_in_memory=True)
    elif input_format == "arrow":
        dataset = datasets.Dataset.from_file(str(input_file), in_memory=True)
    else:
        assert input_format == "parquet"
        dataset = datasets.Dataset.from_parquet(str(input_file), keep_in_memory=True)

    if tokenizer is not None:
        dataset = _tokenize_dataset_impl(
            cast(datasets.Dataset, dataset),
            tokenizer,
            target_col,
            num_proc=num_proc,
            keep_in_memory=True,
        )

    logger.info(f"Writing the processsed data to {output_parquet_file}.")
    dataset.to_parquet(output_parquet_file)
    logger.info(f"Finished writing to {output_parquet_file}.")


def _process_dataset(
    *,
    tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]],
    target_col: str,
    input_dataset: str,
    dataset_subset: Optional[str],
    dataset_split: Optional[str],
    trust_remote_code: bool,
    output_dataset_path: pathlib.Path,
    num_shards: int,
    max_shard_size: str,
    num_proc: int,
) -> None:
    if (
        input_dataset.startswith("/")
        or input_dataset.startswith(".")
        or input_dataset.startswith("~")
    ):
        input_dataset_path = pathlib.Path(input_dataset)
        logger.info(f"Loading {input_dataset_path} from disk...")
        dataset = datasets.Dataset.load_from_disk(str(input_dataset_path))
    else:
        logger.info(f"Loading {input_dataset} from HuggingFace...")
        splits_or_dataset = datasets.load_dataset(
            path=input_dataset,
            name=(dataset_subset or None),
            split=(dataset_split or None),
            trust_remote_code=trust_remote_code,
        )

        if isinstance(
            splits_or_dataset, (datasets.IterableDataset, datasets.IterableDatasetDict)
        ):
            raise ValueError("IterableDataset is not supported with this class.")

        if isinstance(splits_or_dataset, datasets.Dataset):
            dataset = splits_or_dataset
        elif dataset_split:
            dataset = splits_or_dataset[dataset_split]
        elif len(splits_or_dataset) == 1:
            dataset = splits_or_dataset.values().__iter__().__next__()
        else:
            raise ValueError(
                "Multiple splits found in the dataset. Please specify a single split. "
                f"Available splits: {list(splits_or_dataset.keys())}"
            )

    logger.info(
        "\n".join(
            [
                "Dataset Loaded!",
                f"Split: {dataset.split}",
                f"Version: {dataset.version}",
                f"Dataset size: {dataset.dataset_size}",
                f"Download size: {dataset.download_size}",
                f"Size: {dataset.size_in_bytes} bytes",
                f"Column names: {dataset.column_names}",
            ]
        )
    )

    if tokenizer is not None:
        dataset = _tokenize_dataset_impl(
            cast(datasets.Dataset, dataset),
            tokenizer,
            target_col,
            num_proc=num_proc,
            keep_in_memory=False,
        )

    logger.info(f"Writing the output dataset to {output_dataset_path} ...")
    dataset.save_to_disk(
        str(output_dataset_path),
        num_shards=(num_shards if not max_shard_size and num_shards > 0 else None),
        max_shard_size=(max_shard_size or None),
        num_proc=num_proc,
    )
    logger.info(f"Finished writing to {output_dataset_path} !")


class ParsedArgs(NamedTuple):
    config_path: str
    verbose: bool
    input_dataset: str
    dataset_subset: str
    dataset_split: str
    trust_remote_code: bool
    input_paths: list[str]
    input_format: str
    target_col: str
    output_dir: str
    num_shards: int
    max_shard_size: str
    overwrite: bool
    num_proc: int
    skip_tokenize: bool


def parse_cli() -> tuple[ParsedArgs, list[str]]:
    """Parses command line arguments and returns the configuration filename."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Path to the configuration file",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="",
        help="Target text column to tokenize.",
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
    parser.add_argument(
        "--skip_tokenize",
        action="store_true",
        help=(
            "Whether to skip tokenization. "
            "Can be useful if you just want to copy a dataset, "
            "or convert it from one format to another."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="",
        help="Max shard size e.g., 256MB",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=512,
        help="Number of shards.",
    )

    # Parameters to work with HF datasets
    parser.add_argument(
        "--input_dataset",
        type=str,
        help="Directory path to the input Huggingface dataset, or a dataset name.",
    )
    parser.add_argument(
        "--dataset_subset",
        type=str,
        help="Subset of an input dataset",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        help="Split of an input dataset e.g., 'train'",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether to trust remote code.",
    )

    # Parameters to work with individual files
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
        choices=["jsonl", "parquet", "arrow"],
        help="Input format.",
    )

    args, unknown = parser.parse_known_args()
    return (
        ParsedArgs(
            config_path=args.config,
            verbose=args.verbose,
            input_dataset=args.input_dataset,
            dataset_subset=args.dataset_subset,
            dataset_split=args.dataset_split,
            trust_remote_code=args.trust_remote_code,
            input_paths=args.input_path,
            input_format=args.input_format,
            target_col=args.target_col,
            output_dir=args.output_dir,
            num_shards=args.num_shards,
            max_shard_size=args.max_shard_size,
            overwrite=args.overwrite,
            num_proc=args.num_proc,
            skip_tokenize=args.skip_tokenize,
        ),
        unknown,
    )


def main() -> None:
    """Main function."""
    parsed_args, arg_list = parse_cli()

    logger.info(f"Parsed arguments: {parsed_args}")
    logger.info(f"Unknown arguments: {arg_list}")

    config: TrainingConfig = None
    target_col: str = ""
    tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None
    if not parsed_args.skip_tokenize:
        config = TrainingConfig.from_yaml_and_arg_list(
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
        logger.info("Initializing the tokenizer...")
        tokenizer = build_tokenizer(config.model)

    output_dir: pathlib.Path = pathlib.Path(parsed_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    num_proc = (
        (os.cpu_count() or 1) if parsed_args.num_proc == -1 else parsed_args.num_proc
    )

    if parsed_args.input_dataset:
        logger.info(f"Processing the dataset {parsed_args.input_dataset}...")
        _process_dataset(
            tokenizer=tokenizer,
            target_col=target_col,
            input_dataset=parsed_args.input_dataset,
            dataset_subset=parsed_args.dataset_subset,
            dataset_split=parsed_args.dataset_split,
            trust_remote_code=parsed_args.trust_remote_code,
            output_dataset_path=output_dir,
            num_shards=parsed_args.num_shards,
            max_shard_size=parsed_args.max_shard_size,
            num_proc=num_proc,
        )
    else:
        datasets.disable_caching()
        input_files: list[pathlib.Path] = list(
            sorted(_list_input_files(parsed_args.input_paths, parsed_args.input_format))
        )
        if not input_files:
            logger.warning("No files found!")
            return

        logger.info(f"Loading the dataset from {len(input_files)} files...")
        for input_file in tqdm(input_files):
            output_file: pathlib.Path = output_dir / f"{input_file.stem}.parquet"

            if output_file.exists() and not parsed_args.overwrite:
                logger.error(f"{output_file} already exists. Specify --overwrite.")
                continue
            _process_file(
                tokenizer,
                target_col,
                input_file,
                parsed_args.input_format,
                output_file,
                num_proc=num_proc,
            )

    end_time = time.time()
    logger.info(
        f"Finished processing the dataset. Elapsed time: {end_time - start_time} sec!"
    )


if __name__ == "__main__":
    main()
