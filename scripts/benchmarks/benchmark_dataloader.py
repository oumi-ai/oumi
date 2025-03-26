import argparse
import itertools
import time
from pathlib import Path as Pathlib
from pprint import pformat
from typing import Any, Optional

import datasets as hf_datasets
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchMapDataset
from torch.utils.data import IterableDataset as TorchIterableDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from oumi.builders import build_dataset_mixture, build_tokenizer
from oumi.core.configs import DatasetSplit, TrainingConfig
from oumi.core.distributed import (
    cleanup_distributed,
    init_distributed,
    is_distributed,
    is_local_process_zero,
    is_world_process_zero,
)
from oumi.datasets.debug import DebugPretrainingDataset
from oumi.utils.io_utils import save_json
from oumi.utils.logging import logger, update_logger_level

#
# Parameters to benchmark
#

# Parameter matrix for `_benchmark_dataloader_epoch`
BENCHMARK_MATRIX = {
    "batch_size": [8, 32, 64],
    "num_dataloader_workers": [0, 2, 4, 8],
    # "pin_memory": [False, True],
    "model_fwd_bwd_ms": [1.0, 10.0],  # simulate fake forward/bwd pass
}

# Parameter matrix for `ConfigurableTextPretrainingDataset.__init__`
DUMMY_DATASET_MATRIX = {
    "sequence_length": [128, 1024],
    "preprocess_time_ms": [0, 1, 5],  # simulate per item preprocesing time
}

# Disable pre-processing caching for HF datasets
hf_datasets.disable_caching()


#
# Main
#
def main(args):
    """Runs the DataLoader benchmark in distributed mode."""
    if is_distributed():
        logger.info("Benchmarking in distributed mode...")
        init_distributed()
    else:
        logger.info("Running benchmark in single-process mode.")

    #
    # Run benchmarks
    #
    all_results = []

    if args.config:
        logger.info("Benchmarking training dataset...")
        config: TrainingConfig = TrainingConfig.from_yaml(args.config)

        tokenizer = build_tokenizer(config.model)
        init_time, dataset = _load_dataset(
            build_dataset_mixture, config.data, tokenizer, DatasetSplit.TRAIN
        )

        # Anything that's useful for debugging / slicing plots.
        # Will be propagated in the final result.
        metadata = {
            "init_time": init_time,
        }

        # List of dataset param configs to test.
        # Given that we're benchmarking a provided config, we only have one
        dataset_params = {
            "stream": config.data.train.stream,
            "params": config.data.train.pack,
            "mixture_strategy": config.data.train.mixture_strategy,
        }

        matrix = BENCHMARK_MATRIX.copy()
        matrix["dataset_params"] = [dataset_params]

    elif args.dummy:
        matrix = BENCHMARK_MATRIX.copy()
        matrix["dataset_params"] = _generate_config_combinations(
            variable_params=DUMMY_DATASET_MATRIX
        )
        logger.info(f"Found {len(matrix['dataset_params'])} dataset configurations...")

        # Anything that's useful for debugging / slicing plots.
        # Will be propagated in the final result.
        metadata = {}

    # Generate all possible dataloader configurations
    benchmark_configs = _generate_config_combinations(matrix)
    logger.info(f"Starting benchmarking of {len(benchmark_configs)} configurations...")

    # Run all benchmarks
    for benchmark_config in tqdm(
        benchmark_configs, disable=not is_local_process_zero()
    ):
        if args.dummy:
            init_time, dataset = _load_dataset(
                DebugPretrainingDataset, **benchmark_config["dataset_params"]
            )
            metadata["init_time"] = init_time

        run_results = _benchmark_dataloader_epoch(
            dataset,
            use_distributed_sampler=is_distributed(),
            max_steps=args.max_steps,
            **benchmark_config,
        )
        run_results.update(metadata.copy())
        run_results.update(benchmark_config.copy())

        all_results.append(run_results)

        if is_world_process_zero():
            logger.debug(pformat(run_results))

    #
    # Gather results from all processes
    #
    if is_distributed():
        world_size = dist.get_world_size()
        # placeholder object to gather results from each GPU worker
        gathered_results = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_results, all_results)
    else:
        gathered_results = [all_results]

    #
    # Save results
    #
    if is_world_process_zero():
        # combined_results contains results for each worker, indexed by rank
        # each worker results is a list of dicts, one per tested config
        # each test config result dict contains {metric1: value1, ...}
        combined_results = {
            f"rank{rank}": results for rank, results in enumerate(gathered_results)
        }

        output_folder = Pathlib(args.output)
        output_folder.mkdir(exist_ok=True, parents=True)
        save_json(
            data=combined_results,
            filename=output_folder / "benchmark_results.json",
        )
        logger.info(f"Benchmark completed. Saved results to: '{args.output}'")

    if is_distributed():
        cleanup_distributed()


#
# Helper functions
#
def _generate_config_combinations(
    variable_params: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    """Generates a list of configs based on a list of variable parameters."""
    keys, values = zip(*variable_params.items())
    configurations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return [{**conf} for conf in configurations]


def _load_dataset(dataset_fn, *args, **kwargs) -> tuple[float, Any]:
    """Measures the time taken to initialize a dataset using the given dataset function.

    Parameters:
        dataset_fn (callable): The function used to create the dataset.
        *args: Variable length argument list to be passed to the dataset function.
        **kwargs: Arbitrary keyword arguments to be passed to the dataset function.

    Returns:
        float: The time taken to initialize the dataset in seconds.
        Any: The initialized dataset object.

    """
    start_time = time.perf_counter()
    dataset = dataset_fn(*args, **kwargs)
    end_time = time.perf_counter()
    return end_time - start_time, dataset


def _no_op_collate(batch: Any):  # -> Any:
    """No-op identity collate function."""
    return batch


def _benchmark_dataloader_epoch(
    dataset,
    batch_size: int = 1,
    num_dataloader_workers: int = 0,
    pin_memory: bool = False,
    model_fwd_bwd_ms: float = 0.0,
    use_distributed_sampler: bool = False,
    max_steps: Optional[int] = None,
    **kwargs,
) -> dict[str, Any]:
    """Measures the time taken to iterate over a DataLoader for one epoch."""
    if use_distributed_sampler:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    if isinstance(dataset, TorchIterableDataset):
        # shuffle should be unspecified with an iterable dataset
        shuffle = None
    elif isinstance(dataset, TorchMapDataset):
        shuffle = True
    else:
        # if a sampler is provided, should not shuffle
        shuffle = sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_no_op_collate,
        sampler=sampler,
        num_workers=num_dataloader_workers,
        pin_memory=pin_memory,
    )

    steps = 0

    start_time = time.perf_counter()
    for _ in dataloader:
        steps += 1
        time.sleep(model_fwd_bwd_ms / 1000)  # Simulate model forward and backward

        if max_steps and steps >= max_steps:
            logger.debug(f"Stopping DataLoader iteration after {max_steps} steps.")
            break

    logger.debug(f"Total steps: {steps}.")
    end_time = time.perf_counter()

    # Time spent for a full epoch
    total_time_seconds = end_time - start_time
    # Time spent in the dataloader (excluding model forward and backward)
    total_wait_time_seconds = steps * model_fwd_bwd_ms / 1000

    return {
        "total_epoch_time": total_time_seconds,
        "total_dataloading_time": total_time_seconds - total_wait_time_seconds,
        "steps": steps,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark DataLoader performance")
    parser.add_argument(
        "-c", "--config", type=str, help="Path to the Oumi configuration file."
    )
    parser.add_argument(
        "-d",
        "--dummy",
        action="store_true",
        help="Use a dummy dataset instead of an Oumi dataset.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataloader_benchmark_results",
        help="Output directory for benchmark result plots",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        help="Log level.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum number of steps to run the benchmark for.",
    )
    args = parser.parse_args()

    if not args.config and not args.dummy:
        raise ValueError("Either --config or --dummy must be provided")

    update_logger_level("oumi", level=args.log_level)

    main(args)
