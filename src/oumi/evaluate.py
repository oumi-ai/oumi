import argparse
import json
import os
import time
from copy import deepcopy
from pprint import pformat
from typing import Any, Dict

import lm_eval
import torch

from oumi.core.configs import EvaluationConfig
from oumi.core.configs.evaluation_config import EvaluationFramework
from oumi.core.distributed import is_world_process_zero
from oumi.datasets.mmlu import MmluDataset
from oumi.evaluation import compute_multiple_choice_accuracy
from oumi.evaluation.huggingface_leaderboard import (
    BENCHMARK_CONFIGS,
    HUGGINGFACE_LEADERBOARD_V1,
)
from oumi.evaluation.infer_prob import infer_prob
from oumi.utils.batching import batch, unbatch
from oumi.utils.logging import logger

SAVE_FILENAME_JSON = "eval.{benchmark_name}.json"


def parse_cli():
    """Parses command line arguments and return the configuration filename."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", default=None, help="Path to the configuration file"
    )
    args, arg_list = parser.parse_known_args()
    return args.config, arg_list


def main() -> None:
    """Main entry point for evaluating LeMa.

    Evaluation arguments are fetched from the following sources, ordered by
    decreasing priority:
    1. [Optional] Arguments provided as CLI arguments, in dotfile format
    2. [Optional] Arguments provided in a yaml config file
    3. Default arguments values defined in the data class
    """
    # Load configuration
    config_path, arg_list = parse_cli()

    config: EvaluationConfig = EvaluationConfig.from_yaml_and_arg_list(
        config_path, arg_list, logger=logger
    )
    config.validate()

    # Run evaluation
    evaluate(config)


def evaluate(config: EvaluationConfig) -> None:
    """Evaluates a model using the provided configuration.

    Args:
        config: The desired configuration for evaluation.

    Returns:
        None.
    """
    if config.evaluation_framework == EvaluationFramework.LEMA:
        evaluate_lema(config)
    elif config.evaluation_framework == EvaluationFramework.LM_HARNESS:
        if (
            len(config.data.datasets) == 1
            and config.data.datasets[0].dataset_name == HUGGINGFACE_LEADERBOARD_V1
        ):
            evaluate_lm_harness_leaderboard(config)
        else:
            evaluate_lm_harness(config)
    else:
        raise ValueError(
            f"Unsupported evaluation framework: {config.evaluation_framework}"
        )


def evaluate_lema(config: EvaluationConfig) -> None:
    """Evaluates a model using the provided configuration.

    Overview:
        This is a hardcoded function, intending to provide a starting point for our
        evaluations. It only works for the MMLU dataset and evaluates a small
        hardcoded portion of its prompts (for testing purposes).
        We need to extend this function to multiple datasets and metrics.

    Args:
        config: The desired configuration for evaluation.

    Returns:
        None for now, we will return a relevant class in the future.
    """
    # Load the dataset from HuggingFace or a local repository.
    if config.data.datasets[0].dataset_name == "cais/mmlu":
        mmlu_dataset = MmluDataset(subject="all", num_shots=config.num_shots)
        dataset = mmlu_dataset.get_test_split(num_entries=config.num_samples)
        answer_indices = mmlu_dataset.get_test_labels(num_entries=config.num_samples)
    else:
        # FIXME: Generalize: Support for multiple datasets.
        raise NotImplementedError("Model evaluation only for MMLU for now.")

    # Batch the dataset to items of length `batch_size`. If multiple GPUs are available,
    # multiply the `batch_size` by the number of GPUs, to leverage all available GPUs,
    # since Data Parallel (DP) will automatically split the batch.
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        enable_dp = True
        gpu_count = torch.cuda.device_count()
        batch_size = config.generation.batch_size * gpu_count
        logger.info(
            f"Evaluate: The `batch_size` increased from {config.generation.batch_size} "
            f"to {batch_size}, to leverage the {gpu_count} GPUs available."
        )
    else:
        enable_dp = False
        batch_size = config.generation.batch_size
    dataset_batched = batch(dataset, batch_size)

    # Run inference and then unbatch the model responses.
    answer_probabilities_batched = infer_prob(
        model_params=config.model,
        input=dataset_batched,
        acceptable_tokens=MmluDataset.answer_tokens,
        input_filepath=config.generation.input_filepath,
        output_filepath=config.generation.output_filepath,
        enable_dp=enable_dp,
    )
    answer_probabilities = unbatch(answer_probabilities_batched)

    # FIXME: Generalize: Support for multiple metrics.
    accuracy = compute_multiple_choice_accuracy(answer_probabilities, answer_indices)
    if config.output_dir:
        save_evaluation_results(
            output_dir=config.output_dir,
            benchmark_name="mmlu",
            metric_dict={"accuracy": accuracy},
        )
    logger.info(f"MMLU accuracy is {accuracy:.3f}")


def evaluate_lm_harness(config: EvaluationConfig) -> None:
    """Evaluates a model using the LM Evaluation Harness framework (EleutherAI).

    For detailed documentation, we refer you to the following readme:
       https://github.com/EleutherAI/lm-evaluation-harness

    Args:
        config: The desired configuration for evaluation.

    Returns:
        None.
    """
    if torch.cuda.is_available():
        # CUDA device may be overwritten if `accelerate launch`,
        # or `parallelize=True` are used.
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        logger.warning("No GPU available.")

    if config.model.adapter_model:
        logger.info(f"Loading adapter for eval: {config.model.adapter_model}")

    benchmark_names = [dataset.dataset_name for dataset in config.data.datasets]
    batch_size = config.generation.batch_size if config.generation.batch_size else None

    start_time = time.time()
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=config.model.to_lm_harness(),
        tasks=benchmark_names,  # type: ignore
        num_fewshot=config.num_shots,
        batch_size=batch_size,
        device=device,
        limit=config.num_samples,
        log_samples=False,
    )
    elapsed_time_sec = time.time() - start_time

    # Metrics are only available on the main process, and `None` on others.
    if is_world_process_zero():
        assert results is not None
        for benchmark_name in benchmark_names:
            metric_dict = results["results"][benchmark_name]  # type: ignore
            metric_dict["elapsed_time_sec"] = elapsed_time_sec
            if config.output_dir:
                save_evaluation_results(
                    output_dir=config.output_dir,
                    benchmark_name=benchmark_name,
                    metric_dict=metric_dict,
                )
            logger.info(
                f"{benchmark_name}'s metric dictionary is {pformat(metric_dict)}"
            )


def evaluate_lm_harness_leaderboard(config: EvaluationConfig) -> None:
    """Evaluates a model using LM Evaluation Harness and the HF leaderboard benchmarks.

    Args:
        config: The desired configuration for evaluation.

    Returns:
        None.
    """
    # Identify the relevant benchmark group.
    assert len(config.data.datasets) == 1
    if config.data.datasets[0].dataset_name == HUGGINGFACE_LEADERBOARD_V1:
        benchmark_configs = BENCHMARK_CONFIGS[HUGGINGFACE_LEADERBOARD_V1]
    else:
        raise NotImplementedError("Only HuggingFace Leaderboard V1 supported for now.")

    # Evaluate each benchmark in the group.
    for benchmark_config in benchmark_configs:
        mutable_config = deepcopy(config)
        mutable_config.data.datasets[0].dataset_name = benchmark_config.name
        mutable_config.num_shots = benchmark_config.num_shots
        mutable_config.num_samples = benchmark_config.num_samples
        evaluate_lm_harness(mutable_config)


def save_evaluation_results(
    output_dir: str,
    benchmark_name: str,
    metric_dict: Dict[str, Any],
) -> None:
    """Writes metrics as a dict of dicts: Benchmarks -> metric names -> metric vals."""
    os.makedirs(output_dir, exist_ok=True)
    output_filename = SAVE_FILENAME_JSON.format(benchmark_name=benchmark_name)
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, mode="w", encoding="utf-8") as f:
        json.dump(metric_dict, f)


if __name__ == "__main__":
    main()
