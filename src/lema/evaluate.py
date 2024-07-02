import argparse
import json
import os
from typing import Any, Dict

import lm_eval
import torch

from lema.core.types import EvaluationConfig
from lema.core.types.configs import EvaluationFramework
from lema.datasets.mmlu import MmluDataset
from lema.evaluation import compute_multiple_choice_accuracy
from lema.evaluation.infer_prob import infer_prob
from lema.logging import logger
from lema.utils.batching import batch, unbatch

SAVE_FILENAME_JSON = "eval.json"


def parse_cli():
    """Parse command line arguments and return the configuration filename."""
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

    # Run evaluation
    if config.evaluation_framework == EvaluationFramework.LEMA:
        evaluate_lema(config)
    elif config.evaluation_framework == EvaluationFramework.LM_HARNESS:
        evaluate_lm_harness(config)
    else:
        raise ValueError(
            f"Unsupported evaluation framework: {config.evaluation_framework}"
        )


def evaluate_lema(config: EvaluationConfig) -> None:
    """Evaluate a model using the provided configuration.

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
    if config.data.validation.datasets[0].dataset_name == "cais/mmlu":
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
            metric_dict={"cais/mmlu": {"accuracy": accuracy}},
        )
    logger.info(f"MMLU accuracy is {accuracy:.3f}")


def evaluate_lm_harness(config: EvaluationConfig) -> None:
    """Evaluate a model using the LM Evaluation Harness framework (EleutherAI).

    For detailed documentation, we refer you to the following readme:
       https://github.com/EleutherAI/lm-evaluation-harness

    Args:
        config: The desired configuration for evaluation.

    Returns:
        None.
    """
    benchmarks = [dataset.dataset_name for dataset in config.data.validation.datasets]
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        logger.warning("No GPU available.")

    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=(
            f"pretrained={config.model.model_name},"
            f"trust_remote_code={config.model.trust_remote_code}"
        ),
        tasks=benchmarks,  # type: ignore
        num_fewshot=config.num_shots,
        batch_size=config.generation.batch_size,
        device=device,
        limit=config.num_samples if config.num_samples else None,
    )
    if config.output_dir:
        metric_dict = results["results"]  # type: ignore
        save_evaluation_results(
            output_dir=config.output_dir,
            metric_dict=metric_dict,
        )
    for benchmark in benchmarks:
        logger.info(f"{benchmark}'s metric dictionary is {metric_dict[benchmark]}")


def save_evaluation_results(
    output_dir: str,
    metric_dict: Dict[str, Any],
) -> None:
    """Write metrics as a dict of dicts: Benchmarks -> metric names -> metric values."""
    os.makedirs(output_dir, exist_ok=True)
    output_eval_path = os.path.join(output_dir, SAVE_FILENAME_JSON)
    with open(output_eval_path, mode="w", encoding="utf-8") as f:
        json.dump(metric_dict, f)


if __name__ == "__main__":
    main()
