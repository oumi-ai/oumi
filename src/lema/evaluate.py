import argparse

from lema.core.types import EvaluationConfig
from lema.datasets.mmlu import MmluDataset
from lema.evaluation import compute_multiple_choice_accuracy
from lema.evaluation.infer_prob import infer_prob
from lema.logging import logger
from lema.utils.batching import batch, unbatch


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
    evaluate(config)


def evaluate(config: EvaluationConfig) -> None:
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
    if config.data.datasets[0].dataset_name == "cais/mmlu":
        subject, num_entries = "sociology", 8  # Hardcoded for testing.
        mmlu_dataset = MmluDataset(subject=subject)
        dataset = mmlu_dataset.get_test_split(num_entries=num_entries)
        answer_indices = mmlu_dataset.get_test_labels(num_entries=num_entries)
    else:
        # FIXME: Generalize: Support for multiple datasets.
        raise NotImplementedError("Model evaluation only for MMLU for now.")

    # Batch the dataset to items of length `batch_size`.
    dataset_batched = batch(dataset, config.generation.batch_size)

    # Run inference and then unbatch the model responses.
    answer_probabilities_batched = infer_prob(
        model_params=config.model,
        input=dataset_batched,
        acceptable_tokens=MmluDataset.answer_tokens,
        input_filepath=config.generation.input_filepath,
        output_filepath=config.generation.output_filepath,
    )
    answer_probabilities = unbatch(answer_probabilities_batched)

    # FIXME: Generalize: Support for multiple metrics.
    accuracy = compute_multiple_choice_accuracy(answer_probabilities, answer_indices)
    logger.info(f"MMLU accuracy for {subject} is {accuracy:.3f}")


if __name__ == "__main__":
    main()
