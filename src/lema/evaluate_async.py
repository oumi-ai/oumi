import argparse
import os
import re
import time
from copy import deepcopy
from typing import List

from lema import evaluate
from lema.core.types import AsyncEvaluationConfig
from lema.logging import logger

_PREFIX_CHECKPOINT_DIR = "checkpoint"


def parse_cli():
    """Parse command line arguments and return the configuration filename."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", default=None, help="Path to the configuration file"
    )
    args, arg_list = parser.parse_known_args()
    return args.config, arg_list


def main() -> None:
    """Main entry point for running aynsc LeMa evals.

    Evaluation arguments are fetched from the following sources, ordered by
    decreasing priority:
    1. [Optional] Arguments provided as CLI arguments, in dotfile format
    2. [Optional] Arguments provided in a yaml config file
    3. Default arguments values defined in the data class
    """
    # Load configuration
    config_path, arg_list = parse_cli()

    config: AsyncEvaluationConfig = AsyncEvaluationConfig.from_yaml_and_arg_list(
        config_path, arg_list, logger=logger
    )

    # Run evaluation
    evaluate_async(config)


def _get_checkpoints(checkpoint_dir: str) -> List[str]:
    """Returns all checkpoints in the target directory."""
    # Modified from HF's transformers.trainer_utils.get_last_checkpoint().
    re_checkpoint = re.compile(r"^" + _PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")
    directory_list = os.listdir(checkpoint_dir)
    return [
        os.path.join(checkpoint_dir, path)
        for path in directory_list
        if re_checkpoint.search(path) is not None
        and os.path.isdir(os.path.join(checkpoint_dir, path))
    ]


def evaluate_async(config: AsyncEvaluationConfig) -> None:
    """Runs an async evaluation for a model using the provided configuration.

    Overview:
        This is a utility method for running evaluations iteratively over a series
        of checkpoints. This method can be run in parallel with a training job to
        compute metrics per checkpoint without wasting valuable time in the main
        training loop.

    Args:
        config: The desired configuration for evaluation.

    Returns:
        None.
    """
    retry_count = 0
    seen_checkpoints = set()
    base_output_dir = config.evaluation.output_dir
    while retry_count <= config.num_retries:
        # Check for a valid checkpoint.
        unseen_checkpoints = [
            checkpoint
            for checkpoint in _get_checkpoints(config.checkpoints_dir)
            if checkpoint not in seen_checkpoints
        ]
        if len(unseen_checkpoints) == 0:
            retry_count += 1
            time.sleep(config.polling_interval)
            continue
        # Evaluate all unseen checkpoints.
        while len(unseen_checkpoints) > 0:
            checkpoint = unseen_checkpoints.pop()
            seen_checkpoints.add(checkpoint)
            output_eval_dir = os.path.join(
                base_output_dir, os.path.basename(checkpoint)
            )
            mutable_config = deepcopy(config)
            # Update the model to point to the checkpoint.
            mutable_config.evaluation.model.model_name = checkpoint
            # Update the eval output location.
            mutable_config.evaluation.output_dir = output_eval_dir
            logger.info(
                "Starting evaluation for checkpoint: "
                f"{os.path.basename(checkpoint)}..."
            )
            evaluate(mutable_config.evaluation)
            logger.info(
                "Finished evaluation for checkpoint: "
                f"{os.path.basename(checkpoint)} !"
            )
        retry_count = 0
        time.sleep(config.polling_interval)
    logger.info(f"Retries exceeded `num_retries`: {config.num_retries}. Exiting...")


if __name__ == "__main__":
    main()
