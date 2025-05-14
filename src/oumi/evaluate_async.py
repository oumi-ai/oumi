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

import argparse
import re
import time
from copy import deepcopy
from pathlib import Path

from oumi.core.configs import AsyncEvaluationConfig
from oumi.evaluate import evaluate
from oumi.utils.logging import logger

_PREFIX_CHECKPOINT_DIR = "checkpoint"


def parse_cli():
    """Parses command line arguments and return the configuration filename."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", default=None, help="Path to the configuration file"
    )
    args, arg_list = parser.parse_known_args()
    return args.config, arg_list


def main() -> None:
    """Main entry point for running aynsc Oumi evals.

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
    config.finalize_and_validate()

    # Run evaluation
    evaluate_async(config)


def _get_checkpoints(checkpoint_dir: Path) -> list[Path]:
    """Returns all checkpoints in the target directory."""
    # Modified from HF's transformers.trainer_utils.get_last_checkpoint().
    re_checkpoint = re.compile(r"^" + _PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")
    return [
        path
        for path in checkpoint_dir.iterdir()
        if re_checkpoint.search(path.name) is not None and path.is_dir()
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
    base_output_dir = Path(config.evaluation.output_dir)
    checkpoints_dir = Path(config.checkpoints_dir)
    while retry_count <= config.num_retries:
        # Check for a valid checkpoint.
        unseen_checkpoints = [
            checkpoint
            for checkpoint in _get_checkpoints(checkpoints_dir)
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
            output_eval_dir = base_output_dir / checkpoint.name
            mutable_evaluation_config = deepcopy(config.evaluation)
            # Update the model to point to the checkpoint.
            mutable_evaluation_config.model.model_name = str(checkpoint)
            # Update the eval output location.
            mutable_evaluation_config.output_dir = str(output_eval_dir)
            logger.info(f"Starting evaluation for checkpoint: {checkpoint.name}...")
            evaluate(mutable_evaluation_config)
            logger.info(f"Finished evaluation for checkpoint: {checkpoint.name} !")
        retry_count = 0
        time.sleep(config.polling_interval)
    logger.info(f"Retries exceeded `num_retries`: {config.num_retries}. Exiting...")


if __name__ == "__main__":
    main()
