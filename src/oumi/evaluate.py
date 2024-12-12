import argparse
import json
import os
import time
from copy import deepcopy
from pathlib import Path
from pprint import pformat
from typing import Any

import lm_eval
import torch
from lm_eval.loggers import WandbLogger

from oumi.builders import build_processor, build_tokenizer, is_image_text_llm
from oumi.core.configs import EvaluationConfig, LMHarnessParams, ModelParams
from oumi.core.distributed import is_world_process_zero
from oumi.evaluation.huggingface_leaderboard import (
    BENCHMARK_CONFIGS,
    HUGGINGFACE_LEADERBOARD_V1,
)
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
    """Main entry point for evaluating Oumi.

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
    config.finalize_and_validate()

    # Run evaluation
    evaluate(config)


def evaluate(config: EvaluationConfig) -> None:
    """Evaluates a model using the provided configuration.

    Args:
        config: The desired configuration for evaluation.

    Returns:
        None.
    """
    # Make a deep copy of the config before modifying
    config = deepcopy(config)
    if config.lm_harness_params:
        if any(
            task == HUGGINGFACE_LEADERBOARD_V1
            for task in config.lm_harness_params.tasks
        ):
            # Filter the leaderboard tasks.
            config.lm_harness_params.tasks = [
                task
                for task in config.lm_harness_params.tasks
                if task != HUGGINGFACE_LEADERBOARD_V1
            ]
            # Run the leaderboard tasks separately.
            evaluate_lm_harness_leaderboard(config)
        evaluate_lm_harness(config)
    else:
        raise ValueError("An evaluation framework must be specified.")


def _create_extra_lm_harness_args_for_vlm(model_params: ModelParams) -> dict[str, Any]:
    # For details, see:
    # https://github.com/EleutherAI/lm-evaluation-harness/releases/tag/v0.4.5
    # FIXME OPE-355 To remove `max_images=1` limit
    result = {"max_images": 1, "interleave": True, "convert_img_format": True}

    tokenizer = build_tokenizer(model_params)
    processor = build_processor(
        model_params.model_name,
        tokenizer,
        trust_remote_code=model_params.trust_remote_code,
    )
    image_token = processor.image_token
    if image_token:
        result["image_string"] = image_token
    image_token_id = processor.image_token_id
    if image_token_id:
        result["image_token_id"] = image_token_id
    return result


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
    assert config.lm_harness_params is not None
    # If batch size isn't specified, we set it to "auto", which will let LM Harness
    # automatically select the largest batch size that will fit in memory.
    # https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md
    batch_size = (
        config.generation.batch_size if config.generation.batch_size else "auto"
    )
    start_time = time.time()

    lm_harness_args = config.model.to_lm_harness()

    if is_image_text_llm(config.model):
        # Multimodal support is currently restricted to
        # the ['hf-multimodal', 'vllm-vlm'] model types.
        lm_harness_model = "hf-multimodal"
        apply_chat_template = True
        lm_harness_args.update(_create_extra_lm_harness_args_for_vlm(config.model))
    else:
        lm_harness_model = "hf"
        # False is the default value for `simple_evaluate()`
        # TODO Should it be set to True?
        apply_chat_template = False

    logger.info("Starting evaluation...")
    logger.info(f"\tLM Harness args:\n{pformat(lm_harness_args)}")
    results = lm_eval.simple_evaluate(
        model=lm_harness_model,
        model_args=lm_harness_args,
        tasks=config.lm_harness_params.tasks,  # type: ignore
        num_fewshot=config.lm_harness_params.num_fewshot,
        batch_size=batch_size,
        device=device,
        limit=config.lm_harness_params.num_samples,
        log_samples=False,
        apply_chat_template=apply_chat_template,
    )
    elapsed_time_sec = time.time() - start_time

    # Metrics are only available on the main process, and `None` on others.
    if is_world_process_zero():
        assert results is not None
        for benchmark_name in config.lm_harness_params.tasks:
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
        if config.enable_wandb:
            project_name = os.environ.get("WANDB_PROJECT", "oumi")
            logger.info(f"Logging to Weights and Biases project: '{project_name}'")
            wandb_logger = WandbLogger(
                project=project_name, name=config.run_name, job_type="eval"
            )
            wandb_logger.post_init(results)
            wandb_logger.log_eval_result()


def evaluate_lm_harness_leaderboard(config: EvaluationConfig) -> None:
    """Evaluates a model using LM Evaluation Harness and the HF leaderboard benchmarks.

    Args:
        config: The desired configuration for evaluation.

    Returns:
        None.
    """
    # Identify the relevant benchmark group.
    benchmark_configs = BENCHMARK_CONFIGS[HUGGINGFACE_LEADERBOARD_V1]

    # Evaluate each benchmark in the group.
    for benchmark_config in benchmark_configs:
        mutable_config = deepcopy(config)
        mutable_config.lm_harness_params = LMHarnessParams(
            tasks=[benchmark_config.name],
            num_fewshot=benchmark_config.num_fewshot,
            num_samples=benchmark_config.num_samples,
        )
        evaluate_lm_harness(mutable_config)


def save_evaluation_results(
    output_dir: str,
    benchmark_name: str,
    metric_dict: dict[str, Any],
) -> None:
    """Writes metrics as a dict of dicts: Benchmarks -> metric names -> metric vals."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_filename = SAVE_FILENAME_JSON.format(benchmark_name=benchmark_name)
    output_file = output_path / output_filename
    with output_file.open(mode="w", encoding="utf-8") as f:
        json.dump(metric_dict, f)


if __name__ == "__main__":
    main()
