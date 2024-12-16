import json
import os
import time
from copy import deepcopy
from datetime import datetime
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
from oumi.utils.serialization_utils import TorchJsonEncoder
from oumi.utils.version_utils import get_python_package_versions

OUTPUT_FILENAME_RESULTS = "lm_harness_{time}_results.json"
OUTPUT_FILENAME_TASK_CONFIG = "lm_harness_{time}_task_config.json"
OUTPUT_FILENAME_EVAL_CONFIG = "lm_harness_{time}_evaluation_config.yaml"
OUTPUT_FILENAME_PKG_VERSIONS = "lm_harness_{time}_package_versions.json"
JSON_FILE_INDENT = 2


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
    lm_eval_output = lm_eval.simple_evaluate(
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
        assert lm_eval_output is not None
        for benchmark_name in config.lm_harness_params.tasks:
            metric_dict = lm_eval_output["results"][benchmark_name]  # type: ignore
            logger.info(f"{benchmark_name}'s metric dict is {pformat(metric_dict)}")

        if config.enable_wandb:
            project_name = os.environ.get("WANDB_PROJECT", "oumi")
            logger.info(f"Logging to Weights and Biases project: '{project_name}'")
            wandb_logger = WandbLogger(
                project=project_name, name=config.run_name, job_type="eval"
            )
            wandb_logger.post_init(lm_eval_output)
            wandb_logger.log_eval_result()

        if config.output_dir:
            save_lm_harness_output(
                output_dir=config.output_dir,
                lm_harness_output=lm_eval_output,
                evaluation_config=config,
                elapsed_time_sec=elapsed_time_sec,
            )


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


def save_lm_harness_output(
    output_dir: str,
    lm_harness_output: dict[str, Any],
    evaluation_config: EvaluationConfig,
    elapsed_time_sec: float,
) -> None:
    """Writes configuration settings and LM Harness outputs to files."""
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Make sure the output folder exists.
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Save results ---
    # This file includes: all evaluation metrics, completion date and time, duration.
    output_file_results = OUTPUT_FILENAME_RESULTS.format(time=current_time)
    results = {
        key: lm_harness_output.pop(key)
        for key in ["results", "groups"]
        if key in lm_harness_output
    }
    results["duration_sec"] = elapsed_time_sec
    results["completion_time"] = current_time
    results_json = json.dumps(results, indent=JSON_FILE_INDENT)
    with open(output_path / output_file_results, "w") as file_out:
        file_out.write(results_json)

    #  --- Save LM Harness task configuration(s) ---
    # This file includes: number of samples, number of few-shots, task version(s),
    # prompt(s) text, model/git hashes, seeds, and special tokens (pad, eos, bos, eot).
    output_file_task_config = OUTPUT_FILENAME_TASK_CONFIG.format(time=current_time)
    task_config_json = json.dumps(
        lm_harness_output, cls=TorchJsonEncoder, indent=JSON_FILE_INDENT
    )
    with open(output_path / output_file_task_config, "w") as file_out:
        file_out.write(task_config_json)

    #  --- Save evaluation configuration (oumi.core.configs.EvaluationConfig) ---
    output_file_eval_config = OUTPUT_FILENAME_EVAL_CONFIG.format(time=current_time)
    evaluation_config.to_yaml(output_path / output_file_eval_config)

    # --- Save python environment (package versions) ---
    output_file_pkg_versions = OUTPUT_FILENAME_PKG_VERSIONS.format(time=current_time)
    package_versions = get_python_package_versions()
    package_versions_json = json.dumps(package_versions, indent=JSON_FILE_INDENT)
    with open(output_path / output_file_pkg_versions, "w") as file_out:
        file_out.write(package_versions_json)
