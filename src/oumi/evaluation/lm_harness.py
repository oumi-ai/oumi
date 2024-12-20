import os
import time
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Any, Optional

import lm_eval
import torch
from lm_eval.loggers import WandbLogger

from oumi.builders import build_processor, build_tokenizer, is_image_text_llm
from oumi.core.configs import (
    GenerationParams,
    LMHarnessTaskParams,
    ModelParams,
)
from oumi.core.distributed import is_world_process_zero
from oumi.utils.logging import logger
from oumi.utils.serialization_utils import json_serializer
from oumi.utils.version_utils import get_python_package_versions

OUTPUT_FILENAME_RESULTS = "lm_harness_{time}_results.json"
OUTPUT_FILENAME_TASK_CONFIG = "lm_harness_{time}_task_config.json"
OUTPUT_FILENAME_MODEL_PARAMS = "lm_harness_{time}_model_params.json"
OUTPUT_FILENAME_GENERATION_PARAMS = "lm_harness_{time}_generation_params.json"
OUTPUT_FILENAME_HARNESS_PARAMS = "lm_harness_{time}_lm_harness_params.json"
OUTPUT_FILENAME_PKG_VERSIONS = "lm_harness_{time}_package_versions.json"


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


def evaluate_lm_harness(
    model_params: ModelParams,
    lm_harness_params: LMHarnessTaskParams,
    generation_params: GenerationParams,
    output_dir: str,
    enable_wandb: bool,
    run_name: Optional[str] = None,
) -> None:
    """Evaluates a model using the LM Evaluation Harness framework (EleutherAI).

    For detailed documentation, we refer you to the following readme:
    https://github.com/EleutherAI/lm-evaluation-harness

    Args:
        model_params: The parameters of the model to evaluate.
        lm_harness_params: The LM Harness parameters to use for evaluation.
        generation_params: The generation parameters to use for evaluation.
        output_dir: The directory where the evaluation results will be saved.
        enable_wandb: Whether to enable Weights & Biases (wandb) logging.
        run_name: Unique identifier for wandb for the current training run.
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

    if model_params.adapter_model:
        logger.info(f"Loading adapter for eval: {model_params.adapter_model}")
    assert lm_harness_params is not None
    # If batch size isn't specified, we set it to "auto", which will let LM Harness
    # automatically select the largest batch size that will fit in memory.
    # https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md
    batch_size = (
        generation_params.batch_size if generation_params.batch_size else "auto"
    )
    start_time = time.time()

    lm_harness_args = model_params.to_lm_harness()

    if is_image_text_llm(model_params):
        # Multimodal support is currently restricted to
        # the ['hf-multimodal', 'vllm-vlm'] model types.
        lm_harness_model = "hf-multimodal"
        apply_chat_template = True
        lm_harness_args.update(_create_extra_lm_harness_args_for_vlm(model_params))
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
        tasks=lm_harness_params.tasks,  # type: ignore
        num_fewshot=lm_harness_params.num_fewshot,
        batch_size=batch_size,  # type: ignore
        device=device,
        limit=lm_harness_params.num_samples,
        log_samples=False,
        apply_chat_template=apply_chat_template,
    )
    elapsed_time_sec = time.time() - start_time

    # Metrics are only available on the main process, and `None` on others.
    if is_world_process_zero():
        assert lm_eval_output is not None
        for benchmark_name in lm_harness_params.tasks:
            metric_dict = lm_eval_output["results"][benchmark_name]  # type: ignore
            logger.info(f"{benchmark_name}'s metric dict is {pformat(metric_dict)}")

        if enable_wandb:
            project_name = os.environ.get("WANDB_PROJECT", "oumi")
            logger.info(f"Logging to Weights and Biases project: '{project_name}'")
            wandb_logger = WandbLogger(
                project=project_name, name=run_name, job_type="eval"
            )
            wandb_logger.post_init(lm_eval_output)
            wandb_logger.log_eval_result()

        if output_dir:
            save_lm_harness_output(
                output_dir=output_dir,
                lm_harness_output=lm_eval_output,
                model_params=model_params,
                lm_harness_params=lm_harness_params,
                generation_params=generation_params,
                elapsed_time_sec=elapsed_time_sec,
            )


def save_lm_harness_output(
    output_dir: str,
    lm_harness_output: dict[str, Any],
    model_params: ModelParams,
    lm_harness_params: LMHarnessTaskParams,
    generation_params: GenerationParams,
    elapsed_time_sec: float,
) -> None:
    """Writes configuration settings and LM Harness outputs to files."""
    time_now = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Make sure the output folder exists.
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Save results ---
    # This file includes: all evaluation metrics, completion date and time, duration.
    output_file_results = OUTPUT_FILENAME_RESULTS.format(time=time_now)
    results = {
        key: lm_harness_output.pop(key)
        for key in ["results", "groups"]
        if key in lm_harness_output
    }
    results["duration_sec"] = elapsed_time_sec
    results["completion_time"] = time_now
    with open(output_path / output_file_results, "w") as file_out:
        file_out.write(json_serializer(results))

    #  --- Save LM Harness task configuration(s) ---
    # This file includes: number of samples, number of few-shots, task version(s),
    # prompt(s) text, model/git hashes, seeds, and special tokens (pad, eos, bos, eot).
    output_file_task_config = OUTPUT_FILENAME_TASK_CONFIG.format(time=time_now)
    with open(output_path / output_file_task_config, "w") as file_out:
        file_out.write(json_serializer(lm_harness_output))

    #  --- Save evaluation configuration ---
    output_file_model_params = OUTPUT_FILENAME_MODEL_PARAMS.format(time=time_now)
    with open(output_path / output_file_model_params, "w") as file_out:
        file_out.write(json_serializer(model_params))

    output_file_gen_params = OUTPUT_FILENAME_GENERATION_PARAMS.format(time=time_now)
    with open(output_path / output_file_gen_params, "w") as file_out:
        file_out.write(json_serializer(generation_params))

    output_file_harness_params = OUTPUT_FILENAME_HARNESS_PARAMS.format(time=time_now)
    with open(output_path / output_file_harness_params, "w") as file_out:
        file_out.write(json_serializer(lm_harness_params))

    # --- Save python environment (package versions) ---
    output_file_pkg_versions = OUTPUT_FILENAME_PKG_VERSIONS.format(time=time_now)
    package_versions = get_python_package_versions()
    with open(output_path / output_file_pkg_versions, "w") as file_out:
        file_out.write(json_serializer(package_versions))
