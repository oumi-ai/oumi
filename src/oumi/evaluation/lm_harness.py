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

import copy
import os
import time
from datetime import datetime
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
from oumi.evaluation.save_utils import save_evaluation_output
from oumi.utils.logging import logger


def _create_extra_lm_harness_model_params_for_vlm(
    model_params: ModelParams,
) -> dict[str, Any]:
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


def evaluate(
    task_params: LMHarnessTaskParams,
    output_dir: str,
    model_params: ModelParams,
    generation_params: GenerationParams,
    enable_wandb: bool,
    run_name: Optional[str] = None,
) -> dict[str, Any]:
    """Evaluates a model using the LM Evaluation Harness framework (EleutherAI).

    For detailed documentation, we refer you to the following readme:
    https://github.com/EleutherAI/lm-evaluation-harness

    Args:
        model_params: The parameters of the model to evaluate.
        task_params: The LM Harness parameters to use for evaluation.
        generation_params: The generation parameters to use for evaluation.
        output_dir: The directory where the evaluation results will be saved.
        enable_wandb: Whether to enable Weights & Biases (wandb) logging.
        run_name: Unique identifier for wandb for the current training run.

    Returns:
        The evaluation results (dict of metric names and their corresponding values).
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
    assert task_params is not None
    # If batch size isn't specified, we set it to "auto", which will let LM Harness
    # automatically select the largest batch size that will fit in memory.
    # https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md
    batch_size = (
        generation_params.batch_size if generation_params.batch_size else "auto"
    )

    # Get a timestamp for the current run.
    start_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = time.time()

    lm_harness_model_params = model_params.to_lm_harness()

    if is_image_text_llm(model_params):
        # Multimodal support is currently restricted to
        # the ['hf-multimodal', 'vllm-vlm'] model types.
        lm_harness_model = "hf-multimodal"
        apply_chat_template = True
        lm_harness_model_params.update(
            _create_extra_lm_harness_model_params_for_vlm(model_params)
        )
    else:
        lm_harness_model = "hf"
        # False is the default value for `simple_evaluate()`
        # TODO Should it be set to True?
        apply_chat_template = False

    logger.info("Starting evaluation...")
    logger.info(f"\tLM Harness `model_params`:\n{pformat(lm_harness_model_params)}")
    logger.info(f"\tLM Harness `task_params`:\n{pformat(task_params)}")
    lm_eval_output = lm_eval.simple_evaluate(
        model=lm_harness_model,
        model_args=lm_harness_model_params,
        tasks=[task_params.task_name],
        num_fewshot=task_params.num_fewshot,
        batch_size=batch_size,  # type: ignore
        device=device,
        limit=task_params.num_samples,
        log_samples=False,
        apply_chat_template=apply_chat_template,
        **task_params.eval_kwargs,  # type: ignore
    )
    elapsed_time_sec = time.time() - start_time

    # Metrics are only available on the main process, and `None` on others.
    if is_world_process_zero():
        assert lm_eval_output is not None
        task_name = task_params.task_name
        metric_dict = lm_eval_output["results"][task_name]  # type: ignore
        logger.info(f"{task_name}'s metric dict is {pformat(metric_dict)}")

        if enable_wandb:
            project_name = os.environ.get("WANDB_PROJECT", "oumi")
            logger.info(f"Logging to Weights and Biases project: '{project_name}'")
            wandb_logger = WandbLogger(
                project=project_name, name=run_name, job_type="eval"
            )
            wandb_logger.post_init(lm_eval_output)
            wandb_logger.log_eval_result()

        # The LM Harness platform's task configuration is a dictionary which
        # includes: the number of samples, the number of few-shots, task version(s),
        # the prompt(s) text, model/git hashes, seeds, and the special tokens used
        # by the tokenizer (such as `pad`, `eos`, `bos, and `eot`).
        platform_task_config = lm_eval_output

        # The LM Harness platform's results is a dictionary that includes all
        # evaluation metrics, which are oftentimes grouped (in `groups`) by a theme
        # or a classification category.
        platform_results = {
            key: platform_task_config.pop(key)
            for key in ["results", "groups"]
            if key in platform_task_config
        }

        if output_dir:
            save_evaluation_output(
                base_output_dir=output_dir,
                platform=task_params.get_evaluation_platform(),
                platform_results=copy.deepcopy(platform_results),
                platform_task_config=platform_task_config,
                task_params=task_params,
                start_time_str=start_time_str,
                elapsed_time_sec=elapsed_time_sec,
                model_params=model_params,
                generation_params=generation_params,
                inference_config=None,
            )

        return platform_results
    return {}
