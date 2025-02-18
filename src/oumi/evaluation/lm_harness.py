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

from oumi.builders import build_processor, build_tokenizer
from oumi.builders.models import is_image_text_llm_using_model_name
from oumi.core.configs import (
    GenerationParams,
    InferenceEngineType,
    LMHarnessTaskParams,
    ModelParams,
    RemoteParams,
)
from oumi.core.distributed import is_world_process_zero
from oumi.evaluation.save_utils import save_evaluation_output
from oumi.utils.logging import logger

########################################################################################
# How to map LM Harness `model_args` to Oumi's `ModelParams` for evaluation?           #
# Which LM Harness `model` types (hf, vllm, etc) support each parameter?               #
# ------------------- | -------------- | -- | ---- | ------------- | -------- | ------ #
# LM Harness          | Oumi           | LM Harness `model`                            #
# `model_args`        | `model_params` | hf | vllm | hf-multimodal | vllm-vlm | remote #
# ------------------- | -------------- | -- | ---- | ------------- | -------- | ------ #
# trust_remote_code   |                | Υ  | Υ    | Υ             | Υ        | Y      #
# pretrained          | model_name     | Υ  | Υ    | Υ             | Υ        | Y      #
# dtype               | torch_dtype    | Υ  | Υ    | Υ             | Υ        | Y      #
# max_length          |model_max_length| Υ  | Υ    | Υ             | Υ        | Y      #
# tokenizer           | tokenizer_name | Υ  | Υ    | Υ             | Υ        | Y      #
# peft                | adapter_model  | Υ  |      | Υ             |          |        #
# parallelize         | shard_for_eval | Υ  |      | Υ             |          |        #
# device_map          |                | ?? |      | ??            |          |        #
# attn_implementation |                | ?? |      | ??            |          |        #
# ------------------- | -------------- | -- | ---- | ------------- | -------- | ------ #
# max_images          |                | NA | NA   | Υ             | Υ        | NA     #
# interleave          |                | NA | NA   | Υ             | Υ        | NA     #
# convert_img_format  |                | NA | NA   | Υ             |          | NA     #
# image_token_id      |                | NA | NA   | Υ             |          | NA     #
# image_string        |                | NA | NA   | Υ             |          | NA     #
########################################################################################

########################################################################################
# How to map LM Harness `model_args` (specifically the ones related to a remote        #
# inference engine) to Oumi's `remote_params`?                                         #
# ----------------------- | ---------------------------------------------------------- #
# LM Harness `model_args` | Oumi `remote_params`                                       #
# ----------------------- | ---------------------------------------------------------- #
# base_url                |  api_url                                                   #
# num_concurrent          |  num_workers                                               #
# max_retries             |  max_retries                                               #
# timeout                 |  connection_timeout                                        #
########################################################################################

########################################################################################
# Mapping of LM Harness `model` types to the corresponding class and file              #
# --------------------------|---------------------|----------------------------------- #
# LM Harness `model`        | Class               | File in lm-evaluation-harness repo #
# (= inference engine)      | name                | located under lm_eval/models/...   #
# --------------------------|---------------------|----------------------------------- #
# hf                        | HFLM                | huggingface.py                     #
# vllm                      | VLLM                | vllm_causallms.py                  #
# hf-multimodal             | HFMultimodalLM      | hf_vlms.py                         #
# vllm-vlm                  | VLLM_VLM            | vllm_vlms.py                       #
# local-completions         | LocalCompletionsAPI | openai_completions.py              #
########################################################################################


def _generate_lm_harness_model_args(
    lm_harness_model: str,
    is_multimodal: bool,
    model_params: ModelParams,
    inference_engine_type: InferenceEngineType,
    inference_remote_params: Optional[RemoteParams],
) -> dict[str, Any]:
    """Converts Oumi's ModelParams to LM Harness model arguments."""
    # Arguments used across all engines and modalities.
    model_args_dict = {
        "trust_remote_code": model_params.trust_remote_code,
        "pretrained": model_params.model_name,
        "dtype": model_params.torch_dtype,
        "max_length": model_params.model_max_length,
    }
    if model_params.tokenizer_name:
        model_args_dict["tokenizer"] = model_params.tokenizer_name

    # Add NATIVE inference engine's additional parameters.
    if inference_engine_type == InferenceEngineType.NATIVE:
        model_args_dict["parallelize"] = model_params.shard_for_eval
        model_args_dict["device_map"] = model_params.device_map
        if model_params.adapter_model:
            model_args_dict["peft"] = model_params.adapter_model
        if model_params.attn_implementation:
            model_args_dict["attn_implementation"] = model_params.attn_implementation

    # Add REMOTE inference engine's additional parameters.
    if inference_engine_type == InferenceEngineType.REMOTE:
        if not inference_remote_params:
            raise ValueError(
                "The `REMOTE` inference engine requires `inference_remote_params`."
            )
        model_args_dict["base_url"] = inference_remote_params.api_url
        if inference_remote_params.num_workers > 0:
            model_args_dict["num_concurrent"] = inference_remote_params.num_workers
        if inference_remote_params.max_retries > 0:
            model_args_dict["max_retries"] = inference_remote_params.max_retries
        if inference_remote_params.connection_timeout > 0:
            model_args_dict["timeout"] = int(inference_remote_params.connection_timeout)

    # Add multi-modal related parameters.
    # details at https://github.com/EleutherAI/lm-evaluation-harness/releases/tag/v0.4.5
    if is_multimodal:
        # FIXME OPE-355 To remove `max_images=1` limit
        model_args_dict |= {"max_images": 1, "interleave": True}

        # Only applicable to hf-multimodal (NOT vllm-vlm).
        if lm_harness_model == "hf-multimodal":
            model_args_dict["convert_img_format"] = True

            tokenizer = build_tokenizer(model_params)
            processor = build_processor(
                model_params.model_name,
                tokenizer,
                trust_remote_code=model_params.trust_remote_code,
            )
            if image_token := processor.image_token:
                model_args_dict["image_string"] = image_token
            if image_token_id := processor.image_token_id:
                model_args_dict["image_token_id"] = image_token_id

    # Handle extra model_kwargs (construction arguments).
    # Towards OPE-564.
    if model_params.model_kwargs:
        for key in ["load_in_4bit", "load_in_8bit", "max_memory_per_gpu"]:
            if key in model_params.model_kwargs:
                model_args_dict[key] = model_params.model_kwargs[key]
        # TODO: load_in_8bit, load_in_4bit are deprecated and will be removed in
        # future versions of HF. Integrate via PeftConfig.
    return model_args_dict


def evaluate(
    task_params: LMHarnessTaskParams,
    output_dir: str,
    model_params: ModelParams,
    generation_params: GenerationParams,
    enable_wandb: bool,
    inference_engine_type: InferenceEngineType,
    inference_remote_params: Optional[RemoteParams] = None,
    run_name: Optional[str] = None,
) -> dict[str, Any]:
    """Evaluates a model using the LM Evaluation Harness framework (EleutherAI).

    For detailed documentation, we refer you to the following readme:
    https://github.com/EleutherAI/lm-evaluation-harness

    Args:
        task_params: The LM Harness parameters to use for evaluation.
        output_dir: The directory where the evaluation results will be saved.
        model_params: The parameters of the model to evaluate.
        generation_params: The generation parameters to use for evaluation.
        enable_wandb: Whether to enable Weights & Biases (wandb) logging.
        inference_engine_type: The inference engine to use (`VLLM`, `NATIVE`, `REMOTE`).
        inference_remote_params: The parameters for remote inference, if applicable.
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

    # Identify whether the model is multi-modal.
    is_multimodal = is_image_text_llm_using_model_name(
        model_name=model_params.model_name,
        trust_remote_code=model_params.trust_remote_code,
    )

    # Identify the proper LM Harness model (`lm_harness_model`) to use.
    if inference_engine_type == InferenceEngineType.NATIVE:
        lm_harness_model = "hf-multimodal" if is_multimodal else "hf"
        if device.startswith("cuda"):
            logger.warning(
                "Since you have GPU support, it is highly recommended that you set "
                "the `inference_engine` to `VLLM`, instead of the `NATIVE`, for faster "
                "evaluation."
            )
    elif inference_engine_type == InferenceEngineType.VLLM:
        lm_harness_model = "vllm-vlm" if is_multimodal else "vllm"
        if not device.startswith("cuda"):
            raise ValueError("The `VLLM` inference_engine requires a CUDA-enabled GPU.")
    elif inference_engine_type == InferenceEngineType.REMOTE:
        lm_harness_model = "local-completions"
    else:
        raise ValueError(
            f"Unsupported inference engine type: {inference_engine_type}. "
            "Our integration with the `lm_harness` evaluation platform supports "
            "the `NATIVE`, `VLLM` and `REMOTE` inference_engine types."
        )

    if model_params.adapter_model:
        logger.info(f"Loading adapter for eval: {model_params.adapter_model}")
    assert task_params is not None
    # If batch size isn't specified, we set it to "auto", which will let LM Harness
    # automatically select the largest batch size that will fit in memory.
    # https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md
    batch_size = (
        generation_params.batch_size if generation_params.batch_size else "auto"
    )

    lm_harness_model_params = _generate_lm_harness_model_args(
        lm_harness_model=lm_harness_model,
        is_multimodal=is_multimodal,
        model_params=model_params,
        inference_engine_type=inference_engine_type,
        inference_remote_params=inference_remote_params,
    )

    # Get a timestamp for the current run.
    start_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = time.time()

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
        apply_chat_template=is_multimodal,
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
