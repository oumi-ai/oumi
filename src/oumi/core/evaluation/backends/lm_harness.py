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

from __future__ import annotations

import os
import random
from pprint import pformat
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch

if TYPE_CHECKING:
    from lm_eval.api.group import ConfigurableGroup
    from lm_eval.api.task import Task

from oumi.builders import build_processor, build_tokenizer
from oumi.builders.models import is_image_text_llm_using_model_name
from oumi.core.configs import (
    EvaluationConfig,
    GenerationParams,
    InferenceEngineType,
    LMHarnessTaskParams,
    ModelParams,
    RemoteParams,
)
from oumi.core.distributed import get_device_rank_info, is_world_process_zero
from oumi.core.evaluation.evaluation_result import EvaluationResult
from oumi.utils.distributed_utils import is_using_accelerate
from oumi.utils.logging import logger

# Used to set the few-shot seed for lm_eval.api.task.Task. The value is consistent with
# LM Harness `simple_evaluate`'s default `fewshot_random_seed` = 1234.
FEW_SHOT_SEED = 1234

########################################################################################
# How to map LM Harness `model_args` to Oumi's `ModelParams` and `GenerationParams`?   #
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
# device_map          | (always None)  | Υ  |      | Υ             |          |        #
# device              | (auto-detect)  | Υ  |      | Υ             |          |        #
# attn_implementation |                | Υ  |      | Υ             |          |        #
# ------------------- | -------------- | -- | ---- | ------------- | -------- | ------ #
# max_images          |                | NA | NA   | Υ             | Υ        | NA     #
# interleave          |                | NA | NA   | Υ             | Υ        | NA     #
# convert_img_format  |                | NA | NA   | Υ             |          | NA     #
# image_token_id      |                | NA | NA   | Υ             |          | NA     #
# image_string        |                | NA | NA   | Υ             |          | NA     #
########################################################################################
# LM Harness          | Oumi `generat- | LM Harness `model`                            #
# `model_args`        | ion_params`    | hf | vllm | hf-multimodal | vllm-vlm | remote #
# ------------------- | -------------- | -- | ---- | ------------- | -------- | ------ #
# batch_size          |                | Υ  | Υ    | Υ             | Υ        | Y      #
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
    generation_params: GenerationParams,
    inference_engine_type: InferenceEngineType,
    inference_remote_params: RemoteParams | None,
) -> dict[str, Any]:
    """Converts Oumi's ModelParams to LM Harness model arguments."""
    # List of all LM Harness model arguments:
    # https://github.com/EleutherAI/lm-evaluation-harness/blob/365fcda9b85bbb6e0572d91976b8daf409164500/lm_eval/models/huggingface.py#L66

    # If batch size isn't specified, we set it to "auto", which will let LM Harness
    # automatically select the largest batch size that will fit in memory.
    # https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md
    batch_size = generation_params.batch_size or "auto"

    model_args = {
        "trust_remote_code": model_params.trust_remote_code,
        "pretrained": model_params.model_name,
        "dtype": model_params.torch_dtype or model_params.torch_dtype_str,
        "max_length": model_params.model_max_length,
        "batch_size": batch_size,
    }

    if model_params.tokenizer_name:
        model_args["tokenizer"] = model_params.tokenizer_name

    # Add engine-specific arguments
    if inference_engine_type == InferenceEngineType.NATIVE:
        _add_native_engine_args(model_args, model_params)
    elif inference_engine_type == InferenceEngineType.VLLM:
        _add_vllm_engine_args(model_args, model_params)
    elif inference_engine_type == InferenceEngineType.REMOTE:
        _add_remote_engine_args(model_args, inference_remote_params)

    # Add multimodal arguments if needed
    if is_multimodal:
        _add_multimodal_args(model_args, lm_harness_model, model_params)

    # Pass through additional model_kwargs for quantization, etc.
    if model_params.model_kwargs:
        for key in ["load_in_4bit", "load_in_8bit", "max_memory_per_gpu"]:
            if key in model_params.model_kwargs:
                model_args[key] = model_params.model_kwargs[key]

    return model_args


def _apply_to_all_tasks(
    task_dict: dict[str | ConfigurableGroup, Task | dict],
    fn: Callable,
    fn_kwargs: dict[str, Any] | None = None,
) -> None:
    """Apply the provided function `fn` to all tasks in the `task_dict`."""
    from lm_eval.api.task import Task

    fn_kwargs = fn_kwargs or {}
    for task_obj in task_dict.values():
        if isinstance(task_obj, dict):
            _apply_to_all_tasks(task_obj, fn, fn_kwargs)
        elif isinstance(task_obj, Task):
            fn(task_obj, **fn_kwargs)
        else:
            raise ValueError(f"Expected `lm_eval.api.task.Task` but got: {task_obj}")


def _get_task_dict(
    task_params: LMHarnessTaskParams,
) -> dict[str | ConfigurableGroup, Task | dict]:
    """Get a dictionary of LM Harness tasks, given Oumi's `task_params`."""
    from lm_eval.api.group import ConfigurableGroup
    from lm_eval.tasks import get_task_dict as lm_harness_get_task_dict

    if not task_params.task_name:
        raise ValueError("The `task_name` must be specified for LM Harness evaluation.")
    task_dict: dict = lm_harness_get_task_dict(task_params.task_name)

    # Sanity checks for `task_dict`.
    if not task_dict:
        raise ValueError(f"Task `{task_params.task_name}` not available in LM Harness.")
    elif len(task_dict) > 1:
        raise ValueError(
            "Unexpected `task_dict` from LM Harness, consisting of multiple tasks, "
            f"while a single task ({task_params.task_name}) was requested: {task_dict}."
        )
    else:
        assert len(task_dict) == 1
        task_name = next(iter(task_dict))
        if isinstance(task_name, ConfigurableGroup):
            task_name: str = task_name.group_name
        if task_name != task_params.task_name:
            raise ValueError(
                f"Inconsistent task naming. Task `{task_params.task_name}` was "
                f"requested, but LM Harness returned the `task_dict`: {task_dict}."
            )

    # Apply the following function to each task in the task_dict, in order to overwrite
    # the default parameters with the ones specified in `task_params`.
    def overwrite_task_params(task: Task) -> None:
        # Set the number of few shots to be added to the prompt.
        task.set_config(key="num_fewshot", value=task_params.num_fewshot)

        # Set the random seed (for reproducibility and consistency with LM Harness).
        task.set_fewshot_seed(seed=FEW_SHOT_SEED)

    _apply_to_all_tasks(task_dict, fn=overwrite_task_params)

    return task_dict


def _set_random_seeds(random_seed, numpy_random_seed, torch_random_seed) -> None:
    """Setting random seeds for reproducibility and consistency with LM Harness."""
    if random_seed is not None:
        random.seed(random_seed)

    if numpy_random_seed is not None:
        np.random.seed(numpy_random_seed)

    if torch_random_seed is not None:
        torch.manual_seed(torch_random_seed)


def evaluate(
    task_params: LMHarnessTaskParams,
    config: EvaluationConfig,
    random_seed: int | None = 0,
    numpy_random_seed: int | None = 1234,
    torch_random_seed: int | None = 1234,
) -> EvaluationResult:
    """Evaluates a model using the LM Evaluation Harness framework (EleutherAI).

    For detailed documentation, we refer you to the following readme:
    https://github.com/EleutherAI/lm-evaluation-harness

    Args:
        task_params: The LM Harness parameters to use for evaluation.
        config: The evaluation configuration.
        random_seed: The random seed to use for python's `random` package.
        numpy_random_seed: The numpy random seed to use for reproducibility.
        torch_random_seed: The torch random seed to use for reproducibility.

    Note for random seeds (random_seed, numpy_random_seed, torch_random_seed):
        These have been set to be consistent with LM Harness' simple_evaluate().
        See: lm-evaluation-harness/blob/main/lm_eval/evaluator.py

    Returns:
        The evaluation results (dict of metric names and their corresponding values).
    """
    import lm_eval.loggers.utils as lm_harness_log_utils
    from lm_eval import evaluate as lm_harness_evaluate
    from lm_eval.api.registry import get_model as lm_harness_get_model_class
    from lm_eval.loggers import WandbLogger

    _set_random_seeds(
        random_seed=random_seed,
        numpy_random_seed=numpy_random_seed,
        torch_random_seed=torch_random_seed,
    )

    # Check GPU availability for engines that require it
    has_gpu = torch.cuda.is_available()

    # VLLM requires GPU
    if config.inference_engine == InferenceEngineType.VLLM and not has_gpu:
        raise ValueError("The `VLLM` inference_engine requires a CUDA-enabled GPU.")

    # NATIVE engine with accelerate requires GPU
    if (
        config.inference_engine == InferenceEngineType.NATIVE
        and is_using_accelerate()
        and not has_gpu
    ):
        raise ValueError(
            "The `NATIVE` inference_engine with distributed execution "
            "(e.g., via `accelerate launch`) requires a CUDA-enabled GPU."
        )

    # Identify whether the model is multi-modal.
    is_multimodal = is_image_text_llm_using_model_name(
        model_name=config.model.model_name,
        trust_remote_code=config.model.trust_remote_code,
    )

    # Identify the proper LM Harness model (`lm_harness_model`) to use.
    if config.inference_engine == InferenceEngineType.NATIVE:
        lm_harness_model = "hf-multimodal" if is_multimodal else "hf"
    elif config.inference_engine == InferenceEngineType.VLLM:
        lm_harness_model = "vllm-vlm" if is_multimodal else "vllm"
    elif config.inference_engine == InferenceEngineType.REMOTE:
        lm_harness_model = "local-completions"
    else:
        raise ValueError(
            f"Unsupported inference engine type: {config.inference_engine}. "
            "Our integration with the `lm_harness` evaluation backend supports "
            "the `NATIVE`, `VLLM` and `REMOTE` inference_engine types."
        )

    # Instantiate an LM Harness task dictionary.
    task_dict = _get_task_dict(task_params)
    logger.info(f"\tLM Harness `task_params`:\n{pformat(task_params)}")
    logger.info(f"\tLM Harness `task_dict`:\n{pformat(task_dict)}")

    # Instantiate an LM Harness language model (lm).
    lm_harness_model_params = _generate_lm_harness_model_args(
        lm_harness_model=lm_harness_model,
        is_multimodal=is_multimodal,
        model_params=config.model,
        generation_params=config.generation,
        inference_engine_type=config.inference_engine,
        inference_remote_params=config.inference_remote_params,
    )
    logger.info(f"\tLM Harness `model_params`:\n{pformat(lm_harness_model_params)}")
    lm_class = lm_harness_get_model_class(lm_harness_model)
    lm = lm_class(**lm_harness_model_params)

    logger.info("Starting evaluation...")

    lm_eval_output = lm_harness_evaluate(
        lm,
        task_dict,
        log_samples=task_params.log_samples or False,
        limit=task_params.num_samples,
        apply_chat_template=is_multimodal,
        **task_params.eval_kwargs,  # type: ignore
    )

    # Metrics are only available on the main process, and `None` on others.
    if not is_world_process_zero():
        return EvaluationResult()

    assert lm_eval_output is not None
    task_name = task_params.task_name
    metric_dict = lm_eval_output["results"][task_name]  # type: ignore
    logger.info(f"{task_name}'s metric dict is {pformat(metric_dict)}")

    if config.enable_wandb:
        project_name = os.environ.get("WANDB_PROJECT", "oumi")
        logger.info(f"Logging to Weights and Biases project: '{project_name}'")
        wandb_logger = WandbLogger(
            init_args={
                "project": project_name,
                "name": config.run_name,
                "job_type": "eval",
            },
            config_args={"config": config},
        )
        wandb_logger.post_init(lm_eval_output)
        wandb_logger.log_eval_result()

    # The LM Harness backend's task configuration is a dictionary which
    # includes: the number of samples, the number of few-shots, task version(s),
    # the prompt(s) text, model/git hashes, seeds, and the special tokens used
    # by the tokenizer (such as `pad`, `eos`, `bos, and `eot`).
    backend_task_config = lm_eval_output

    # The LM Harness backend's results is a dictionary that includes all
    # evaluation metrics, which are oftentimes grouped (in `groups`) by a theme
    # or a classification category.
    backend_results = {
        key: backend_task_config.pop(key)
        for key in ["results", "groups"]
        if key in backend_task_config
    }

    # Add LM Harness-specific configuration settings to the results.
    backend_task_config.setdefault("config", {})

    # Add configuration settings related to the model.
    backend_task_config["config"]["model"] = lm_harness_model
    backend_task_config["config"]["model_args"] = lm_harness_model_params
    if hasattr(lm, "get_model_info"):
        backend_task_config["config"].update(lm.get_model_info())

    # Add configuration settings related to the task.
    backend_task_config["config"]["task_params"] = task_params
    backend_task_config["config"]["task_dict"] = task_dict

    # Add other configuration settings.
    backend_task_config["git_hash"] = lm_harness_log_utils.get_git_commit_hash()
    lm_harness_log_utils.add_env_info(backend_task_config)
    lm_harness_log_utils.add_tokenizer_info(backend_task_config, lm)

    return EvaluationResult(
        task_name=task_params.task_name,
        task_result=backend_results,
        backend_config=backend_task_config,
    )


def _add_native_engine_args(
    model_args: dict[str, Any],
    model_params: ModelParams,
) -> None:
    """Add NATIVE (HuggingFace) engine-specific arguments."""
    model_args["device_map"] = None

    if torch.cuda.is_available():
        device_info = get_device_rank_info()
        device = f"cuda:{device_info.local_rank}"
        logger.info(f"Using {device_info.world_size} GPUs for evaluation.")
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using MPS for evaluation")
    else:
        device = "cpu"
        logger.info("Using CPU for evaluation")

    model_args["device"] = device
    model_args["parallelize"] = model_params.shard_for_eval

    # Optional NATIVE-specific parameters
    if model_params.adapter_model:
        model_args["peft"] = model_params.adapter_model
    if model_params.attn_implementation:
        model_args["attn_implementation"] = model_params.attn_implementation


def _add_vllm_engine_args(
    model_args: dict[str, Any],
    model_params: ModelParams,
) -> None:
    """Add VLLM engine-specific arguments.

    Supports tensor parallelism: Splits model across GPUs (for large models).
    """
    # Tensor parallelism: from model_kwargs or auto-detect
    tensor_parallel_size = (
        model_params.model_kwargs.get("tensor_parallel_size", -1)
        if torch.cuda.is_available()
        else 1
    )
    if tensor_parallel_size <= 0:
        # Auto-detect: use all available GPUs
        tensor_parallel_size = (
            torch.cuda.device_count() if torch.cuda.is_available() else 1
        )
    model_args["tensor_parallel_size"] = tensor_parallel_size

    logger.info(f"VLLM parallelism: tensor_parallel_size={tensor_parallel_size}.")

    if model_args.get("batch_size", "auto") != "auto":
        logger.info(
            "Setting batch_size to 'auto' for VLLM evaluation. "
            "To set the maximum batch size, use the `max_batch_size` in model_kwargs."
        )
        model_args["batch_size"] = "auto"

    if model_params.model_kwargs.get("max_batch_size") is not None:
        model_args["max_batch_size"] = model_params.model_kwargs.get("max_batch_size")


def _add_remote_engine_args(
    model_args: dict[str, Any],
    inference_remote_params: RemoteParams | None,
) -> None:
    """Add REMOTE engine-specific arguments."""
    if not inference_remote_params:
        raise ValueError(
            "The `REMOTE` inference engine requires `inference_remote_params`."
        )

    model_args["base_url"] = inference_remote_params.api_url
    if inference_remote_params.num_workers > 0:
        model_args["num_concurrent"] = inference_remote_params.num_workers
    if inference_remote_params.max_retries > 0:
        model_args["max_retries"] = inference_remote_params.max_retries
    if inference_remote_params.connection_timeout > 0:
        model_args["timeout"] = int(inference_remote_params.connection_timeout)


def _add_multimodal_args(
    model_args: dict[str, Any],
    lm_harness_model: str,
    model_params: ModelParams,
) -> None:
    """Add multimodal (vision) model arguments."""
    # TODO: OPE-355 - To remove `max_images=1` limit
    model_args["max_images"] = 1
    model_args["interleave"] = True

    # HuggingFace multimodal needs additional processor setup
    if lm_harness_model == "hf-multimodal":
        model_args["convert_img_format"] = True

        tokenizer = build_tokenizer(model_params)
        processor = build_processor(
            model_params.model_name,
            tokenizer,
            trust_remote_code=model_params.trust_remote_code,
            processor_kwargs=model_params.processor_kwargs,
        )
        if image_token := processor.image_token:
            model_args["image_string"] = image_token
        if image_token_id := processor.image_token_id:
            model_args["image_token_id"] = image_token_id

    if model_args.get("batch_size") == "auto":
        logger.info("Setting batch_size to 1 for multimodal evaluation.")
        model_args["batch_size"] = 1
