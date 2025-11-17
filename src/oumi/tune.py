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

import time
from importlib.metadata import version
from pathlib import Path
from pprint import pformat
from typing import Any, Optional

from oumi.builders import (
    build_tuner,
)
from oumi.core.configs import (
    EvaluationConfig,
    PeftParams,
    TrainerType,
    TrainingConfig,
    TrainingParams,
    TuningConfig,
)
from oumi.core.distributed import (
    barrier,
    cleanup_distributed,
    get_device_rank_info,
    is_distributed,
    is_local_process_zero,
    is_world_process_zero,
)
from oumi.train import _ensure_dir_exists, _log_feedback_request, train
from oumi.utils.device_utils import (
    log_nvidia_gpu_runtime_info,
)
from oumi.utils.git_utils import get_git_revision_hash, get_git_tag
from oumi.utils.io_utils import save_json
from oumi.utils.logging import configure_logger, logger
from oumi.utils.torch_utils import (
    device_cleanup,
    log_devices_info,
    log_peak_gpu_memory,
    log_versioning_info,
)
from oumi.utils.version_utils import is_dev_build


def _create_tuning_dirs(config: TuningConfig) -> None:
    """Creates misc directories referenced in config."""
    _ensure_dir_exists(config.tuning.output_dir, "tuning.output_dir")
    telemetry_dir = config.tuning.telemetry_dir
    if telemetry_dir:
        _ensure_dir_exists(telemetry_dir, "training.telemetry_dir")


def _log_tuning_info(config: TuningConfig) -> None:
    """Logs misc infos about training config/devices/etc. Writes to files."""
    telemetry_dir = config.tuning.telemetry_dir
    if telemetry_dir and is_world_process_zero():
        device_rank_info = get_device_rank_info()
        save_json(
            {
                "LOCAL_WORLD_SIZE": device_rank_info.local_world_size,
                "WORLD_SIZE": device_rank_info.world_size,
            },
            telemetry_dir / "world_size.json",
        )

    if is_local_process_zero():
        log_versioning_info()
        log_devices_info(
            (telemetry_dir / "devices_info.txt")
            if telemetry_dir and is_world_process_zero()
            else None
        )
        oumi_version = version("oumi")
        logger.info(f"Oumi version: {oumi_version}")
        if is_dev_build():
            logger.info(f"Git revision hash: {get_git_revision_hash()}")
            logger.info(f"Git tag: {get_git_tag()}")


def tune(
    config: TuningConfig,
    additional_model_kwargs: Optional[dict[str, Any]] = None,
    additional_tuner_kwargs: Optional[dict[str, Any]] = None,
    verbose: bool = False,
) -> None:
    """Tunes a model using the provided configuration."""
    _START_TIME = time.time()

    _create_tuning_dirs(config)
    _log_tuning_info(config)

    # Configure logging to file
    log_dir = Path(config.tuning.output_dir) / "logs"
    for logger_name in ("oumi", "oumi.telemetry"):
        configure_logger(
            logger_name,
            level=config.tuning.log_level,
            log_dir=log_dir,
        )

    telemetry_dir = config.tuning.telemetry_dir
    if is_local_process_zero():
        if verbose:
            logger.info(f"TuningConfig:\n{pformat(config)}")
        if telemetry_dir and is_world_process_zero():
            config_path = telemetry_dir / "tuning_config.yaml"
            config.to_yaml(str(config_path))
            logger.info(f"Training config saved to {config_path}")

    # metrics_functions = build_metrics_functions(config.tuning)
    tuner = build_tuner(config.tuning)

    def objective(
        suggested_training_params: dict[str, Any],
        suggested_peft_params: dict[str, Any],
        trial_number: int,
    ) -> dict[str, float]:
        """Objective function for hyperparameter tuning."""
        # Merge suggested training params with fixed params
        training_params = {
            **config.tuning.fixed_training_params,
            **suggested_training_params,
        }
        training_params["output_dir"] = str(
            Path(
                config.tuning.output_dir,
                f"trial_{trial_number}",
            )
        )
        trial_training_params = TrainingParams(**training_params)
        trial_training_params.trainer_type = TrainerType.TRL_SFT

        # Merged suggested and fixed PEFT params
        peft_params = {
            **config.tuning.fixed_peft_params,
            **suggested_peft_params,
        }
        trial_peft_params = PeftParams(**peft_params)

        trial_train_config = TrainingConfig(
            model=config.model,
            data=config.data,
            training=trial_training_params,
            peft=trial_peft_params,
        )

        # Trains model for this trial
        # TODO: This could be a boolean, but I think for the future a dict like this is
        # good.
        eval_results = train(
            trial_train_config, additional_tuning_kwargs={"some_variable": True}
        )

        # TODO: Add better support for tuning with custom evaluations results
        #  there should be a better way to do this other then reloading
        #  everything post-training.
        custom_results = {}
        if config.tuning.custom_eval_metrics:
            # Delayed import in case the user needs custom evaluations
            from oumi.core.configs import EvaluationTaskParams
            from oumi.core.configs.params.evaluation_params import EvaluationBackend
            from oumi.core.configs.params.generation_params import GenerationParams
            from oumi.evaluate import evaluate

            task_names = config.tuning.custom_eval_metrics
            # Build EvaluationConfig mirroring training configuration
            eval_model_params = trial_train_config.model
            # Load the just-saved fine-tuned model from output_dir
            eval_model_params.model_name = trial_train_config.training.output_dir

            eval_tasks = [
                EvaluationTaskParams(
                    evaluation_backend=EvaluationBackend.CUSTOM.value,
                    task_name=task_name,
                )
                for task_name in task_names
            ]

            generation = GenerationParams(
                batch_size=(
                    trial_train_config.training.per_device_eval_batch_size or 1
                ),
                seed=trial_train_config.training.seed,
                max_new_tokens=(
                    eval_model_params.model_max_length
                    if eval_model_params.model_max_length
                    else 1024
                ),
            )

            eval_output_dir = str(
                Path(trial_train_config.training.output_dir) / "custom_eval"
            )

            eval_config = EvaluationConfig(
                tasks=eval_tasks,
                model=eval_model_params,
                generation=generation,
                run_name=trial_train_config.training.run_name,
                enable_wandb=trial_train_config.training.enable_wandb,
                output_dir=eval_output_dir,
            )

            device_cleanup()
            task_results = evaluate(eval_config)
            # Unpack result metrics
            custom_results = {
                task_name: task_result[task_name]
                for task_name, task_result in zip(task_names, task_results)
            }

        if not eval_results:
            logger.error(
                "Eval results needs to be an dictionary of metrics, "
                "for tune to work correctly"
            )
            raise

        logger.info(
            f"Trial {trial_number} finished. Evaluation results: {eval_results}"
        )

        output_metrics_dict = {}
        metrics_list = list(config.tuning.evaluation_metrics)
        if config.tuning.custom_eval_metrics:
            metrics_list.extend(config.tuning.custom_eval_metrics)
        for metric in metrics_list:
            if metric in eval_results:
                output_metrics_dict[metric] = eval_results[metric]
            elif metric in custom_results:
                output_metrics_dict[metric] = custom_results[metric]

        return output_metrics_dict

    logger.info(f"Tuning init time: {time.time() - _START_TIME:.3f}s")
    logger.info("Starting hyperparameter tuning...")
    tuner.optimize(objective, n_trials=config.tuning.n_trials)

    if len(config.tuning.evaluation_direction) == 1:
        best_trial = tuner.get_best_trial()
        logger.info(f"Best trial: {best_trial}")
    else:
        best_trials = tuner.get_best_trials()
        logger.info(f"Best trials: {pformat(best_trials)}")

    logger.info(
        "Tuning is Complete. Saving study results at "
        f"{config.tuning.output_dir}/trials_results.csv ..."
    )
    tuner.save_study(config)

    log_nvidia_gpu_runtime_info(log_prefix="GPU Metrics After Tuning:")
    log_peak_gpu_memory()
    barrier()
    if is_distributed():
        cleanup_distributed()

    _log_feedback_request()
