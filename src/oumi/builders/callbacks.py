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

from typing import Any, Optional

import torch

from oumi.core.callbacks.base_trainer_callback import BaseTrainerCallback
from oumi.core.callbacks.bitnet_callback import BitNetCallback
from oumi.core.callbacks.hf_mfu_callback import HfMfuTrainerCallback
from oumi.core.callbacks.mfu_callback import MfuTrainerCallback
from oumi.core.callbacks.nan_inf_detection_callback import NanInfDetectionCallback
from oumi.core.callbacks.profiler_step_callback import ProfilerStepCallback
from oumi.core.callbacks.telemetry_callback import TelemetryCallback
from oumi.core.configs import TrainerType, TrainingConfig
from oumi.performance.mfu import _get_device_flops
from oumi.utils.logging import logger
from oumi.utils.torch_utils import (
    count_model_parameters,
    get_device_name,
)


def build_training_callbacks(
    config: TrainingConfig, model: torch.nn.Module, profiler: Optional[Any]
) -> list[BaseTrainerCallback]:
    """Builds the training callbacks for the given training config and model.

    This function creates a list of callback objects to be used during training.
    It includes callbacks for performance metrics, profiling, telemetry, and
    Model Flops Utilization (MFU) logging based on the provided configuration.

    Args:
        config: The training configuration object.
        model: The PyTorch model being trained. This is needed to calculate
               the number of parameters for MFU (Model Flops Utilization) logging,
               and to determine the model's data type for accurate MFU calculations.
        profiler: The profiler object, if profiling is enabled.

    Returns:
        List[BaseTrainerCallback]: A list of callback objects to be used
        during training.

    Note:
        - MFU logging is only supported on GPU and is skipped for PEFT models or
          training with non-packed datasets.
    """
    result: list[BaseTrainerCallback] = []
    if not config.training.include_performance_metrics:
        return result

    dtype = next(model.parameters()).dtype
    add_mfu_callbacks: bool = True
    if not torch.cuda.is_available():
        logger.warning("MFU logging is only supported on GPU. Skipping MFU callbacks.")
        add_mfu_callbacks = False
    elif config.training.use_peft:
        logger.warning("MFU logging is not supported for PEFT. Skipping MFU callbacks.")
        add_mfu_callbacks = False
    elif not config.data.train.pack:
        logger.warning("MFU logging requires packed datasets. Skipping MFU callbacks.")
        add_mfu_callbacks = False
    else:
        device_name = get_device_name()
        try:
            _get_device_flops(device_name, dtype)
        except NotImplementedError:
            logger.warning(
                f"MFU logging is currently not supported for device {device_name}. "
                "Skipping MFU callbacks."
            )
            add_mfu_callbacks = False

    if add_mfu_callbacks:
        if config.model.model_max_length is not None and (
            config.model.model_max_length > 0
        ):
            num_total_params = count_model_parameters(model)
            num_mfu_params = (
                num_total_params.all_params - num_total_params.embedding_params
            )
            logger.info(f"Number of model parameters for MFU: {num_mfu_params:,}")
            # Ignore attention and rematerialization to ensure metric matches most
            # common implementations.
            mfu_callback = MfuTrainerCallback(
                dtype=dtype,
                num_params=num_mfu_params,
                sequence_length=config.model.model_max_length,
            )
            result.append(mfu_callback)
        else:
            logger.warning(
                "model_max_length must be set to log MFU performance information."
            )

        if config.training.include_alternative_mfu_metrics and (
            config.training.trainer_type
            in (
                TrainerType.TRL_SFT,
                TrainerType.TRL_DPO,
                TrainerType.TRL_GRPO,
                TrainerType.HF,
            )
        ):
            result.append(HfMfuTrainerCallback(dtype=dtype))

    if profiler is not None:
        result.append(ProfilerStepCallback(profiler=profiler))
    elif config.training.profiler.schedule.enable_schedule:
        logger.warning(
            "Scheduled profiling is requested, but profiler is not available!"
        )

    # Loss can have different names, depending on context.
    result.append(
        NanInfDetectionCallback(metrics=["loss", "train/loss", " train_loss"])
    )

    # TelemetryCallback goes last to make sure it can read MFU metrics.
    result.append(
        TelemetryCallback(
            skip_first_steps=2,
            world_process_zero_only=(
                not config.training.telemetry.collect_telemetry_for_all_ranks
            ),
            output_dir=config.training.telemetry_dir,
            track_gpu_temperature=config.training.telemetry.track_gpu_temperature,
        )
    )

    # Add the BitNet saving callbacks if relevant
    if config.model.model_name in (
        "tiiuae/Falcon-E-1B-Base",
        "tiiuae/Falcon-E-1B-Instruct",
        "tiiuae/Falcon-E-3B-Base",
        "tiiuae/Falcon-E-3B-Instruct",
    ):
        result.append(BitNetCallback())

    return result
