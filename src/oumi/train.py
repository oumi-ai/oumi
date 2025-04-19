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

import os
import time
from pathlib import Path
from typing import Any, Callable, Final, Optional

import ray
import transformers

from oumi.builders import (
    build_dataset_mixture,
    build_reward_functions,
    build_tokenizer,
    build_trainer,
)
from oumi.core.configs import (
    DatasetSplit,
    TrainerType,
    TrainingConfig,
)
from oumi.utils.logging import configure_logger, logger


def train(
    config: TrainingConfig,
    additional_model_kwargs: Optional[dict[str, Any]] = None,
    additional_trainer_kwargs: Optional[dict[str, Any]] = None,
) -> None:
    """Trains a model using the provided configuration."""
    _START_TIME = time.time()
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get(
        "CUDA_VISIBLE_DEVICES", ""
    )

    logger.info(f"Available resources: {ray.available_resources()}")

    log_dir = Path(config.training.output_dir) / "logs"
    for logger_name in ("oumi", "oumi.telemetry"):
        configure_logger(logger_name, level=config.training.log_level, log_dir=log_dir)

    tokenizer = build_tokenizer(config.model)

    dataset = build_dataset_mixture(
        config.data,
        tokenizer,
        DatasetSplit.TRAIN,
        seq_length=config.model.model_max_length,
    )

    eval_dataset = None
    if len(config.data.get_split(DatasetSplit.VALIDATION).datasets) != 0:
        eval_dataset = build_dataset_mixture(
            config.data,
            tokenizer,
            DatasetSplit.VALIDATION,
            seq_length=config.model.model_max_length,
        )

    trainer_type: Final[TrainerType] = config.training.trainer_type
    create_trainer_fn = build_trainer(trainer_type, processor=None)

    reward_functions: list[Callable] = build_reward_functions(config.training)

    trainer = create_trainer_fn(
        processing_class=tokenizer,
        args=config.training,
        reward_funcs=reward_functions,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
    )

    logger.info(f"Training init time: {time.time() - _START_TIME:.3f}s")
    logger.info(
        f"Starting training... "
        f"({config.training.trainer_type}, "
        f"transformers: {transformers.__version__})"
    )
    trainer.train(resume_from_checkpoint=None)

    logger.info("Training is Complete.")
