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

from unittest.mock import MagicMock, patch

from oumi.builders.training import build_trainer
from oumi.core.configs import DpoParams, TrainerType, TrainingParams


def test_dpo_builder_passes_reference_log_probs_cache_dir():
    training_params = TrainingParams(
        trainer_type=TrainerType.TRL_DPO,
        dpo=DpoParams(
            precompute_ref_log_probs=True,
            ref_log_probs_cache_dir="/tmp/ref-logps",
        ),
    )

    with patch("oumi.builders.training.TrlDpoTrainer") as trainer_class:
        create_trainer = build_trainer(TrainerType.TRL_DPO, processor=None)
        create_trainer(
            model=MagicMock(),
            args=training_params,
            train_dataset=MagicMock(),
            processing_class=MagicMock(),
        )

    assert trainer_class.call_args.kwargs["ref_log_probs_cache_dir"] == (
        "/tmp/ref-logps"
    )
