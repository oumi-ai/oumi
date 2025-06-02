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

"""Simple Bitnet model saving callback."""

from pathlib import Path
from typing import Optional, Union

import transformers
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from oumi.core.callbacks.base_trainer_callback import BaseTrainerCallback
from oumi.core.configs import TrainingParams

# Import `onebitllms` utils methods
try:
    import onebitllms  # type: ignore
    from onebitllms import quantize_to_1bit  # type: ignore
except ImportError:
    onebitllms = None


class BitNetCallback(BaseTrainerCallback):
    """BitNet model saving callback.

    Simple callback that saves the model into BitNet quantized
    format during training.
    """

    def on_save(
        self,
        args: Union[transformers.TrainingArguments, TrainingParams],
        state: Optional[transformers.TrainerState] = None,
        control: Optional[transformers.TrainerControl] = None,
        **kwargs,
    ):
        """Saving callback.

        Gets triggered at each saving step to quantize trained models
        in 1bit precision.
        """
        if onebitllms is None:
            raise ValueError(
                """You need `onebitllms` to be installed in order to save
                correctly BitNet models - `pip install onebitllms`"""
            )

        output_dir = Path(args.output_dir)  # type: ignore
        quantized_subdir = Path(
            f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}-quantized"  # type: ignore
        )
        output_subdir = Path(f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")  # type: ignore

        checkpoint_folder = output_dir / output_subdir
        quantized_checkpoint_folder = output_dir / quantized_subdir

        quantize_to_1bit(str(checkpoint_folder), str(quantized_checkpoint_folder))
