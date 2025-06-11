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

import pathlib
from typing import Optional, cast

import peft
import transformers

from oumi.core.configs import TrainingConfig
from oumi.core.configs.params.peft_params import PeftSaveMode
from oumi.core.distributed import is_world_process_zero
from oumi.core.processors.base_processor import BaseProcessor
from oumi.core.trainers.base_trainer import BaseTrainer
from oumi.utils.logging import logger


class HuggingFaceTrainer(BaseTrainer):
    def __init__(
        self,
        hf_trainer: transformers.Trainer,
        processor: Optional[BaseProcessor] = None,
    ):
        """Initializes HuggingFace-specific Trainer version."""
        self._hf_trainer = hf_trainer
        self._processor = processor

    def train(self, resume_from_checkpoint: Optional[str] = None) -> None:
        """Trains a model."""
        self._hf_trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    def save_state(self) -> None:
        """See base class.

        Saves the Trainer state, since Trainer.save_model saves only the tokenizer
        with the model.

        HuggingFace normally writes state into "trainer_state.json" under output_dir.
        """
        if not is_world_process_zero():
            return

        self._hf_trainer.save_state()

    def save_model(self, config: TrainingConfig, final: bool = True) -> None:
        """Saves the model's weights to the specified output directory.

        Args:
            config: The Oumi training config.
            final: Whether this is the final model being saved during training.
                - Applies optimizations for the final model checkpoint.
                - In the case of FSDP, this will always save the FULL_STATE_DICT
                instead of the default STATE_DICT.

        Returns:
            None
        """
        if self._hf_trainer.is_fsdp_enabled:
            # FSDP is enabled, so we need to save the model in a special way.
            return self._save_fsdp_model(config=config, final=final)
        else:
            return self._save_model(config=config, final=final)

    def _save_model(self, config: TrainingConfig, final: bool = True) -> None:
        """Saves the model's weights to the specified output directory."""
        if not is_world_process_zero():
            return

        output_dir = config.training.output_dir
        if not config.training.use_peft:
            self._hf_trainer.save_model(output_dir)
        else:
            if config.peft.peft_save_mode == PeftSaveMode.MERGED:
                # Saving the merged model only saves the model weights, not the
                # tokenizer files and training args. To ensure we're saving all relevant
                # files, we save the PEFT model first, delete the adapter files, then
                # save the merged model.
                # The adapter files are moved to the "adapter/" subdirectory to not
                # interfere with the other saved model files.

                self._hf_trainer.save_model(output_dir)
                output_dir_path = pathlib.Path(output_dir)
                adapter_dir = output_dir_path / "adapter"
                adapter_dir.mkdir(parents=True, exist_ok=True)
                for filename in ["adapter_config.json", "adapter_model.safetensors"]:
                    file_path = output_dir_path / filename
                    if file_path.exists():
                        file_path.rename(adapter_dir / filename)
                    else:
                        logger.warning(
                            f"{filename} not found in {output_dir} when "
                            "attempting to delete during model saving."
                        )

                model = cast(peft.LoraModel, self._hf_trainer.model)
                merged_model = model.merge_and_unload(progressbar=True, safe_merge=True)
                merged_model = cast(transformers.PreTrainedModel, merged_model)
                merged_model.save_pretrained(output_dir)
            elif config.peft.peft_save_mode == PeftSaveMode.ADAPTER_ONLY:
                # Save the LoRA adapter (doesn't include the base model).
                self._hf_trainer.save_model(output_dir)
            elif config.peft.peft_save_mode == PeftSaveMode.ADAPTER_AND_BASE_MODEL:
                self._hf_trainer.save_model(output_dir)
                # Saving the base model requires a separate call.
                model = cast(
                    transformers.PreTrainedModel, self._hf_trainer.model.base_model
                )
                model.save_pretrained(output_dir)
            else:
                raise ValueError(
                    f"Unsupported PEFT save mode: {config.peft.peft_save_mode}"
                )
        logger.info(f"Model has been saved at {output_dir}")

        if self._processor is not None:
            self._processor.save_config(output_dir)
            logger.info(f"Processor config has been saved at {output_dir}")

    def _save_fsdp_model(self, config: TrainingConfig, final: bool = True) -> None:
        """Saves the model's weights to the specified output directory.

        For FSDP, all ranks should call into this function
        """
        if final:
            # For the final checkpoint, we need to save the FULL_STATE_DICT instead of
            # the default STATE_DICT.
            if (
                self._hf_trainer.is_fsdp_enabled
                and self._hf_trainer.accelerator.state.fsdp_plugin is not None
            ):
                logger.info("Saving FULL_STATE_DICT for final model checkpoint.")
                self._hf_trainer.accelerator.state.fsdp_plugin.set_state_dict_type(
                    "FULL_STATE_DICT"
                )

        output_dir = config.training.output_dir
        self._hf_trainer.save_model(output_dir)
        logger.info(f"Model has been saved at {output_dir}")

        if self._processor is not None:
            self._processor.save_config(output_dir)
            logger.info(f"Processor config has been saved at {output_dir}")
