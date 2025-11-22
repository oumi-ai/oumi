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

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Optional, Union

import safetensors.torch
import torch
import torch.nn as nn

from oumi.utils.logging import logger


class BaseModel(nn.Module, ABC):
    def __init__(self, **kwargs):
        """Initializes a new instance of the model class, and builds the model scaffold.

        Note:
            - All model layers should be registered in this method.
            - The weights should not be loaded or moved to devices at this point.

        Args:
            **kwargs: should contain all the parameters needed
                to build the model scaffold.
        """
        super().__init__()
        # Store initialization kwargs for reproducible reload
        self._init_kwargs = kwargs

    @abstractmethod
    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Performs the forward pass of the model.

        Optionally computes the loss if the necessary keyword arguments are provided.

        Args:
            **kwargs: should contain all the parameters needed to perform the forward
                pass, and compute the loss if needed.

        Returns:
            A dictionary containing the output tensors.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def criterion(self) -> Callable:
        """Returns the criterion function used for model training.

        Returns:
            A callable object representing the criterion function.
        """
        raise NotImplementedError

    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        *,
        save_config: bool = True,
        weights_filename: str = "model.safetensors",
        config_filename: str = "config.json",
    ) -> None:
        """Saves model weights and initialization config to a directory.

        This method saves the model in a format compatible with `from_pretrained()`,
        allowing the model to be reloaded later for inference or further training.

        Args:
            save_directory: Directory where the model will be saved.
                Will be created if it doesn't exist.
            save_config: If True, saves initialization kwargs as JSON config.
                Defaults to True.
            weights_filename: Name of the weights file. Defaults to "model.safetensors".
            config_filename: Name of the config file. Defaults to "config.json".

        Raises:
            OSError: If the directory cannot be created or files cannot be written.

        Example:
            >>> model = MyCustomModel(hidden_dim=128, num_layers=4)
            >>> model.save_pretrained("./my_model_checkpoint")
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # Saving weights using safetensors
        weights_path = save_directory / weights_filename
        safetensors.torch.save_model(model=self, filename=str(weights_path))
        logger.info(f"Model weights saved to {weights_path}")

        # Saving initialization config if available
        if save_config and hasattr(self, "_init_kwargs"):
            config_path = save_directory / config_filename
            config_data = {
                "model_type": self.__class__.__name__,
                "init_kwargs": self._init_kwargs,
                "oumi_version": self._get_oumi_version(),
            }
            with config_path.open("w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=2)
            logger.info(f"Model config saved to {config_path}")

    @classmethod
    def from_pretrained(
        cls,
        load_directory: Union[str, Path],
        *,
        map_location: Optional[Union[str, torch.device]] = None,
        strict: bool = True,
        weights_filename: str = "model.safetensors",
        config_filename: str = "config.json",
        override_kwargs: Optional[dict[str, Any]] = None,
    ) -> "BaseModel":
        """Loads a model from a directory saved with `save_pretrained()`.

        This classmethod instantiates a model and loads pretrained weights from disk.
        It reads both the model configuration and weights, ensuring compatibility.

        Args:
            load_directory: Directory containing the saved model files.
            map_location: Device to load tensors to (e.g., "cpu", "cuda:0").
                If None, loads to CPU by default.
            strict: If True, requires exact match between state_dict keys.
                Defaults to True for safety.
            weights_filename: Expected name of weights file.
                Defaults to "model.safetensors".
            config_filename: Expected name of config file.
                Defaults to "config.json".
            override_kwargs: Dict of initialization kwargs to override those in config.
                Useful for modifying model architecture during loading.

        Returns:
            An instance of the model class with loaded weights.

        Raises:
            FileNotFoundError: If weights file doesn't exist.
            RuntimeError: If there are missing or unexpected keys when loading
                state_dict.
            ValueError: If model type in config doesn't match the class.

        Example:
            >>> model = MyCustomModel.from_pretrained("./my_model_checkpoint")
            >>> # Or with overrides
            >>> model = MyCustomModel.from_pretrained(
            ...     "./my_model_checkpoint",
            ...     override_kwargs={"dropout": 0.0}
            ... )
        """
        load_directory = Path(load_directory)
        weights_path = load_directory / weights_filename

        # Checking if weights file exists
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Pretrained weights file not found: {weights_path}. "
                f"Expected to find '{weights_filename}' in {load_directory}. "
                "Ensure the model was saved with save_pretrained()."
            )

        # Loading config if available
        init_kwargs: dict[str, Any] = {}
        config_path = load_directory / config_filename
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as f:
                config_data = json.load(f)

            # Validating model type matches
            if "model_type" in config_data:
                if config_data["model_type"] != cls.__name__:
                    logger.warning(
                        f"Model type mismatch: config has "
                        f"'{config_data['model_type']}' but loading into "
                        f"'{cls.__name__}'. This may cause issues."
                    )

            # Extracting init kwargs
            if "init_kwargs" in config_data:
                init_kwargs = config_data["init_kwargs"]

            # Logging version info if available
            if "oumi_version" in config_data:
                logger.info(
                    f"Loading model saved with Oumi version "
                    f"{config_data['oumi_version']}"
                )
        else:
            logger.warning(
                f"Config file not found at {config_path}. "
                "Model will be instantiated with override_kwargs only. "
                "Ensure you provide all necessary initialization parameters."
            )

        # Applying overrides
        if override_kwargs:
            logger.info(f"Applying initialization kwargs overrides: {override_kwargs}")
            init_kwargs.update(override_kwargs)

        # Instantiating model
        try:
            model = cls(**init_kwargs)
        except TypeError as e:
            raise TypeError(
                f"Failed to instantiate {cls.__name__} with kwargs: {init_kwargs}. "
                f"Error: {e}. If the config is missing required parameters, "
                "provide them via override_kwargs."
            ) from e

        # Loading state dict
        if map_location is None:
            map_location = "cpu"

        try:
            state_dict = safetensors.torch.load_file(
                str(weights_path), device=str(map_location)
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load weights from {weights_path}. Error: {e}"
            ) from e

        # Loading state dict into model
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)

        if strict:
            if missing_keys:
                raise RuntimeError(
                    f"Missing keys when loading pretrained custom model: "
                    f"{missing_keys}. This indicates the saved weights don't "
                    f"match the model architecture. Set strict=False to load "
                    f"anyway (not recommended)."
                )
            if unexpected_keys:
                raise RuntimeError(
                    f"Unexpected keys when loading pretrained custom model: "
                    f"{unexpected_keys}. This indicates the saved weights "
                    f"contain extra parameters. Set strict=False to load "
                    f"anyway (not recommended)."
                )
        else:
            if missing_keys:
                logger.warning(f"Missing keys in state_dict: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in state_dict: {unexpected_keys}")

        logger.info(
            f"Successfully loaded pretrained {cls.__name__} from {load_directory}"
        )
        return model

    @staticmethod
    def _get_oumi_version() -> str:
        """Gets the current Oumi version string."""
        try:
            import oumi

            return getattr(oumi, "__version__", "unknown")
        except Exception:
            return "unknown"
