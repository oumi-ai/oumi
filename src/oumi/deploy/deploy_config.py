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

"""Typed configuration for ``oumi deploy up`` YAML files."""

import logging
import warnings
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from oumi.deploy.base_client import (
    AutoscalingConfig,
    DeploymentProvider,
    HardwareConfig,
    ModelType,
)
from oumi.deploy.fireworks_client import FIREWORKS_ACCELERATORS

logger = logging.getLogger(__name__)

_DEFAULT_ACCELERATOR = "nvidia_a100_80gb"
_DEFAULT_GPU_COUNT = 1
_DEFAULT_MIN_REPLICAS = 1
_DEFAULT_MAX_REPLICAS = 1
_DEFAULT_MODEL_NAME = "deployed-model"
_DEFAULT_MODEL_TYPE = "full"

_KNOWN_TOP_LEVEL_KEYS = frozenset(
    {
        "model_source",
        "provider",
        "model_name",
        "model_type",
        "base_model",
        "hardware",
        "autoscaling",
        "test_prompts",
    }
)

_KNOWN_HARDWARE_KEYS = frozenset({"accelerator", "count"})
_KNOWN_AUTOSCALING_KEYS = frozenset({"min_replicas", "max_replicas"})

_PROVIDER_ACCELERATORS: dict[str, frozenset[str]] = {
    DeploymentProvider.FIREWORKS.value: frozenset(FIREWORKS_ACCELERATORS.keys()),
}


class DeploymentConfig(BaseModel):
    """Typed configuration for ``oumi deploy up``.

    Mirrors the YAML schema accepted by the ``up`` sub-command and provides
    schema validation, type checking, and cross-field semantic validation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_source: str | None = None
    provider: str | None = None
    model_name: str = _DEFAULT_MODEL_NAME
    model_type: str = _DEFAULT_MODEL_TYPE
    base_model: str | None = None
    hardware: HardwareConfig = Field(
        default_factory=lambda: HardwareConfig(
            accelerator=_DEFAULT_ACCELERATOR, count=_DEFAULT_GPU_COUNT
        )
    )
    autoscaling: AutoscalingConfig = Field(
        default_factory=lambda: AutoscalingConfig(
            min_replicas=_DEFAULT_MIN_REPLICAS, max_replicas=_DEFAULT_MAX_REPLICAS
        )
    )
    test_prompts: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _parse_and_warn(cls, data: Any) -> Any:
        """Warns about unknown keys and coerces nested dicts into config objects."""
        if not isinstance(data, dict):
            return data

        _warn_unknown_keys(data, _KNOWN_TOP_LEVEL_KEYS, prefix="")

        hw_raw = data.get("hardware", {})
        if isinstance(hw_raw, dict):
            _warn_unknown_keys(hw_raw, _KNOWN_HARDWARE_KEYS, prefix="hardware.")
            data = {
                **data,
                "hardware": HardwareConfig(
                    accelerator=hw_raw.get("accelerator", _DEFAULT_ACCELERATOR),
                    count=hw_raw.get("count", _DEFAULT_GPU_COUNT),
                ),
            }
        elif hw_raw and not isinstance(hw_raw, HardwareConfig):
            raise ValueError(
                f"'hardware' must be a YAML mapping (dict), "
                f"got {type(hw_raw).__name__}: {hw_raw!r}"
            )

        asc_raw = data.get("autoscaling", {})
        if isinstance(asc_raw, dict):
            _warn_unknown_keys(asc_raw, _KNOWN_AUTOSCALING_KEYS, prefix="autoscaling.")
            data = {
                **data,
                "autoscaling": AutoscalingConfig(
                    min_replicas=asc_raw.get("min_replicas", _DEFAULT_MIN_REPLICAS),
                    max_replicas=asc_raw.get("max_replicas", _DEFAULT_MAX_REPLICAS),
                ),
            }
        elif asc_raw and not isinstance(asc_raw, AutoscalingConfig):
            raise ValueError(
                f"'autoscaling' must be a YAML mapping (dict), "
                f"got {type(asc_raw).__name__}: {asc_raw!r}"
            )

        return data

    # -- Constructors ---------------------------------------------------------

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "DeploymentConfig":
        """Loads and validates a deployment config from a YAML file.

        Args:
            config_path: Path to the YAML file.

        Returns:
            A validated ``DeploymentConfig`` instance.

        Raises:
            FileNotFoundError: If *config_path* does not exist.
            ValueError: If the YAML is malformed or not a mapping.
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path) as f:
            try:
                raw = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                raise ValueError(f"Invalid YAML in {config_path}: {exc}") from exc

        if raw is None:
            raise ValueError(f"Config file is empty: {config_path}")
        if not isinstance(raw, dict):
            raise ValueError(
                f"Config file must contain a YAML mapping (dict), "
                f"got {type(raw).__name__}: {config_path}"
            )

        try:
            return cls.model_validate(raw)
        except ValidationError as exc:
            raise ValueError(
                f"Invalid deployment config in {config_path}: {exc}"
            ) from exc

    # -- CLI override helpers -------------------------------------------------

    def apply_cli_overrides(
        self,
        *,
        model_path: str | None = None,
        provider: str | None = None,
        hardware: str | None = None,
    ) -> None:
        """Applies CLI flag overrides on top of values loaded from YAML.

        Only non-``None`` arguments overwrite the corresponding config field.
        """
        if model_path is not None:
            self.model_source = model_path
        if provider is not None:
            self.provider = provider
        if hardware is not None:
            self.hardware = HardwareConfig(
                accelerator=hardware, count=self.hardware.count
            )

    # -- Validation -----------------------------------------------------------

    def finalize_and_validate(self) -> None:
        """Validates the configuration, raising on the first error found.

        Raises:
            ValueError: For any semantic or constraint violation.
        """
        self._validate_required_fields()
        self._validate_provider()
        self._validate_model_type()
        self._validate_hardware()
        self._validate_autoscaling()
        self._validate_adapter_base_model()

    def _validate_required_fields(self) -> None:
        if not self.model_source:
            raise ValueError(
                "model_source is required "
                "(set in config YAML or pass --model-path on the CLI)"
            )
        if not self.provider:
            raise ValueError(
                "provider is required "
                "(set in config YAML or pass --provider on the CLI)"
            )

    @staticmethod
    def _validate_enum_field(
        value: str | None,
        enum_cls: type[Enum],
        field_name: str,
        *,
        label: str = "Invalid",
    ) -> None:
        """Validates that *value* is a member of *enum_cls* (case-insensitive)."""
        if value is None:
            return
        valid = [e.value for e in enum_cls]
        if value.lower() not in valid:
            raise ValueError(
                f"{label} {field_name}: '{value}'. Must be one of: {valid}"
            )

    def _validate_provider(self) -> None:
        self._validate_enum_field(
            self.provider, DeploymentProvider, "provider", label="Unsupported"
        )

    def _validate_model_type(self) -> None:
        self._validate_enum_field(self.model_type, ModelType, "model_type")

    def _validate_hardware(self) -> None:
        if not isinstance(self.hardware.count, int):
            raise ValueError(
                f"hardware.count must be an integer, "
                f"got {type(self.hardware.count).__name__}: {self.hardware.count!r}"
            )
        if self.hardware.count < 1:
            raise ValueError(f"hardware.count must be >= 1, got {self.hardware.count}")

        provider = (self.provider or "").lower()
        supported = _PROVIDER_ACCELERATORS.get(provider)
        if supported is not None:
            accel = self.hardware.accelerator.lower()
            if accel not in supported:
                raise ValueError(
                    f"Unsupported accelerator '{self.hardware.accelerator}' "
                    f"for provider '{provider}'. "
                    f"Supported accelerators: {sorted(supported)}"
                )

    def _validate_autoscaling(self) -> None:
        mn, mx = self.autoscaling.min_replicas, self.autoscaling.max_replicas
        if not isinstance(mn, int) or not isinstance(mx, int):
            raise ValueError(
                "autoscaling.min_replicas and autoscaling.max_replicas must be integers"
            )
        if mn < 0:
            raise ValueError(f"autoscaling.min_replicas must be >= 0, got {mn}")
        if mx < 1:
            raise ValueError(f"autoscaling.max_replicas must be >= 1, got {mx}")
        if mn > mx:
            raise ValueError(
                f"autoscaling.min_replicas ({mn}) must be "
                f"<= autoscaling.max_replicas ({mx})"
            )

    def _validate_adapter_base_model(self) -> None:
        if self.model_type == ModelType.ADAPTER.value and not self.base_model:
            raise ValueError("base_model is required when model_type is 'adapter'")


def _warn_unknown_keys(
    data: dict[str, Any], known: frozenset[str], prefix: str
) -> None:
    unknown = set(data.keys()) - known
    if unknown:
        sorted_keys = sorted(unknown)
        warnings.warn(
            f"Unknown config key(s) at '{prefix}': {sorted_keys}. "
            f"Valid keys: {sorted(known)}",
            UserWarning,
            stacklevel=3,
        )
