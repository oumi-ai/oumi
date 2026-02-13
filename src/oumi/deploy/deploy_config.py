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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from oumi.deploy.base_client import (
    AutoscalingConfig,
    DeploymentProvider,
    HardwareConfig,
    ModelType,
)

logger = logging.getLogger(__name__)

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


@dataclass
class DeploymentConfig:
    """Typed configuration for ``oumi deploy up``.

    Mirrors the YAML schema accepted by the ``up`` sub-command and provides
    schema validation, type checking, and cross-field semantic validation.
    """

    model_source: str | None = None
    provider: str | None = None
    model_name: str = "deployed-model"
    model_type: str = "full"
    base_model: str | None = None
    hardware: HardwareConfig = field(
        default_factory=lambda: HardwareConfig(accelerator="nvidia_a100_80gb", count=1)
    )
    autoscaling: AutoscalingConfig = field(
        default_factory=lambda: AutoscalingConfig(min_replicas=1, max_replicas=1)
    )
    test_prompts: list[str] = field(default_factory=list)

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
                raise ValueError(
                    f"Invalid YAML in {config_path}: {exc}"
                ) from exc

        if raw is None:
            raise ValueError(f"Config file is empty: {config_path}")
        if not isinstance(raw, dict):
            raise ValueError(
                f"Config file must contain a YAML mapping (dict), "
                f"got {type(raw).__name__}: {config_path}"
            )

        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> "DeploymentConfig":
        """Builds a ``DeploymentConfig`` from a raw dict (parsed YAML)."""
        _warn_unknown_keys(data, _KNOWN_TOP_LEVEL_KEYS, prefix="")

        hw_raw = data.get("hardware", {})
        if isinstance(hw_raw, dict):
            _warn_unknown_keys(hw_raw, _KNOWN_HARDWARE_KEYS, prefix="hardware.")
            hw = HardwareConfig(
                accelerator=hw_raw.get("accelerator", "nvidia_a100_80gb"),
                count=hw_raw.get("count", 1),
            )
        else:
            hw = HardwareConfig(accelerator="nvidia_a100_80gb", count=1)

        asc_raw = data.get("autoscaling", {})
        if isinstance(asc_raw, dict):
            _warn_unknown_keys(
                asc_raw, _KNOWN_AUTOSCALING_KEYS, prefix="autoscaling."
            )
            asc = AutoscalingConfig(
                min_replicas=asc_raw.get("min_replicas", 1),
                max_replicas=asc_raw.get("max_replicas", 1),
            )
        else:
            asc = AutoscalingConfig(min_replicas=1, max_replicas=1)

        return cls(
            model_source=data.get("model_source"),
            provider=data.get("provider"),
            model_name=data.get("model_name", "deployed-model"),
            model_type=data.get("model_type", "full"),
            base_model=data.get("base_model"),
            hardware=hw,
            autoscaling=asc,
            test_prompts=data.get("test_prompts", []),
        )

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

    def _validate_provider(self) -> None:
        valid_providers = [p.value for p in DeploymentProvider]
        if self.provider and self.provider.lower() not in valid_providers:
            raise ValueError(
                f"Unsupported provider: '{self.provider}'. "
                f"Supported providers: {valid_providers}"
            )

    def _validate_model_type(self) -> None:
        valid_types = [t.value for t in ModelType]
        if self.model_type not in valid_types:
            raise ValueError(
                f"Invalid model_type: '{self.model_type}'. "
                f"Must be one of: {valid_types}"
            )

    def _validate_hardware(self) -> None:
        if not isinstance(self.hardware.count, int):
            raise ValueError(
                f"hardware.count must be an integer, "
                f"got {type(self.hardware.count).__name__}: {self.hardware.count!r}"
            )
        if self.hardware.count < 1:
            raise ValueError(
                f"hardware.count must be >= 1, got {self.hardware.count}"
            )

    def _validate_autoscaling(self) -> None:
        mn, mx = self.autoscaling.min_replicas, self.autoscaling.max_replicas
        if not isinstance(mn, int) or not isinstance(mx, int):
            raise ValueError(
                "autoscaling.min_replicas and autoscaling.max_replicas "
                "must be integers"
            )
        if mn < 0:
            raise ValueError(
                f"autoscaling.min_replicas must be >= 0, got {mn}"
            )
        if mx < 1:
            raise ValueError(
                f"autoscaling.max_replicas must be >= 1, got {mx}"
            )
        if mn > mx:
            raise ValueError(
                f"autoscaling.min_replicas ({mn}) must be "
                f"<= autoscaling.max_replicas ({mx})"
            )

    def _validate_adapter_base_model(self) -> None:
        if self.model_type == ModelType.ADAPTER.value and not self.base_model:
            raise ValueError(
                "base_model is required when model_type is 'adapter'"
            )


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
