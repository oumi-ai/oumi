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

"""Unit tests for DeploymentConfig validation (Phase 0c).

Covers: valid loading, required fields, type/value validation,
cross-field semantics, unknown-key warnings, CLI overrides,
and malformed YAML handling.
"""

import textwrap
import warnings

import pytest

from oumi.deploy.deploy_config import DeploymentConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_yaml(tmp_path, content: str) -> str:
    """Write *content* to a temp YAML file and return the path."""
    p = tmp_path / "deploy_config.yaml"
    p.write_text(textwrap.dedent(content))
    return str(p)


MINIMAL_VALID_YAML = """\
model_source: Qwen/Qwen3-4B
provider: fireworks
"""

FULL_VALID_YAML = """\
model_source: s3://bucket/model/
provider: fireworks
model_name: my-model-v1
model_type: full
hardware:
  accelerator: nvidia_h100_80gb
  count: 2
autoscaling:
  min_replicas: 1
  max_replicas: 4
test_prompts:
  - "Hello, how are you?"
  - "Write fibonacci in Python."
"""

ADAPTER_VALID_YAML = """\
model_source: /data/adapters/lora/
provider: fireworks
model_name: my-lora
model_type: adapter
base_model: meta-llama/Llama-2-7b-hf
hardware:
  accelerator: nvidia_a100_80gb
  count: 1
autoscaling:
  min_replicas: 1
  max_replicas: 2
"""


# ===================================================================
# 19.6.1 Valid Config Loading
# ===================================================================


class TestValidConfigLoading:
    """U-CFG-1 through U-CFG-4."""

    def test_load_valid_full_config(self, tmp_path):
        path = _write_yaml(tmp_path, FULL_VALID_YAML)
        cfg = DeploymentConfig.from_yaml(path)
        cfg.finalize_and_validate()

        assert cfg.model_source == "s3://bucket/model/"
        assert cfg.provider == "fireworks"
        assert cfg.model_name == "my-model-v1"
        assert cfg.model_type == "full"
        assert cfg.hardware.accelerator == "nvidia_h100_80gb"
        assert cfg.hardware.count == 2
        assert cfg.autoscaling.min_replicas == 1
        assert cfg.autoscaling.max_replicas == 4
        assert cfg.test_prompts == [
            "Hello, how are you?",
            "Write fibonacci in Python.",
        ]

    def test_load_valid_adapter_config(self, tmp_path):
        path = _write_yaml(tmp_path, ADAPTER_VALID_YAML)
        cfg = DeploymentConfig.from_yaml(path)
        cfg.finalize_and_validate()

        assert cfg.model_type == "adapter"
        assert cfg.base_model == "meta-llama/Llama-2-7b-hf"

    def test_config_defaults(self, tmp_path):
        path = _write_yaml(tmp_path, MINIMAL_VALID_YAML)
        cfg = DeploymentConfig.from_yaml(path)
        cfg.finalize_and_validate()

        assert cfg.model_name == "deployed-model"
        assert cfg.model_type == "full"
        assert cfg.base_model is None
        assert cfg.hardware.accelerator == "nvidia_a100_80gb"
        assert cfg.hardware.count == 1
        assert cfg.autoscaling.min_replicas == 1
        assert cfg.autoscaling.max_replicas == 1
        assert cfg.test_prompts == []

    def test_config_roundtrip(self):
        original = DeploymentConfig(
            model_source="Qwen/Qwen3-4B",
            provider="fireworks",
            model_name="test-model",
            model_type="full",
        )
        original.finalize_and_validate()

        rebuilt = DeploymentConfig(
            model_source=original.model_source,
            provider=original.provider,
            model_name=original.model_name,
            model_type=original.model_type,
            hardware=original.hardware,
            autoscaling=original.autoscaling,
            test_prompts=original.test_prompts,
        )
        rebuilt.finalize_and_validate()

        assert original.model_source == rebuilt.model_source
        assert original.provider == rebuilt.provider
        assert original.model_name == rebuilt.model_name


# ===================================================================
# 19.6.2 Required Field Validation
# ===================================================================


class TestRequiredFields:
    """U-CFG-5 through U-CFG-8."""

    def test_config_missing_model_source(self, tmp_path):
        path = _write_yaml(tmp_path, "provider: fireworks\n")
        cfg = DeploymentConfig.from_yaml(path)
        with pytest.raises(ValueError, match="model_source"):
            cfg.finalize_and_validate()

    def test_config_missing_provider(self, tmp_path):
        path = _write_yaml(tmp_path, "model_source: Qwen/Qwen3-4B\n")
        cfg = DeploymentConfig.from_yaml(path)
        with pytest.raises(ValueError, match="provider"):
            cfg.finalize_and_validate()

    def test_config_empty_model_source(self, tmp_path):
        yaml_content = 'model_source: ""\nprovider: fireworks\n'
        path = _write_yaml(tmp_path, yaml_content)
        cfg = DeploymentConfig.from_yaml(path)
        with pytest.raises(ValueError, match="model_source"):
            cfg.finalize_and_validate()

    def test_config_empty_provider(self, tmp_path):
        yaml_content = 'model_source: Qwen/Qwen3-4B\nprovider: ""\n'
        path = _write_yaml(tmp_path, yaml_content)
        cfg = DeploymentConfig.from_yaml(path)
        with pytest.raises(ValueError, match="provider"):
            cfg.finalize_and_validate()


# ===================================================================
# 19.6.3 Type and Value Validation
# ===================================================================


class TestTypeAndValueValidation:
    """U-CFG-9 through U-CFG-16."""

    def test_config_invalid_model_type(self, tmp_path):
        yaml_content = "model_source: x\nprovider: fireworks\nmodel_type: quantized\n"
        path = _write_yaml(tmp_path, yaml_content)
        cfg = DeploymentConfig.from_yaml(path)
        with pytest.raises(ValueError, match="model_type.*quantized"):
            cfg.finalize_and_validate()

    def test_config_invalid_provider(self, tmp_path):
        yaml_content = "model_source: x\nprovider: unsupported\n"
        path = _write_yaml(tmp_path, yaml_content)
        cfg = DeploymentConfig.from_yaml(path)
        with pytest.raises(ValueError, match="Unsupported provider.*unsupported"):
            cfg.finalize_and_validate()

    def test_config_hardware_count_zero(self, tmp_path):
        yaml_content = (
            "model_source: x\nprovider: fireworks\n"
            "hardware:\n  accelerator: nvidia_a100_80gb\n  count: 0\n"
        )
        path = _write_yaml(tmp_path, yaml_content)
        cfg = DeploymentConfig.from_yaml(path)
        with pytest.raises(ValueError, match="hardware.count.*>= 1"):
            cfg.finalize_and_validate()

    def test_config_hardware_count_negative(self, tmp_path):
        yaml_content = (
            "model_source: x\nprovider: fireworks\n"
            "hardware:\n  accelerator: nvidia_a100_80gb\n  count: -1\n"
        )
        path = _write_yaml(tmp_path, yaml_content)
        cfg = DeploymentConfig.from_yaml(path)
        with pytest.raises(ValueError, match="hardware.count.*>= 1"):
            cfg.finalize_and_validate()

    def test_config_hardware_count_not_int(self):
        cfg = DeploymentConfig(
            model_source="x",
            provider="fireworks",
        )
        cfg.hardware.count = "two"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="hardware.count.*integer"):
            cfg.finalize_and_validate()

    def test_config_invalid_accelerator_for_provider(self, tmp_path):
        yaml_content = (
            "model_source: x\nprovider: fireworks\n"
            "hardware:\n  accelerator: nvidia_z999_1tb\n  count: 1\n"
        )
        path = _write_yaml(tmp_path, yaml_content)
        cfg = DeploymentConfig.from_yaml(path)
        with pytest.raises(
            ValueError, match="Unsupported accelerator.*nvidia_z999_1tb"
        ):
            cfg.finalize_and_validate()

    def test_config_valid_accelerator_for_provider(self, tmp_path):
        for accel in [
            "nvidia_a100_80gb",
            "nvidia_a100_40gb",
            "nvidia_a10g_24gb",
            "nvidia_h100_80gb",
            "nvidia_h200_141gb",
            "nvidia_b200_180gb",
            "nvidia_l4_24gb",
            "amd_mi300x_192gb",
            "amd_mi325x_256gb",
            "amd_mi350x_288gb",
        ]:
            yaml_content = (
                f"model_source: x\nprovider: fireworks\n"
                f"hardware:\n  accelerator: {accel}\n  count: 1\n"
            )
            path = _write_yaml(tmp_path, yaml_content)
            cfg = DeploymentConfig.from_yaml(path)
            cfg.finalize_and_validate()

    def test_config_invalid_accelerator_message_lists_supported(self, tmp_path):
        yaml_content = (
            "model_source: x\nprovider: fireworks\n"
            "hardware:\n  accelerator: bogus_gpu\n  count: 1\n"
        )
        path = _write_yaml(tmp_path, yaml_content)
        cfg = DeploymentConfig.from_yaml(path)
        with pytest.raises(ValueError, match="Supported accelerators"):
            cfg.finalize_and_validate()

    def test_config_min_replicas_greater_than_max(self, tmp_path):
        yaml_content = (
            "model_source: x\nprovider: fireworks\n"
            "autoscaling:\n  min_replicas: 5\n  max_replicas: 2\n"
        )
        path = _write_yaml(tmp_path, yaml_content)
        cfg = DeploymentConfig.from_yaml(path)
        with pytest.raises(ValueError, match="min_replicas.*<=.*max_replicas"):
            cfg.finalize_and_validate()

    def test_config_min_replicas_negative(self, tmp_path):
        yaml_content = (
            "model_source: x\nprovider: fireworks\n"
            "autoscaling:\n  min_replicas: -1\n  max_replicas: 1\n"
        )
        path = _write_yaml(tmp_path, yaml_content)
        cfg = DeploymentConfig.from_yaml(path)
        with pytest.raises(ValueError, match="min_replicas.*>= 0"):
            cfg.finalize_and_validate()

    def test_config_max_replicas_zero(self, tmp_path):
        yaml_content = (
            "model_source: x\nprovider: fireworks\n"
            "autoscaling:\n  min_replicas: 0\n  max_replicas: 0\n"
        )
        path = _write_yaml(tmp_path, yaml_content)
        cfg = DeploymentConfig.from_yaml(path)
        with pytest.raises(ValueError, match="max_replicas.*>= 1"):
            cfg.finalize_and_validate()


# ===================================================================
# 19.6.4 Cross-Field Semantic Validation
# ===================================================================


class TestCrossFieldValidation:
    """U-CFG-17 through U-CFG-19."""

    def test_config_adapter_requires_base_model(self, tmp_path):
        yaml_content = (
            "model_source: /data/adapter/\nprovider: fireworks\nmodel_type: adapter\n"
        )
        path = _write_yaml(tmp_path, yaml_content)
        cfg = DeploymentConfig.from_yaml(path)
        with pytest.raises(ValueError, match="base_model.*required"):
            cfg.finalize_and_validate()

    def test_config_adapter_with_base_model_ok(self, tmp_path):
        path = _write_yaml(tmp_path, ADAPTER_VALID_YAML)
        cfg = DeploymentConfig.from_yaml(path)
        cfg.finalize_and_validate()
        assert cfg.model_type == "adapter"
        assert cfg.base_model == "meta-llama/Llama-2-7b-hf"

    def test_config_full_ignores_base_model(self, tmp_path):
        yaml_content = (
            "model_source: x\n"
            "provider: fireworks\n"
            "model_type: full\n"
            "base_model: meta-llama/Llama-2-7b-hf\n"
        )
        path = _write_yaml(tmp_path, yaml_content)
        cfg = DeploymentConfig.from_yaml(path)
        cfg.finalize_and_validate()
        assert cfg.base_model == "meta-llama/Llama-2-7b-hf"


# ===================================================================
# 19.6.5 Unknown Keys Detection
# ===================================================================


class TestUnknownKeys:
    """U-CFG-20 and U-CFG-21."""

    def test_config_unknown_top_level_key_warning(self, tmp_path):
        yaml_content = "model_source: x\nprovider: fireworks\nmodell_source: typo\n"
        path = _write_yaml(tmp_path, yaml_content)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DeploymentConfig.from_yaml(path)

        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) >= 1
        assert "modell_source" in str(user_warnings[0].message)

    def test_config_unknown_nested_key_warning(self, tmp_path):
        yaml_content = (
            "model_source: x\nprovider: fireworks\nhardware:\n  acelerator: typo\n"
        )
        path = _write_yaml(tmp_path, yaml_content)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DeploymentConfig.from_yaml(path)

        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) >= 1
        assert "acelerator" in str(user_warnings[0].message)


# ===================================================================
# 19.6.6 CLI Override Application
# ===================================================================


class TestCLIOverrides:
    """U-CFG-22 through U-CFG-25."""

    def test_cli_overrides_model_path(self, tmp_path):
        path = _write_yaml(tmp_path, FULL_VALID_YAML)
        cfg = DeploymentConfig.from_yaml(path)
        cfg.apply_cli_overrides(model_path="overridden/path")
        cfg.finalize_and_validate()
        assert cfg.model_source == "overridden/path"

    def test_cli_overrides_provider(self, tmp_path):
        path = _write_yaml(tmp_path, FULL_VALID_YAML)
        cfg = DeploymentConfig.from_yaml(path)
        cfg.apply_cli_overrides(provider="fireworks")
        cfg.finalize_and_validate()
        assert cfg.provider == "fireworks"

    def test_cli_overrides_hardware(self, tmp_path):
        path = _write_yaml(tmp_path, FULL_VALID_YAML)
        cfg = DeploymentConfig.from_yaml(path)
        cfg.apply_cli_overrides(hardware="nvidia_a100_80gb")
        cfg.finalize_and_validate()
        assert cfg.hardware.accelerator == "nvidia_a100_80gb"
        assert cfg.hardware.count == 2  # count preserved from YAML

    def test_cli_provides_missing_required(self, tmp_path):
        path = _write_yaml(tmp_path, "provider: fireworks\n")
        cfg = DeploymentConfig.from_yaml(path)
        cfg.apply_cli_overrides(model_path="cli-provided/model")
        cfg.finalize_and_validate()
        assert cfg.model_source == "cli-provided/model"


# ===================================================================
# 19.6.7 Malformed YAML
# ===================================================================


class TestMalformedYAML:
    """U-CFG-26 through U-CFG-29."""

    def test_config_malformed_yaml(self, tmp_path):
        path = _write_yaml(tmp_path, "key: [unclosed\n")
        with pytest.raises(ValueError, match="Invalid YAML"):
            DeploymentConfig.from_yaml(path)

    def test_config_yaml_not_a_dict(self, tmp_path):
        path = _write_yaml(tmp_path, "- item1\n- item2\n")
        with pytest.raises(ValueError, match="YAML mapping"):
            DeploymentConfig.from_yaml(path)

    def test_config_empty_yaml(self, tmp_path):
        path = _write_yaml(tmp_path, "")
        with pytest.raises(ValueError, match="empty"):
            DeploymentConfig.from_yaml(path)

    def test_config_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            DeploymentConfig.from_yaml("/nonexistent/path/config.yaml")
