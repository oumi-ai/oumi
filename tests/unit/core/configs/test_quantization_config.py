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

"""Unit tests for QuantizationConfig."""

import os
import tempfile

import pytest

from oumi.core.configs import ModelParams
from oumi.core.configs.quantization_config import (
    QuantizationAlgorithm,
    QuantizationConfig,
    QuantizationMethod,
)


def _make_config(**overrides):
    defaults = dict(
        model=ModelParams(model_name="test/model"),
        method=QuantizationMethod.FP8_DYNAMIC,
        output_path="test_output",
    )
    defaults.update(overrides)
    return QuantizationConfig(**defaults)


class TestQuantizationConfigDefaults:

    def test_method_is_required(self):
        with pytest.raises(ValueError, match="Quantization method is required"):
            QuantizationConfig(
                model=ModelParams(model_name="test/model"),
                output_path="test",
            )

    def test_algorithm_defaults_to_auto(self):
        config = _make_config()
        assert config.algorithm == QuantizationAlgorithm.AUTO

    def test_new_fields_have_defaults(self):
        config = _make_config()
        assert config.ignore_layers == ["lm_head"]
        assert config.calibration_dataset == "HuggingFaceH4/ultrachat_200k"
        assert config.calibration_split == "train_sft"
        assert config.calibration_samples == 512
        assert config.max_seq_length == 2048
        assert config.group_size == 128
        assert config.dampening_frac == 0.1
        assert config.save_compressed is True
        assert config.output_format == "safetensors"
        assert config.output_path == "test_output"


class TestQuantizationConfigMethodValidation:

    @pytest.mark.parametrize(
        "method_str",
        ["fp8_dynamic", "fp8_block", "w4a16", "w4a16_asym", "w8a16",
         "bnb_4bit", "bnb_8bit"],
    )
    def test_valid_methods_from_string(self, method_str):
        config = _make_config(method=method_str)
        assert isinstance(config.method, QuantizationMethod)
        assert config.method.value == method_str

    @pytest.mark.parametrize(
        "method",
        list(QuantizationMethod),
    )
    def test_valid_methods_from_enum(self, method):
        config = _make_config(method=method)
        assert config.method == method

    def test_invalid_method_rejected(self):
        with pytest.raises(ValueError, match="Unsupported quantization method"):
            _make_config(method="nonexistent")

    def test_old_awq_names_rejected(self):
        for old_name in ("awq_q4_0", "awq_q4_1", "awq_q8_0"):
            with pytest.raises(ValueError, match="Unsupported quantization method"):
                _make_config(method=old_name)


class TestQuantizationConfigAlgorithm:

    @pytest.mark.parametrize(
        "algo_str,expected",
        [
            ("auto", QuantizationAlgorithm.AUTO),
            ("rtn", QuantizationAlgorithm.RTN),
            ("gptq", QuantizationAlgorithm.GPTQ),
            ("awq", QuantizationAlgorithm.AWQ),
            ("bnb", QuantizationAlgorithm.BNB),
        ],
    )
    def test_valid_algorithm_from_string(self, algo_str, expected):
        config = _make_config(algorithm=algo_str)
        assert config.algorithm == expected

    def test_invalid_algorithm_rejected(self):
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            _make_config(algorithm="invalid_algo")


class TestQuantizationConfigOutputFormat:

    def test_safetensors_accepted(self):
        config = _make_config(output_format="safetensors")
        assert config.output_format == "safetensors"

    def test_invalid_format_rejected(self):
        with pytest.raises(ValueError, match="Unsupported output format"):
            _make_config(output_format="pytorch")


class TestQuantizationConfigYaml:
    """YAML round-trip tests (UC-11, UC-12)."""

    def test_from_yaml_new_config(self):
        config = _make_config(
            algorithm="gptq",
            calibration_dataset="test/dataset",
            calibration_samples=256,
            max_seq_length=1024,
            ignore_layers=["lm_head", "re:.*gate$"],
            dampening_frac=0.05,
            save_compressed=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "quant.yaml")
            config.to_yaml(path)
            loaded = QuantizationConfig.from_yaml(path)

        assert loaded.method == QuantizationMethod.FP8_DYNAMIC
        assert loaded.algorithm == QuantizationAlgorithm.GPTQ
        assert loaded.calibration_dataset == "test/dataset"
        assert loaded.calibration_samples == 256
        assert loaded.max_seq_length == 1024
        assert loaded.ignore_layers == ["lm_head", "re:.*gate$"]
        assert loaded.dampening_frac == 0.05
        assert loaded.save_compressed is False

    def test_from_yaml_minimal_config(self):
        config = QuantizationConfig(
            model=ModelParams(model_name="meta-llama/Llama-3.1-8B-Instruct"),
            method="w4a16",
            output_path="llama3-w4a16",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "quant.yaml")
            config.to_yaml(path)
            loaded = QuantizationConfig.from_yaml(path)

        assert loaded.method == QuantizationMethod.W4A16
        assert loaded.model.model_name == "meta-llama/Llama-3.1-8B-Instruct"
        assert loaded.output_path == "llama3-w4a16"
        assert loaded.algorithm == QuantizationAlgorithm.AUTO
        assert loaded.save_compressed is True
