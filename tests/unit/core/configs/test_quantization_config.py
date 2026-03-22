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
from typing import Any

import pytest

from oumi.core.configs import ModelParams
from oumi.core.configs.quantization_config import (
    QuantizationAlgorithm,
    QuantizationBackend,
    QuantizationConfig,
    QuantizationScheme,
)
from oumi.exceptions import OumiConfigError


def _make_config(**overrides: Any) -> QuantizationConfig:
    defaults = {
        "model": ModelParams(model_name="test/model"),
        "scheme": QuantizationScheme.FP8_DYNAMIC,
        "output_path": "test_output",
    }
    defaults.update(overrides)
    return QuantizationConfig(**defaults)


class TestQuantizationConfigDefaults:
    def test_scheme_is_required(self):
        with pytest.raises(OumiConfigError, match="Quantization scheme is required"):
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


class TestQuantizationConfigBackendInference:
    def test_llmc_scheme_infers_llm_compressor_backend(self):
        config = _make_config(scheme=QuantizationScheme.FP8_DYNAMIC)
        assert config.backend == QuantizationBackend.LLM_COMPRESSOR

    def test_bnb_scheme_infers_bnb_backend(self):
        config = _make_config(scheme=QuantizationScheme.BNB_NF4)
        assert config.backend == QuantizationBackend.BNB

    @pytest.mark.parametrize(
        "scheme,expected_backend",
        [
            (QuantizationScheme.FP8_DYNAMIC, QuantizationBackend.LLM_COMPRESSOR),
            (QuantizationScheme.FP8_BLOCK, QuantizationBackend.LLM_COMPRESSOR),
            (QuantizationScheme.W4A16, QuantizationBackend.LLM_COMPRESSOR),
            (QuantizationScheme.W4A16_ASYM, QuantizationBackend.LLM_COMPRESSOR),
            (QuantizationScheme.W8A16, QuantizationBackend.LLM_COMPRESSOR),
            (QuantizationScheme.BNB_NF4, QuantizationBackend.BNB),
            (QuantizationScheme.BNB_FP4, QuantizationBackend.BNB),
            (QuantizationScheme.BNB_INT8, QuantizationBackend.BNB),
        ],
    )
    def test_all_schemes_infer_correct_backend(self, scheme, expected_backend):
        config = _make_config(scheme=scheme)
        assert config.backend == expected_backend


class TestQuantizationConfigSchemeValidation:
    @pytest.mark.parametrize(
        "scheme_str",
        ["fp8_dynamic", "fp8_block", "w4a16", "w4a16_asym", "w8a16"],
    )
    def test_valid_llmc_schemes_from_string(self, scheme_str):
        config = _make_config(scheme=scheme_str)
        assert isinstance(config.scheme, QuantizationScheme)
        assert config.scheme.value == scheme_str

    @pytest.mark.parametrize(
        "scheme_str",
        ["bnb_nf4", "bnb_fp4", "bnb_int8"],
    )
    def test_valid_bnb_schemes_from_string(self, scheme_str):
        config = _make_config(scheme=scheme_str)
        assert isinstance(config.scheme, QuantizationScheme)
        assert config.scheme.value == scheme_str

    def test_invalid_scheme_rejected(self):
        with pytest.raises(OumiConfigError, match="Unsupported scheme"):
            _make_config(scheme="nonexistent")


class TestQuantizationConfigAlgorithm:
    @pytest.mark.parametrize(
        "algo_str,expected",
        [
            ("auto", QuantizationAlgorithm.AUTO),
            ("rtn", QuantizationAlgorithm.RTN),
            ("gptq", QuantizationAlgorithm.GPTQ),
            ("awq", QuantizationAlgorithm.AWQ),
        ],
    )
    def test_valid_algorithm_from_string(self, algo_str, expected):
        config = _make_config(algorithm=algo_str)
        assert config.algorithm == expected

    def test_invalid_algorithm_rejected(self):
        with pytest.raises(OumiConfigError, match="Unsupported algorithm"):
            _make_config(algorithm="invalid_algo")

    def test_bnb_scheme_forces_bnb_algorithm(self):
        config = _make_config(
            scheme=QuantizationScheme.BNB_NF4,
            algorithm="auto",
        )
        assert config.algorithm == QuantizationAlgorithm.BNB

    def test_llmc_scheme_rejects_bnb_algorithm(self):
        with pytest.raises(OumiConfigError, match="not compatible with"):
            _make_config(
                scheme=QuantizationScheme.FP8_DYNAMIC,
                algorithm=QuantizationAlgorithm.BNB,
            )


class TestQuantizationConfigOutputFormat:
    def test_safetensors_accepted(self):
        config = _make_config(output_format="safetensors")
        assert config.output_format == "safetensors"

    def test_invalid_format_rejected(self):
        with pytest.raises(OumiConfigError, match="Unsupported output format"):
            _make_config(output_format="pytorch")


class TestQuantizationConfigYaml:
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

        assert loaded.backend == QuantizationBackend.LLM_COMPRESSOR
        assert loaded.scheme == QuantizationScheme.FP8_DYNAMIC
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
            scheme="w4a16",
            output_path="llama3-w4a16",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "quant.yaml")
            config.to_yaml(path)
            loaded = QuantizationConfig.from_yaml(path)

        assert loaded.backend == QuantizationBackend.LLM_COMPRESSOR
        assert loaded.scheme == QuantizationScheme.W4A16
        assert loaded.model.model_name == "meta-llama/Llama-3.1-8B-Instruct"
        assert loaded.output_path == "llama3-w4a16"
        assert loaded.algorithm == QuantizationAlgorithm.AUTO
        assert loaded.save_compressed is True

    def test_from_yaml_bnb_config(self):
        config = QuantizationConfig(
            model=ModelParams(model_name="test/model"),
            scheme="bnb_nf4",
            output_path="test-bnb",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "quant.yaml")
            config.to_yaml(path)
            loaded = QuantizationConfig.from_yaml(path)

        assert loaded.backend == QuantizationBackend.BNB
        assert loaded.scheme == QuantizationScheme.BNB_NF4
        assert loaded.algorithm == QuantizationAlgorithm.BNB
