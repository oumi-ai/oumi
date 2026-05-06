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

"""Unit tests for LLM Compressor quantization backend."""

import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from oumi.core.configs import ModelParams, QuantizationConfig
from oumi.core.configs.quantization_config import (
    QuantizationAlgorithm,
    QuantizationBackend,
    QuantizationScheme,
)
from oumi.exceptions import OumiConfigError
from oumi.quantize.base import QuantizationResult
from oumi.quantize.llmcompressor import LLMCompressorQuantization

LLMCOMPRESSOR_SCHEMES = list(LLMCompressorQuantization.schemes.keys())


def _make_config(
    scheme: QuantizationScheme,
    output_path: str = "test",
    **overrides: Any,
) -> QuantizationConfig:
    return QuantizationConfig(
        model=ModelParams(model_name="test/model"),
        scheme=scheme,
        output_path=output_path,
        **overrides,
    )


def _make_llmcompressor_mock_modules():
    llmcompressor = types.ModuleType("llmcompressor")
    llmcompressor.__spec__ = None  # type: ignore[assignment]

    mods = types.ModuleType("llmcompressor.modifiers")
    mods.__spec__ = None  # type: ignore[assignment]

    quant = types.ModuleType("llmcompressor.modifiers.quantization")
    quant.__spec__ = None  # type: ignore[assignment]
    quant.QuantizationModifier = MagicMock(name="QuantizationModifier")  # type: ignore[attr-defined]
    quant.GPTQModifier = MagicMock(name="GPTQModifier")  # type: ignore[attr-defined]

    awq = types.ModuleType("llmcompressor.modifiers.awq")
    awq.__spec__ = None  # type: ignore[assignment]
    awq.AWQModifier = MagicMock(name="AWQModifier")  # type: ignore[attr-defined]

    llmcompressor.modifiers = mods  # type: ignore[attr-defined]
    mods.quantization = quant  # type: ignore[attr-defined]
    mods.awq = awq  # type: ignore[attr-defined]

    llmcompressor.oneshot = MagicMock(name="oneshot")  # type: ignore[attr-defined]

    return {
        "llmcompressor": llmcompressor,
        "llmcompressor.modifiers": mods,
        "llmcompressor.modifiers.quantization": quant,
        "llmcompressor.modifiers.awq": awq,
    }


class _LLMCompressorModuleInjector:
    def __init__(self):
        self.mock_modules = _make_llmcompressor_mock_modules()
        self._saved: dict[str, types.ModuleType | None] = {}

    def __enter__(self):
        for name in self.mock_modules:
            self._saved[name] = sys.modules.get(name)
            sys.modules[name] = self.mock_modules[name]
        return self.mock_modules

    def __exit__(self, *exc):
        for name, prev in self._saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


class TestLLMCompressorSchemeMetadata:
    def test_backend_identity(self):
        assert (
            LLMCompressorQuantization.backend is QuantizationBackend.LLM_COMPRESSOR
        )

    def test_output_format(self):
        assert LLMCompressorQuantization.output_format == "safetensors"

    @pytest.mark.parametrize("scheme", LLMCOMPRESSOR_SCHEMES)
    def test_owns(self, scheme):
        assert LLMCompressorQuantization.owns(scheme) is True

    def test_does_not_own_bnb(self):
        assert LLMCompressorQuantization.owns(QuantizationScheme.BNB_NF4) is False

    def test_config_rejects_bnb_algorithm(self):
        with pytest.raises(OumiConfigError, match="not allowed"):
            _make_config(QuantizationScheme.FP8_DYNAMIC, algorithm="bnb")

    def test_default_algorithms(self):
        s = LLMCompressorQuantization.schemes
        assert (
            s[QuantizationScheme.FP8_DYNAMIC].default_algorithm
            is QuantizationAlgorithm.RTN
        )
        assert (
            s[QuantizationScheme.FP8_BLOCK].default_algorithm
            is QuantizationAlgorithm.RTN
        )
        assert (
            s[QuantizationScheme.W4A16].default_algorithm
            is QuantizationAlgorithm.GPTQ
        )
        assert (
            s[QuantizationScheme.W4A16_ASYM].default_algorithm
            is QuantizationAlgorithm.AWQ
        )
        assert (
            s[QuantizationScheme.W8A16].default_algorithm
            is QuantizationAlgorithm.GPTQ
        )

    def test_calibration_required_for_overrides(self):
        spec = LLMCompressorQuantization.schemes[QuantizationScheme.FP8_DYNAMIC]
        assert spec.needs_calibration_for(QuantizationAlgorithm.RTN) is False
        assert spec.needs_calibration_for(QuantizationAlgorithm.GPTQ) is True
        assert spec.needs_calibration_for(QuantizationAlgorithm.AWQ) is True


class TestLLMCompressorRequirements:
    def setup_method(self):
        self.quantizer = LLMCompressorQuantization()

    def test_missing_llmcompressor(self):
        self.quantizer._llmcompressor_available = False
        with pytest.raises(RuntimeError, match="requires the llmcompressor library"):
            self.quantizer.raise_if_requirements_not_met()

    @patch("torch.cuda.is_available", return_value=False)
    def test_no_cuda(self, _mock_cuda):
        self.quantizer._llmcompressor_available = True
        with pytest.raises(RuntimeError, match="requires a CUDA GPU"):
            self.quantizer.raise_if_requirements_not_met()

    @patch("torch.cuda.is_available", return_value=True)
    def test_all_requirements_met(self, _mock_cuda):
        self.quantizer._llmcompressor_available = True
        self.quantizer.raise_if_requirements_not_met()


class TestBuildRecipe:
    def setup_method(self):
        self.quantizer = LLMCompressorQuantization()

    def _run(self, config, scheme, algorithm):
        with _LLMCompressorModuleInjector() as mocks:
            recipe = self.quantizer._build_recipe(config, scheme, algorithm)
            return recipe, mocks

    def test_fp8_dynamic_rtn_uses_quantization_modifier(self):
        config = _make_config(QuantizationScheme.FP8_DYNAMIC)
        _, mocks = self._run(
            config, QuantizationScheme.FP8_DYNAMIC, QuantizationAlgorithm.RTN
        )
        mocks[
            "llmcompressor.modifiers.quantization"
        ].QuantizationModifier.assert_called_once_with(
            targets="Linear",
            scheme="FP8_DYNAMIC",
            ignore=["lm_head"],
        )

    def test_fp8_block_rtn(self):
        config = _make_config(QuantizationScheme.FP8_BLOCK)
        _, mocks = self._run(
            config, QuantizationScheme.FP8_BLOCK, QuantizationAlgorithm.RTN
        )
        mocks[
            "llmcompressor.modifiers.quantization"
        ].QuantizationModifier.assert_called_once_with(
            targets="Linear",
            scheme="FP8_BLOCK",
            ignore=["lm_head"],
        )

    def test_w4a16_gptq_uses_gptq_modifier(self):
        config = _make_config(QuantizationScheme.W4A16)
        _, mocks = self._run(
            config, QuantizationScheme.W4A16, QuantizationAlgorithm.GPTQ
        )
        mocks[
            "llmcompressor.modifiers.quantization"
        ].GPTQModifier.assert_called_once_with(
            targets="Linear",
            scheme="W4A16",
            ignore=["lm_head"],
            dampening_frac=0.1,
        )

    def test_w8a16_gptq(self):
        config = _make_config(QuantizationScheme.W8A16)
        _, mocks = self._run(
            config, QuantizationScheme.W8A16, QuantizationAlgorithm.GPTQ
        )
        mocks[
            "llmcompressor.modifiers.quantization"
        ].GPTQModifier.assert_called_once_with(
            targets="Linear",
            scheme="W8A16",
            ignore=["lm_head"],
            dampening_frac=0.1,
        )

    def test_w4a16_asym_awq(self):
        config = _make_config(QuantizationScheme.W4A16_ASYM)
        _, mocks = self._run(
            config, QuantizationScheme.W4A16_ASYM, QuantizationAlgorithm.AWQ
        )
        mocks["llmcompressor.modifiers.awq"].AWQModifier.assert_called_once_with(
            targets="Linear",
            scheme="W4A16_ASYM",
            ignore=["lm_head"],
        )

    def test_respects_ignore_layers(self):
        custom_ignore = ["lm_head", "re:.*gate$"]
        config = _make_config(
            QuantizationScheme.FP8_DYNAMIC, ignore_layers=custom_ignore
        )
        _, mocks = self._run(
            config, QuantizationScheme.FP8_DYNAMIC, QuantizationAlgorithm.RTN
        )
        mocks[
            "llmcompressor.modifiers.quantization"
        ].QuantizationModifier.assert_called_once_with(
            targets="Linear",
            scheme="FP8_DYNAMIC",
            ignore=custom_ignore,
        )

    def test_explicit_algorithm_override_to_gptq(self):
        # User picks GPTQ on FP8_DYNAMIC: recipe should use GPTQModifier.
        config = _make_config(
            QuantizationScheme.FP8_DYNAMIC, algorithm=QuantizationAlgorithm.GPTQ
        )
        _, mocks = self._run(
            config, QuantizationScheme.FP8_DYNAMIC, QuantizationAlgorithm.GPTQ
        )
        mocks[
            "llmcompressor.modifiers.quantization"
        ].GPTQModifier.assert_called_once()


class TestQuantizeIntegration:
    def setup_method(self):
        self.quantizer = LLMCompressorQuantization()

    @patch("oumi.quantize.llmcompressor.warn_if_local_gpu_below_inference_capability")
    @patch("oumi.quantize.llmcompressor.assert_output_path_writable")
    @patch("oumi.quantize.llmcompressor.get_directory_size", return_value=4096)
    @patch("oumi.quantize.llmcompressor.load_model_and_tokenizer")
    def test_quantize_calls_oneshot(
        self, mock_load, _mock_size, _mock_writable, _mock_warn
    ):
        config = _make_config(
            QuantizationScheme.FP8_DYNAMIC, output_path="/tmp/test_output"
        )
        mock_model = MagicMock()
        mock_load.return_value = (mock_model, MagicMock())

        with _LLMCompressorModuleInjector() as mocks:
            mock_oneshot = mocks["llmcompressor"].oneshot
            self.quantizer.quantize(config)
            mock_oneshot.assert_called_once()
            assert mock_oneshot.call_args.kwargs["model"] is mock_model

    @patch("oumi.quantize.llmcompressor.warn_if_local_gpu_below_inference_capability")
    @patch("oumi.quantize.llmcompressor.assert_output_path_writable")
    @patch("oumi.quantize.llmcompressor.get_directory_size", return_value=4096)
    @patch("oumi.quantize.llmcompressor.load_model_and_tokenizer")
    def test_quantize_saves_model_and_tokenizer(
        self, mock_load, _mock_size, _mock_writable, _mock_warn
    ):
        config = _make_config(
            QuantizationScheme.FP8_DYNAMIC, output_path="/tmp/test_output"
        )
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        with _LLMCompressorModuleInjector():
            self.quantizer.quantize(config)

        mock_model.save_pretrained.assert_called_once_with(
            config.output_path, save_compressed=True
        )
        mock_tokenizer.save_pretrained.assert_called_once_with(config.output_path)

    @patch("oumi.quantize.llmcompressor.warn_if_local_gpu_below_inference_capability")
    @patch("oumi.quantize.llmcompressor.assert_output_path_writable")
    @patch("oumi.quantize.llmcompressor.get_directory_size", return_value=2048)
    @patch("oumi.quantize.llmcompressor.load_model_and_tokenizer")
    def test_quantize_returns_quantization_result(
        self, mock_load, _mock_size, _mock_writable, _mock_warn
    ):
        config = _make_config(
            QuantizationScheme.FP8_DYNAMIC, output_path="/tmp/test_output"
        )
        mock_load.return_value = (MagicMock(), MagicMock())

        with _LLMCompressorModuleInjector():
            result = self.quantizer.quantize(config)

        assert isinstance(result, QuantizationResult)
        assert result.backend is QuantizationBackend.LLM_COMPRESSOR
        assert result.scheme is QuantizationScheme.FP8_DYNAMIC
        assert result.output_path == "/tmp/test_output"
        assert result.format_type == "safetensors"
        assert result.quantized_size_bytes == 2048

    @patch("oumi.quantize.llmcompressor.warn_if_local_gpu_below_inference_capability")
    @patch("oumi.quantize.llmcompressor.assert_output_path_writable")
    @patch("oumi.quantize.llmcompressor.get_directory_size", return_value=1024)
    @patch("oumi.quantize.llmcompressor.load_model_and_tokenizer")
    def test_quantize_calibration_method_passes_dataset(
        self, mock_load, _mock_size, _mock_writable, _mock_warn
    ):
        config = _make_config(
            QuantizationScheme.W4A16,
            output_path="/tmp/test_output",
            calibration_samples=64,
            max_seq_length=512,
        )
        mock_load.return_value = (MagicMock(), MagicMock())
        mock_dataset = MagicMock()

        with _LLMCompressorModuleInjector() as mocks:
            mock_oneshot = mocks["llmcompressor"].oneshot
            with patch.object(
                self.quantizer,
                "_prepare_calibration_data",
                return_value=mock_dataset,
            ):
                self.quantizer.quantize(config)

            call_kw = mock_oneshot.call_args.kwargs
            assert call_kw["dataset"] is mock_dataset
            assert call_kw["max_seq_length"] == 512
            assert call_kw["num_calibration_samples"] == 64


class TestPrepareCalibrationData:
    def setup_method(self):
        self.quantizer = LLMCompressorQuantization()

    def test_prepare_calibration_data_calls_load_dataset(self):
        config = _make_config(
            QuantizationScheme.W4A16,
            calibration_dataset="test/dataset",
            calibration_split="train",
            calibration_samples=128,
            max_seq_length=1024,
        )
        mock_tokenizer = MagicMock()

        mock_ds = MagicMock()
        mock_ds.column_names = ["text"]
        mock_ds.map.return_value = mock_ds

        with patch("datasets.load_dataset", return_value=mock_ds) as mock_load:
            result = self.quantizer._prepare_calibration_data(config, mock_tokenizer)

            mock_load.assert_called_once_with(
                "test/dataset",
                split="train[:128]",
            )
            mock_ds.map.assert_called_once()
            assert result is mock_ds
