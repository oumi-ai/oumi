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

"""Unit tests for LLM Compressor quantization."""

import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from oumi.core.configs import ModelParams, QuantizationConfig
from oumi.core.configs.quantization_config import (
    QuantizationBackend,
    QuantizationScheme,
)
from oumi.exceptions import OumiConfigError
from oumi.quantize.base import QuantizationResult
from oumi.quantize.constants import (
    LLMCOMPRESSOR_SCHEMES,
    SCHEME_REGISTRY,
    QuantizationAlgorithm,
    SchemeInfo,
)
from oumi.quantize.llmcompressor_quantizer import LLMCompressorQuantization


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
    """Create a hierarchy of mock modules mimicking llmcompressor."""
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
    """Context manager that injects mock llmcompressor modules into sys.modules."""

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


class TestLLMCompressorSupportsAndValidation:
    def setup_method(self):
        self.quantizer = LLMCompressorQuantization()

    def test_supported_schemes(self):
        assert sorted(self.quantizer.supported_schemes) == sorted(LLMCOMPRESSOR_SCHEMES)

    def test_supported_formats(self):
        assert self.quantizer.supported_formats == ["safetensors"]

    @pytest.mark.parametrize("scheme", LLMCOMPRESSOR_SCHEMES)
    def test_supports_scheme_valid(self, scheme):
        assert self.quantizer.supports_scheme(scheme) is True

    @pytest.mark.parametrize(
        "scheme",
        [
            QuantizationScheme.BNB_NF4,
            QuantizationScheme.BNB_FP4,
            QuantizationScheme.BNB_INT8,
        ],
    )
    def test_supports_scheme_invalid(self, scheme):
        assert self.quantizer.supports_scheme(scheme) is False

    def test_validate_config_valid(self):
        self.quantizer.validate_config(_make_config(QuantizationScheme.FP8_DYNAMIC))

    def test_validate_config_wrong_scheme(self):
        config = QuantizationConfig(
            model=ModelParams(model_name="test/model"),
            scheme=QuantizationScheme.BNB_NF4,
            output_path="test",
        )
        with pytest.raises(ValueError, match="not supported by"):
            self.quantizer.validate_config(config)

    def test_validate_config_wrong_format(self):
        with pytest.raises(OumiConfigError, match="Unsupported output format"):
            _make_config(QuantizationScheme.FP8_DYNAMIC, output_format="pytorch")


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

    def _run_recipe(self, config, scheme_info):
        with _LLMCompressorModuleInjector() as mocks:
            recipe = self.quantizer._build_recipe(config, scheme_info)
            return recipe, mocks

    def test_build_recipe_fp8_dynamic(self):
        config = _make_config(QuantizationScheme.FP8_DYNAMIC)
        info = SCHEME_REGISTRY[QuantizationScheme.FP8_DYNAMIC]
        _, mocks = self._run_recipe(config, info)
        mocks[
            "llmcompressor.modifiers.quantization"
        ].QuantizationModifier.assert_called_once_with(
            targets="Linear",
            scheme="FP8_DYNAMIC",
            ignore=["lm_head"],
        )

    def test_build_recipe_fp8_block(self):
        config = _make_config(QuantizationScheme.FP8_BLOCK)
        info = SCHEME_REGISTRY[QuantizationScheme.FP8_BLOCK]
        _, mocks = self._run_recipe(config, info)
        mocks[
            "llmcompressor.modifiers.quantization"
        ].QuantizationModifier.assert_called_once_with(
            targets="Linear",
            scheme="FP8_BLOCK",
            ignore=["lm_head"],
        )

    def test_build_recipe_w4a16(self):
        config = _make_config(QuantizationScheme.W4A16)
        info = SCHEME_REGISTRY[QuantizationScheme.W4A16]
        _, mocks = self._run_recipe(config, info)
        mocks[
            "llmcompressor.modifiers.quantization"
        ].GPTQModifier.assert_called_once_with(
            targets="Linear",
            scheme="W4A16",
            ignore=["lm_head"],
            dampening_frac=0.1,
        )

    def test_build_recipe_w8a16(self):
        config = _make_config(QuantizationScheme.W8A16)
        info = SCHEME_REGISTRY[QuantizationScheme.W8A16]
        _, mocks = self._run_recipe(config, info)
        mocks[
            "llmcompressor.modifiers.quantization"
        ].GPTQModifier.assert_called_once_with(
            targets="Linear",
            scheme="W8A16",
            ignore=["lm_head"],
            dampening_frac=0.1,
        )

    def test_build_recipe_w4a16_asym(self):
        config = _make_config(QuantizationScheme.W4A16_ASYM)
        info = SCHEME_REGISTRY[QuantizationScheme.W4A16_ASYM]
        _, mocks = self._run_recipe(config, info)
        mocks["llmcompressor.modifiers.awq"].AWQModifier.assert_called_once_with(
            targets="Linear",
            scheme="W4A16_ASYM",
            ignore=["lm_head"],
        )

    def test_build_recipe_respects_ignore_layers(self):
        custom_ignore = ["lm_head", "re:.*gate$"]
        config = _make_config(
            QuantizationScheme.FP8_DYNAMIC,
            ignore_layers=custom_ignore,
        )
        info = SCHEME_REGISTRY[QuantizationScheme.FP8_DYNAMIC]
        _, mocks = self._run_recipe(config, info)
        mocks[
            "llmcompressor.modifiers.quantization"
        ].QuantizationModifier.assert_called_once_with(
            targets="Linear",
            scheme="FP8_DYNAMIC",
            ignore=custom_ignore,
        )

    def test_algorithm_auto_defers_to_scheme_info(self):
        config = _make_config(QuantizationScheme.FP8_DYNAMIC)
        info = SCHEME_REGISTRY[QuantizationScheme.FP8_DYNAMIC]
        _, mocks = self._run_recipe(config, info)
        mocks[
            "llmcompressor.modifiers.quantization"
        ].QuantizationModifier.assert_called_once()

    def test_explicit_algorithm_override(self):
        config = _make_config(
            QuantizationScheme.FP8_DYNAMIC,
            algorithm=QuantizationAlgorithm.GPTQ,
        )
        info = SchemeInfo(
            backend=QuantizationBackend.LLM_COMPRESSOR,
            llmc_scheme="FP8_DYNAMIC",
            default_algorithm=QuantizationAlgorithm.RTN,
            needs_calibration=False,
            min_compute_capability=8.9,
            description="test",
        )
        _, mocks = self._run_recipe(config, info)
        mocks["llmcompressor.modifiers.quantization"].GPTQModifier.assert_called_once()

    def test_unsupported_algorithm_raises(self):
        # BNB algorithm on LLM Compressor backend is rejected at config level,
        # so we use a valid config (auto algorithm) and test _build_recipe
        # with a synthetic SchemeInfo whose default is BNB.
        config = _make_config(QuantizationScheme.FP8_DYNAMIC)
        info = SchemeInfo(
            backend=QuantizationBackend.LLM_COMPRESSOR,
            llmc_scheme="FP8_DYNAMIC",
            default_algorithm=QuantizationAlgorithm.BNB,
            needs_calibration=False,
            min_compute_capability=8.9,
            description="test",
        )
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            self._run_recipe(config, info)


class TestQuantizeIntegration:
    def setup_method(self):
        self.quantizer = LLMCompressorQuantization()

    @patch(
        "oumi.quantize.llmcompressor_quantizer.get_directory_size", return_value=4096
    )
    @patch("transformers.AutoTokenizer")
    @patch("transformers.AutoModelForCausalLM")
    def test_quantize_calls_oneshot(self, mock_auto_model, mock_auto_tok, _mock_size):
        config = _make_config(
            QuantizationScheme.FP8_DYNAMIC, output_path="/tmp/test_output"
        )
        mock_model = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_auto_tok.from_pretrained.return_value = MagicMock()

        with _LLMCompressorModuleInjector() as mocks:
            mock_oneshot = mocks["llmcompressor"].oneshot
            self.quantizer.quantize(config)

            mock_oneshot.assert_called_once()
            assert mock_oneshot.call_args.kwargs["model"] is mock_model

    @patch(
        "oumi.quantize.llmcompressor_quantizer.get_directory_size", return_value=4096
    )
    @patch("transformers.AutoTokenizer")
    @patch("transformers.AutoModelForCausalLM")
    def test_quantize_saves_model_and_tokenizer(
        self, mock_auto_model, mock_auto_tok, _mock_size
    ):
        config = _make_config(
            QuantizationScheme.FP8_DYNAMIC, output_path="/tmp/test_output"
        )
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_auto_tok.from_pretrained.return_value = mock_tokenizer

        with _LLMCompressorModuleInjector():
            self.quantizer.quantize(config)

        mock_model.save_pretrained.assert_called_once_with(
            config.output_path,
            save_compressed=True,
        )
        mock_tokenizer.save_pretrained.assert_called_once_with(config.output_path)

    @patch(
        "oumi.quantize.llmcompressor_quantizer.get_directory_size", return_value=2048
    )
    @patch("transformers.AutoTokenizer")
    @patch("transformers.AutoModelForCausalLM")
    def test_quantize_returns_quantization_result(
        self, mock_auto_model, mock_auto_tok, _mock_size
    ):
        config = _make_config(
            QuantizationScheme.FP8_DYNAMIC, output_path="/tmp/test_output"
        )
        mock_auto_model.from_pretrained.return_value = MagicMock()
        mock_auto_tok.from_pretrained.return_value = MagicMock()

        with _LLMCompressorModuleInjector():
            result = self.quantizer.quantize(config)

        assert isinstance(result, QuantizationResult)
        assert result.backend == QuantizationBackend.LLM_COMPRESSOR
        assert result.scheme == QuantizationScheme.FP8_DYNAMIC
        assert result.output_path == "/tmp/test_output"
        assert result.format_type == "safetensors"
        assert result.quantized_size_bytes == 2048
        assert result.additional_info["llmc_scheme"] == "FP8_DYNAMIC"
        assert result.additional_info["algorithm"] == "rtn"
        assert result.additional_info["needs_calibration"] is False

    @patch(
        "oumi.quantize.llmcompressor_quantizer.get_directory_size", return_value=1024
    )
    @patch("transformers.AutoTokenizer")
    @patch("transformers.AutoModelForCausalLM")
    def test_quantize_calibration_method_passes_dataset(
        self, mock_auto_model, mock_auto_tok, _mock_size
    ):
        config = _make_config(
            QuantizationScheme.W4A16,
            output_path="/tmp/test_output",
            calibration_samples=64,
            max_seq_length=512,
        )
        mock_auto_model.from_pretrained.return_value = MagicMock()
        mock_auto_tok.from_pretrained.return_value = MagicMock()
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

        with patch(
            "datasets.load_dataset",
            return_value=mock_ds,
        ) as mock_load:
            result = self.quantizer._prepare_calibration_data(config, mock_tokenizer)

            mock_load.assert_called_once_with(
                "test/dataset",
                split="train[:128]",
            )
            mock_ds.map.assert_called_once()
            assert result is mock_ds
