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

"""LLM Compressor quantization backend.

Wraps the vLLM LLM Compressor library to provide FP8, GPTQ, and AWQ
quantization through a unified ``oneshot()`` interface. Produces models
in the compressed-tensors format optimized for vLLM serving.
"""

import importlib.util
from typing import Any, cast

import torch
from typing_extensions import override

from oumi.core.configs import QuantizationConfig
from oumi.core.configs.quantization_config import (
    QuantizationBackend,
    QuantizationScheme,
)
from oumi.quantize.base import BaseQuantization, QuantizationResult
from oumi.quantize.constants import (
    LLMCOMPRESSOR_SCHEMES,
    SCHEME_REGISTRY,
    QuantizationAlgorithm,
    SchemeInfo,
)
from oumi.quantize.utils import (
    assert_output_path_writable,
    format_size,
    get_directory_size,
    pop_with_override_warning,
    warn_if_local_gpu_below_inference_capability,
)
from oumi.utils.logging import logger

_VALID_LLMC_ALGORITHMS = frozenset(
    {
        QuantizationAlgorithm.RTN,
        QuantizationAlgorithm.GPTQ,
        QuantizationAlgorithm.AWQ,
    }
)

# Algorithms that always require calibration data, even when the scheme's
# default does not (e.g. user overrides FP8_DYNAMIC's RTN with GPTQ).
_CALIBRATION_REQUIRED_ALGORITHMS = frozenset(
    {QuantizationAlgorithm.GPTQ, QuantizationAlgorithm.AWQ}
)


def _resolve_algorithm(
    config: QuantizationConfig, scheme_info: SchemeInfo
) -> QuantizationAlgorithm:
    """Return the effective algorithm (resolves AUTO to the scheme default)."""
    # config.algorithm is coerced to QuantizationAlgorithm in __post_init__.
    config_algorithm = cast(QuantizationAlgorithm, config.algorithm)
    if config_algorithm == QuantizationAlgorithm.AUTO:
        return scheme_info.default_algorithm
    return config_algorithm


class LLMCompressorQuantization(BaseQuantization):
    """LLM Compressor quantization backend.

    Uses ``llmcompressor.oneshot()`` with QuantizationModifier, GPTQModifier,
    or AWQModifier depending on the scheme's algorithm.
    """

    supported_schemes = LLMCOMPRESSOR_SCHEMES
    supported_formats = ["safetensors"]

    def __init__(self):
        """Initialize LLM Compressor quantizer."""
        self._llmcompressor_available = (
            importlib.util.find_spec("llmcompressor") is not None
        )

    @override
    def raise_if_requirements_not_met(self) -> None:
        """Check that llmcompressor is installed and a GPU is available."""
        if not self._llmcompressor_available:
            raise RuntimeError(
                "LLM Compressor quantization requires the llmcompressor library.\n"
                "Install with: pip install oumi[quantization]"
            )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "LLM Compressor quantization requires a CUDA GPU. "
                "Please use a machine with at least 1 GPU."
            )

    @override
    def quantize(self, config: QuantizationConfig) -> QuantizationResult:
        """Quantize a model using LLM Compressor.

        Args:
            config: Quantization configuration.

        Returns:
            QuantizationResult with output path and size information.
        """
        self.validate_config(config)

        from llmcompressor import oneshot  # pyright: ignore[reportMissingImports]
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # config.scheme is coerced to QuantizationScheme in __post_init__
        scheme = cast(QuantizationScheme, config.scheme)
        scheme_info = SCHEME_REGISTRY[scheme]

        warn_if_local_gpu_below_inference_capability(scheme)
        assert_output_path_writable(config.output_path)

        logger.info(
            f"Starting LLM Compressor quantization: "
            f"scheme={scheme.value}, llmc_scheme={scheme_info.llmc_scheme}"
        )

        logger.info(f"Loading model: {config.model.model_name}")
        model_kwargs = dict(config.model.model_kwargs or {})
        pop_with_override_warning(
            model_kwargs,
            ("device_map", "torch_dtype"),
            "LLM Compressor quantization",
        )
        model_kwargs["trust_remote_code"] = config.model.trust_remote_code
        model = AutoModelForCausalLM.from_pretrained(
            config.model.model_name,
            device_map="auto",
            torch_dtype="auto",
            **model_kwargs,
        )

        resolved_path = getattr(model.config, "name_or_path", config.model.model_name)
        logger.info(f"Model loaded from: {resolved_path}")

        tokenizer = AutoTokenizer.from_pretrained(
            config.model.tokenizer_name or config.model.model_name,
            trust_remote_code=config.model.trust_remote_code,
            **(config.model.tokenizer_kwargs or {}),
        )

        resolved_algorithm = _resolve_algorithm(config, scheme_info)
        needs_calibration = (
            scheme_info.needs_calibration
            or resolved_algorithm in _CALIBRATION_REQUIRED_ALGORITHMS
        )

        recipe = self._build_recipe(config, scheme_info, resolved_algorithm)
        oneshot_kwargs: dict[str, Any] = {"model": model, "recipe": recipe}

        if needs_calibration:
            oneshot_kwargs["dataset"] = self._prepare_calibration_data(
                config, tokenizer
            )
            oneshot_kwargs["max_seq_length"] = config.max_seq_length
            oneshot_kwargs["num_calibration_samples"] = config.calibration_samples

        logger.info("Running oneshot quantization...")
        oneshot(**oneshot_kwargs)

        logger.info(f"Saving quantized model to: {config.output_path}")
        model.save_pretrained(
            config.output_path,
            save_compressed=config.save_compressed,
        )
        tokenizer.save_pretrained(config.output_path)

        output_size = get_directory_size(config.output_path)

        logger.info(f"Quantization complete. Output size: {format_size(output_size)}")

        return QuantizationResult(
            quantized_size_bytes=output_size,
            output_path=config.output_path,
            backend=QuantizationBackend.LLM_COMPRESSOR,
            scheme=scheme,
            format_type=config.output_format,
            additional_info={
                "llmc_scheme": scheme_info.llmc_scheme,
                "algorithm": resolved_algorithm.value,
                "needs_calibration": needs_calibration,
            },
        )

    def _build_recipe(
        self,
        config: QuantizationConfig,
        scheme_info: SchemeInfo,
        algorithm: QuantizationAlgorithm | None = None,
    ):
        """Build an LLM Compressor recipe modifier for the given scheme.

        Returns a single modifier (QuantizationModifier, GPTQModifier, or
        AWQModifier) configured with the appropriate scheme and parameters.

        Args:
            config: Quantization configuration.
            scheme_info: Registry entry for the scheme.
            algorithm: Pre-resolved algorithm. If None, resolved from ``config``.

        Raises:
            ValueError: If the resolved algorithm is not supported by LLM
                Compressor, or if an explicit override is incompatible with
                the chosen scheme.
        """
        if algorithm is None:
            algorithm = _resolve_algorithm(config, scheme_info)

        if algorithm not in _VALID_LLMC_ALGORITHMS:
            scheme_name = cast(QuantizationScheme, config.scheme).value
            raise ValueError(
                f"Unsupported algorithm '{algorithm.value}' for LLM Compressor "
                f"scheme '{scheme_name}'. "
                f"Supported: {sorted(a.value for a in _VALID_LLMC_ALGORITHMS)}."
            )

        llmc_scheme = scheme_info.llmc_scheme
        ignore = config.ignore_layers

        if algorithm == QuantizationAlgorithm.RTN:
            from llmcompressor.modifiers.quantization import (  # pyright: ignore[reportMissingImports]
                QuantizationModifier,
            )

            return QuantizationModifier(
                targets="Linear",
                scheme=llmc_scheme,
                ignore=ignore,
            )
        elif algorithm == QuantizationAlgorithm.GPTQ:
            from llmcompressor.modifiers.quantization import (  # pyright: ignore[reportMissingImports]
                GPTQModifier,
            )

            return GPTQModifier(
                targets="Linear",
                scheme=llmc_scheme,
                ignore=ignore,
                dampening_frac=config.dampening_frac,
            )
        else:  # AWQ
            from llmcompressor.modifiers.awq import (  # pyright: ignore[reportMissingImports]
                AWQModifier,
            )

            return AWQModifier(
                targets="Linear",
                scheme=llmc_scheme,
                ignore=ignore,
            )

    def _prepare_calibration_data(self, config: QuantizationConfig, tokenizer):
        """Load and tokenize calibration data for calibration-based methods.

        Returns a HuggingFace Dataset with input_ids and attention_mask columns
        ready for ``oneshot()``.
        """
        from datasets import load_dataset

        logger.info(
            f"Loading calibration data: {config.calibration_dataset} "
            f"(split={config.calibration_split}, "
            f"samples={config.calibration_samples})"
        )

        ds = load_dataset(
            config.calibration_dataset,
            split=f"{config.calibration_split}[:{config.calibration_samples}]",
        )

        def _get_text_column(dataset) -> str:
            """Detect the text column in the dataset."""
            recognized = (
                "text",
                "content",
                "messages",
                "prompt",
                "instruction",
                "input",
                "question",
                "query",
                "body",
            )
            for col in recognized:
                if col in dataset.column_names:
                    return col
            raise ValueError(
                f"Calibration dataset '{config.calibration_dataset}' has no "
                f"recognized text column. Available columns: "
                f"{dataset.column_names}. Recognized names: {list(recognized)}. "
                "Use a dataset that contains one of these columns, or rename "
                "the relevant column before quantization."
            )

        text_column = _get_text_column(ds)
        logger.info(f"Using calibration text column: '{text_column}'")

        def _tokenize(sample):
            text = sample[text_column]
            if isinstance(text, list):
                text = tokenizer.apply_chat_template(
                    text, tokenize=False, add_generation_prompt=False
                )
            return tokenizer(
                text,
                padding=False,
                max_length=config.max_seq_length,
                truncation=True,
                add_special_tokens=False,
            )

        ds = ds.map(_tokenize, remove_columns=ds.column_names)
        return ds
