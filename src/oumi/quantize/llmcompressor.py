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

Scheme metadata (allowed algorithms, calibration rules, min compute
capability, descriptions) is declared on the class. References:
  - https://docs.vllm.ai/projects/llm-compressor/en/latest/steps/choosing-scheme/
  - https://docs.vllm.ai/projects/llm-compressor/en/latest/guides/compression_schemes/
  - https://docs.vllm.ai/projects/llm-compressor/en/latest/steps/choosing-algo/
"""

import importlib.util
from typing import Any, ClassVar, cast

import torch
from typing_extensions import override

from oumi.core.configs import QuantizationConfig
from oumi.core.configs.quantization_config import (
    QuantizationAlgorithm,
    QuantizationBackend,
    QuantizationScheme,
)
from oumi.quantize.base import BaseQuantization, QuantizationResult, SchemeSpec
from oumi.quantize.utils import (
    assert_output_path_writable,
    format_size,
    get_directory_size,
    load_model_and_tokenizer,
    warn_if_local_gpu_below_inference_capability,
)
from oumi.utils.logging import logger

_LLMC_ALGOS = (
    QuantizationAlgorithm.RTN,
    QuantizationAlgorithm.GPTQ,
    QuantizationAlgorithm.AWQ,
)
_NEEDS_CALIB = (QuantizationAlgorithm.GPTQ, QuantizationAlgorithm.AWQ)


class LLMCompressorQuantization(BaseQuantization):
    """LLM Compressor backend (FP8, GPTQ, AWQ via ``llmcompressor.oneshot``)."""

    backend: ClassVar[QuantizationBackend] = QuantizationBackend.LLM_COMPRESSOR

    schemes: ClassVar[dict[QuantizationScheme, SchemeSpec]] = {
        # --- data-free methods (no calibration needed by default) ---
        QuantizationScheme.FP8_DYNAMIC: SchemeSpec(
            default_algorithm=QuantizationAlgorithm.RTN,
            allowed_algorithms=_LLMC_ALGOS,
            needs_calibration_default=False,
            calibration_required_for=_NEEDS_CALIB,
            min_compute_capability=8.9,
            description="FP8 dynamic quantization (data-free, Hopper+)",
        ),
        QuantizationScheme.FP8_BLOCK: SchemeSpec(
            default_algorithm=QuantizationAlgorithm.RTN,
            allowed_algorithms=_LLMC_ALGOS,
            needs_calibration_default=False,
            calibration_required_for=_NEEDS_CALIB,
            min_compute_capability=8.9,
            description="FP8 block-wise quantization (data-free, Hopper+)",
        ),
        # --- calibration-based weight-only methods ---
        QuantizationScheme.W4A16: SchemeSpec(
            default_algorithm=QuantizationAlgorithm.GPTQ,
            allowed_algorithms=_LLMC_ALGOS,
            needs_calibration_default=True,
            min_compute_capability=7.5,
            description="4-bit weight quantization via GPTQ (Turing+)",
        ),
        QuantizationScheme.W4A16_ASYM: SchemeSpec(
            default_algorithm=QuantizationAlgorithm.AWQ,
            allowed_algorithms=_LLMC_ALGOS,
            needs_calibration_default=True,
            min_compute_capability=7.5,
            description="4-bit asymmetric weight quantization via AWQ (Turing+)",
        ),
        QuantizationScheme.W8A16: SchemeSpec(
            default_algorithm=QuantizationAlgorithm.GPTQ,
            allowed_algorithms=_LLMC_ALGOS,
            needs_calibration_default=True,
            min_compute_capability=7.5,
            description="8-bit weight quantization via GPTQ (Turing+)",
        ),
    }

    def __init__(self):
        self._llmcompressor_available = (
            importlib.util.find_spec("llmcompressor") is not None
        )

    @override
    def raise_if_requirements_not_met(self) -> None:
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
        from llmcompressor import oneshot  # pyright: ignore[reportMissingImports]

        scheme = cast(QuantizationScheme, config.scheme)
        spec = self.schemes[scheme]
        algorithm = spec.resolve_algorithm(
            cast(QuantizationAlgorithm, config.algorithm)
        )
        needs_calibration = spec.needs_calibration_for(algorithm)

        warn_if_local_gpu_below_inference_capability(scheme, spec.min_compute_capability)
        assert_output_path_writable(config.output_path)

        logger.info(
            f"Starting LLM Compressor quantization: scheme={scheme.value}, "
            f"algorithm={algorithm.value}"
        )
        logger.info(f"Loading model: {config.model.model_name}")
        model, tokenizer = load_model_and_tokenizer(config, torch_dtype="auto")
        resolved_path = getattr(model.config, "name_or_path", config.model.model_name)
        logger.info(f"Model loaded from: {resolved_path}")

        recipe = self._build_recipe(config, scheme, algorithm)
        oneshot_kwargs: dict[str, Any] = {"model": model, "recipe": recipe}
        if needs_calibration:
            oneshot_kwargs["dataset"] = self._prepare_calibration_data(config, tokenizer)
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

        size = get_directory_size(config.output_path)
        logger.info(f"Quantization complete. Output size: {format_size(size)}")
        return QuantizationResult(
            output_path=config.output_path,
            backend=self.backend,
            scheme=scheme,
            format_type=config.output_format,
            quantized_size_bytes=size,
        )

    def _build_recipe(
        self,
        config: QuantizationConfig,
        scheme: QuantizationScheme,
        algorithm: QuantizationAlgorithm,
    ):
        """Build the modifier matching ``algorithm`` for the given scheme.

        ``QuantizationScheme.name`` matches the LLM Compressor scheme string
        verbatim (``FP8_DYNAMIC``, ``W4A16``, etc.), so we pass it directly.
        """
        common = {
            "targets": "Linear",
            "scheme": scheme.name,
            "ignore": config.ignore_layers,
        }
        if algorithm is QuantizationAlgorithm.RTN:
            from llmcompressor.modifiers.quantization import (  # pyright: ignore[reportMissingImports]
                QuantizationModifier,
            )

            return QuantizationModifier(**common)
        if algorithm is QuantizationAlgorithm.GPTQ:
            from llmcompressor.modifiers.quantization import (  # pyright: ignore[reportMissingImports]
                GPTQModifier,
            )

            return GPTQModifier(**common, dampening_frac=config.dampening_frac)
        # AWQ
        from llmcompressor.modifiers.awq import (  # pyright: ignore[reportMissingImports]
            AWQModifier,
        )

        return AWQModifier(**common)

    def _prepare_calibration_data(self, config: QuantizationConfig, tokenizer):
        """Load and tokenize calibration data for calibration-based methods."""
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
        text_column = next((c for c in recognized if c in ds.column_names), None)
        if text_column is None:
            raise ValueError(
                f"Calibration dataset '{config.calibration_dataset}' has no "
                f"recognized text column. Available columns: {ds.column_names}. "
                f"Recognized names: {list(recognized)}. Use a dataset that "
                "contains one of these columns, or rename the relevant column "
                "before quantization."
            )
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

        return ds.map(_tokenize, remove_columns=ds.column_names)
