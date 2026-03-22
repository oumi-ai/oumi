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
from typing import cast

import torch
from typing_extensions import override

from oumi.core.configs import QuantizationConfig
from oumi.quantize.base import BaseQuantization, QuantizationResult
from oumi.quantize.constants import (
    LLMCOMPRESSOR_METHODS,
    METHOD_REGISTRY,
    MethodInfo,
    QuantizationAlgorithm,
    QuantizationMethod,
)
from oumi.quantize.utils import format_size, get_directory_size
from oumi.utils.logging import logger

_VALID_LLMC_ALGORITHMS = frozenset(
    {
        QuantizationAlgorithm.RTN,
        QuantizationAlgorithm.GPTQ,
        QuantizationAlgorithm.AWQ,
    }
)


class LLMCompressorQuantization(BaseQuantization):
    """LLM Compressor quantization backend.

    Uses ``llmcompressor.oneshot()`` with QuantizationModifier, GPTQModifier,
    or AWQModifier depending on the method's algorithm.
    """

    supported_methods = LLMCOMPRESSOR_METHODS
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

        from llmcompressor import oneshot
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # config.method is coerced to QuantizationMethod in __post_init__
        method = cast(QuantizationMethod, config.method)
        method_info = METHOD_REGISTRY[method]

        logger.info(
            f"Starting LLM Compressor quantization: "
            f"method={config.method}, scheme={method_info.scheme}"
        )

        logger.info(f"Loading model: {config.model.model_name}")
        model_kwargs = dict(config.model.model_kwargs or {})
        # Remove keys passed explicitly below to avoid duplicate keyword argument errors.
        model_kwargs.pop("device_map", None)
        model_kwargs.pop("torch_dtype", None)
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

        recipe = self._build_recipe(config, method_info)

        oneshot_kwargs: dict = {
            "model": model,
            "recipe": recipe,
        }

        if method_info.needs_calibration:
            dataset = self._prepare_calibration_data(config, tokenizer)
            oneshot_kwargs["dataset"] = dataset
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
            quantization_method=method,
            format_type=config.output_format,
            additional_info={
                "scheme": method_info.scheme,
                "algorithm": method_info.algorithm.value,
                "needs_calibration": method_info.needs_calibration,
            },
        )

    def _build_recipe(self, config: QuantizationConfig, method_info: MethodInfo):
        """Build an LLM Compressor recipe modifier for the given method.

        Returns a single modifier (QuantizationModifier, GPTQModifier, or
        AWQModifier) configured with the appropriate scheme and parameters.

        Raises:
            ValueError: If the resolved algorithm is not supported by LLM
                Compressor, or if an explicit override is incompatible with
                the chosen method.
        """
        # config.algorithm is coerced to QuantizationAlgorithm in __post_init__
        config_algorithm = cast(QuantizationAlgorithm, config.algorithm)
        algorithm = (
            method_info.algorithm
            if config_algorithm == QuantizationAlgorithm.AUTO
            else config_algorithm
        )

        if algorithm not in _VALID_LLMC_ALGORITHMS:
            # config.method is coerced to QuantizationMethod in __post_init__
            method_name = cast(QuantizationMethod, config.method).value
            raise ValueError(
                f"Unsupported algorithm '{algorithm.value}' for LLM Compressor "
                f"method '{method_name}'. "
                f"Supported: {sorted(a.value for a in _VALID_LLMC_ALGORITHMS)}."
            )

        scheme = method_info.scheme
        ignore = config.ignore_layers

        if algorithm == QuantizationAlgorithm.RTN:
            from llmcompressor.modifiers.quantization import QuantizationModifier

            return QuantizationModifier(
                targets="Linear",
                scheme=scheme,
                ignore=ignore,
            )
        elif algorithm == QuantizationAlgorithm.GPTQ:
            from llmcompressor.modifiers.quantization import GPTQModifier

            return GPTQModifier(
                targets="Linear",
                scheme=scheme,
                ignore=ignore,
                dampening_frac=config.dampening_frac,
            )
        else:  # AWQ
            from llmcompressor.modifiers.quantization import AWQModifier

            return AWQModifier(
                targets="Linear",
                scheme=scheme,
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
            for col in ("text", "content", "messages"):
                if col in dataset.column_names:
                    return col
            return dataset.column_names[0]

        text_column = _get_text_column(ds)

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
