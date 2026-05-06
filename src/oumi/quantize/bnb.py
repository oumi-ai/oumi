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

"""BitsAndBytes quantization backend.

Uses ``transformers.BitsAndBytesConfig`` to apply NF4, FP4, or INT8
quantization at model-load time. References:
  - https://huggingface.co/docs/transformers/main/en/main_classes/quantization
  - https://huggingface.co/docs/bitsandbytes/en/reference/nn/linear4bit
  - https://huggingface.co/docs/bitsandbytes/en/reference/nn/linear8bit
"""

import importlib.util
from pathlib import Path
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

# Per-scheme arguments for transformers.BitsAndBytesConfig.
_BNB_KWARGS: dict[QuantizationScheme, dict[str, Any]] = {
    QuantizationScheme.BNB_NF4: dict(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    ),
    QuantizationScheme.BNB_FP4: dict(
        load_in_4bit=True,
        bnb_4bit_quant_type="fp4",
        bnb_4bit_use_double_quant=True,
    ),
    QuantizationScheme.BNB_INT8: dict(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    ),
}


def _bnb_spec(description: str) -> SchemeSpec:
    return SchemeSpec(
        default_algorithm=QuantizationAlgorithm.BNB,
        allowed_algorithms=(QuantizationAlgorithm.BNB,),
        needs_calibration_default=False,
        min_compute_capability=7.0,
        description=description,
    )


class BitsAndBytesQuantization(BaseQuantization):
    """BitsAndBytes backend (NF4, FP4, INT8 via ``transformers.BitsAndBytesConfig``)."""

    backend: ClassVar[QuantizationBackend] = QuantizationBackend.BNB

    schemes: ClassVar[dict[QuantizationScheme, SchemeSpec]] = {
        QuantizationScheme.BNB_NF4: _bnb_spec(
            "BitsAndBytes NormalFloat 4-bit quantization"
        ),
        QuantizationScheme.BNB_FP4: _bnb_spec(
            "BitsAndBytes FloatingPoint 4-bit quantization"
        ),
        QuantizationScheme.BNB_INT8: _bnb_spec("BitsAndBytes LLM.int8() quantization"),
    }

    def __init__(self):
        self._bnb_available = importlib.util.find_spec("bitsandbytes") is not None

    @override
    def raise_if_requirements_not_met(self) -> None:
        if not self._bnb_available:
            raise RuntimeError(
                "BitsAndBytes quantization requires bitsandbytes library.\n"
                "Install with: pip install bitsandbytes"
            )
        import bitsandbytes  # type: ignore

        version = getattr(bitsandbytes, "__version__", None)
        if version:
            logger.info(f"BitsAndBytes library found: {version}")
        else:
            logger.info("BitsAndBytes library found (version unknown)")

    @override
    def quantize(self, config: QuantizationConfig) -> QuantizationResult:
        from transformers import BitsAndBytesConfig

        scheme = cast(QuantizationScheme, config.scheme)
        spec = self.schemes[scheme]
        # Validate scheme×algorithm combination (algorithm always BNB here).
        spec.resolve_algorithm(cast(QuantizationAlgorithm, config.algorithm))

        warn_if_local_gpu_below_inference_capability(scheme, spec.min_compute_capability)
        assert_output_path_writable(config.output_path)

        logger.info(f"Starting BitsAndBytes {scheme.value} quantization...")
        bnb_cfg = BitsAndBytesConfig(
            bnb_4bit_compute_dtype=torch.float16, **_BNB_KWARGS[scheme]
        )
        model, tokenizer = load_model_and_tokenizer(
            config,
            torch_dtype=torch.float16,
            quantization_config=bnb_cfg,
        )

        output_dir = Path(config.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving quantized model to: {output_dir}")
        model.save_pretrained(str(output_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(output_dir))

        size = get_directory_size(str(output_dir))
        logger.info(f"Quantization complete. Output size: {format_size(size)}")
        return QuantizationResult(
            output_path=str(output_dir),
            backend=self.backend,
            scheme=scheme,
            format_type=config.output_format,
            quantized_size_bytes=size,
        )
