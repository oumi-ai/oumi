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

from dataclasses import dataclass, field
from enum import Enum

from oumi.core.configs.base_config import BaseConfig
from oumi.core.configs.params.model_params import ModelParams
from oumi.exceptions import OumiConfigError


class QuantizationBackend(str, Enum):
    """Quantization library/backend selection (internal)."""

    LLM_COMPRESSOR = "llm_compressor"
    BNB = "bnb"


class QuantizationScheme(str, Enum):
    """Compression scheme for quantization.

    LLM Compressor schemes are passed directly to LLM Compressor modifiers.
    BnB schemes (prefixed with ``bnb_``) map to BitsAndBytes quantization types.
    """

    # LLM Compressor schemes
    FP8_DYNAMIC = "fp8_dynamic"
    FP8_BLOCK = "fp8_block"
    W4A16 = "w4a16"
    W4A16_ASYM = "w4a16_asym"
    W8A16 = "w8a16"

    # BitsAndBytes schemes
    BNB_NF4 = "bnb_nf4"
    BNB_FP4 = "bnb_fp4"
    BNB_INT8 = "bnb_int8"


class QuantizationAlgorithm(str, Enum):
    """Quantization algorithm selection.

    AUTO defers to the chosen scheme's ``default_algorithm`` declared on the
    owning backend's :class:`~oumi.quantize.base.SchemeSpec`.
    """

    AUTO = "auto"
    RTN = "rtn"
    GPTQ = "gptq"
    AWQ = "awq"
    BNB = "bnb"


def _coerce_enum(value, enum_cls: type[Enum], label: str):
    """Coerce a plain string to an enum member, raising OumiConfigError on mismatch.

    Accepts both enum values (e.g. 'fp8_dynamic') and enum names (e.g. 'FP8_DYNAMIC'),
    since OmegaConf serializes enum fields by name when saving to YAML.
    """
    if isinstance(value, str) and not isinstance(value, enum_cls):
        # Try by value first (e.g. 'fp8_dynamic'), then by name (e.g. 'FP8_DYNAMIC').
        try:
            return enum_cls(value)
        except ValueError:
            pass
        try:
            return enum_cls[value]
        except KeyError:
            pass
        raise OumiConfigError(
            f"Unsupported {label}: {value}. "
            f"Must be one of: {[e.value for e in enum_cls]}."
        )
    return value


@dataclass
class QuantizationConfig(BaseConfig):
    """Configuration for model quantization via LLM Compressor or BitsAndBytes.

    Reduces model size by converting weights from higher precision to lower
    precision formats, optimized for deployment with vLLM.

    The backend is inferred automatically from the scheme: schemes prefixed
    with ``bnb_`` use BitsAndBytes, all others use LLM Compressor.

    Example:
        >>> config = QuantizationConfig(
        ...     model=ModelParams(model_name="meta-llama/Llama-3.1-8B-Instruct"),
        ...     scheme=QuantizationScheme.FP8_DYNAMIC,
        ...     output_path="llama3-8b-fp8"
        ... )
    """

    model: ModelParams = field(default_factory=ModelParams)
    """Model to quantize. Supports HuggingFace IDs, local paths, or Oumi models."""

    scheme: str | None = None
    """Compression scheme (required).
    LLM Compressor: fp8_dynamic, fp8_block, w4a16, w4a16_asym, w8a16.
    BitsAndBytes: bnb_nf4, bnb_fp4, bnb_int8.
    Accepts enum values or names; coerced in __post_init__."""

    output_path: str = "quantized_model"
    """Output directory for the quantized model."""

    output_format: str = "safetensors"
    """Output format: 'safetensors'."""

    # --- Algorithm control ---

    algorithm: str = QuantizationAlgorithm.AUTO.value
    """Compression algorithm: 'auto', 'rtn', 'gptq', 'awq', 'bnb'.
    'auto' selects the best algorithm for the chosen scheme.
    Accepts enum values (auto) or names (AUTO); coerced in __post_init__."""

    verbose: bool = False
    """Enable detailed progress logging."""

    ignore_layers: list[str] = field(default_factory=lambda: ["lm_head"])
    """Layer name patterns to exclude from quantization (regex supported)."""

    # --- Calibration settings ---

    calibration_dataset: str = "HuggingFaceH4/ultrachat_200k"
    """HuggingFace dataset ID for calibration data."""

    calibration_split: str = "train_sft"
    """Dataset split to use for calibration."""

    calibration_samples: int = 512
    """Number of calibration samples. 512 (balanced), 128 (faster), 1024 (accurate)."""

    max_seq_length: int = 2048
    """Max sequence length for calibration tokenization."""

    # --- Algorithm-specific settings ---

    group_size: int = 128
    """Weight grouping size for GPTQ/AWQ (64/128/256)."""

    dampening_frac: float = 0.1
    """GPTQ dampening fraction."""

    # --- Output control ---

    save_compressed: bool = True
    """Save in compressed-tensors format for optimized vLLM serving."""

    # --- Computed (non-init) fields ---

    backend: QuantizationBackend = field(
        init=False, default=QuantizationBackend.LLM_COMPRESSOR
    )
    """Quantization backend, inferred from scheme. Not user-settable."""

    def __post_init__(self):
        """Post-initialization validation.

        Coerces ``scheme`` and ``algorithm`` to their enum types, infers
        ``backend`` from the scheme's owning quantization backend, and
        validates the algorithm × scheme combination via the backend's
        :class:`SchemeSpec`.
        """
        # Lazy import: oumi.quantize imports backends which reference this
        # module's QuantizationConfig at type-check time.
        from oumi.quantize import backend_for_scheme

        if self.scheme is None:
            raise OumiConfigError(
                "Quantization scheme is required. "
                f"Must be one of: {[s.value for s in QuantizationScheme]}."
            )
        self.scheme = _coerce_enum(self.scheme, QuantizationScheme, "scheme")
        self.algorithm = _coerce_enum(
            self.algorithm, QuantizationAlgorithm, "algorithm"
        )

        backend_cls = backend_for_scheme(self.scheme)
        self.backend = backend_cls.backend
        # Resolve AUTO to the scheme default and validate the combination.
        self.algorithm = backend_cls.schemes[self.scheme].resolve_algorithm(
            self.algorithm
        )

        if self.output_format != backend_cls.output_format:
            raise OumiConfigError(
                f"Backend {backend_cls.backend.value!r} only supports output "
                f"format {backend_cls.output_format!r}, got {self.output_format!r}."
            )

    def __finalize_and_validate__(self) -> None:
        """Re-infer computed fields after config deserialization.

        OmegaConf.to_object() bypasses __post_init__, so computed fields
        like ``backend`` (derived from ``scheme``) need to be re-set here.
        """
        self.__post_init__()
