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

"""llama.cpp (GGUF) quantization backend.

Three-stage subprocess pipeline:

  1. ``convert_hf_to_gguf.py`` — HuggingFace safetensors → fp16 GGUF.
  2. ``llama-imatrix`` (k-quants only) — compute importance matrix from a
     calibration corpus.
  3. ``llama-quantize`` — fp16 GGUF → target Q* GGUF, optionally with the
     imatrix to improve k-quant quality.

Binaries (``llama-quantize``, ``llama-imatrix``) must be on ``PATH`` —
typically via ``brew install llama.cpp`` on Mac or a source build on Linux.
The convert script is bootstrapped automatically (see ``_bootstrap.py``).
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import ClassVar, cast

from typing_extensions import override

from oumi.core.configs import QuantizationConfig
from oumi.core.configs.quantization_config import (
    QuantizationAlgorithm,
    QuantizationBackend,
    QuantizationScheme,
)
from oumi.quantize._bootstrap import (
    convert_script_path,
    ensure_llamacpp_python_tools,
    gguf_py_path,
)
from oumi.quantize.base import BaseQuantization, QuantizationResult, SchemeSpec
from oumi.quantize.utils import (
    assert_output_path_writable,
    format_size,
    get_directory_size,
    run_subprocess,
    warn_if_local_gpu_below_inference_capability,
)
from oumi.utils.logging import logger

# llama-quantize accepts the canonical name; QuantizationScheme.value is
# lowercase, so we map to the upstream (uppercase) form here.
_SCHEME_TO_LLAMACPP_NAME: dict[QuantizationScheme, str] = {
    QuantizationScheme.Q4_0: "Q4_0",
    QuantizationScheme.Q4_K_M: "Q4_K_M",
    QuantizationScheme.Q5_K_M: "Q5_K_M",
    QuantizationScheme.Q6_K: "Q6_K",
    QuantizationScheme.Q8_0: "Q8_0",
}


def _spec(*, needs_calib: bool, description: str) -> SchemeSpec:
    return SchemeSpec(
        default_algorithm=QuantizationAlgorithm.LLAMACPP,
        allowed_algorithms=(QuantizationAlgorithm.LLAMACPP,),
        needs_calibration_default=needs_calib,
        # GGUF inference works on CPU + most GPUs; no SM threshold.
        min_compute_capability=0.0,
        description=description,
    )


class LlamaCppQuantization(BaseQuantization):
    """GGUF backend (Q4_0, Q4_K_M, Q5_K_M, Q6_K, Q8_0 via llama.cpp tools)."""

    backend: ClassVar[QuantizationBackend] = QuantizationBackend.LLAMACPP
    output_format: ClassVar[str] = "gguf"

    schemes: ClassVar[dict[QuantizationScheme, SchemeSpec]] = {
        QuantizationScheme.Q4_0: _spec(
            needs_calib=False, description="GGUF 4-bit (legacy, no imatrix)"
        ),
        QuantizationScheme.Q4_K_M: _spec(
            needs_calib=True, description="GGUF 4-bit k-quant medium (imatrix)"
        ),
        QuantizationScheme.Q5_K_M: _spec(
            needs_calib=True, description="GGUF 5-bit k-quant medium (imatrix)"
        ),
        QuantizationScheme.Q6_K: _spec(
            needs_calib=True, description="GGUF 6-bit k-quant (imatrix)"
        ),
        QuantizationScheme.Q8_0: _spec(
            needs_calib=False,
            description="GGUF 8-bit (near-lossless, no imatrix)",
        ),
    }

    @override
    def raise_if_requirements_not_met(self) -> None:
        missing = [
            b for b in ("llama-quantize", "llama-imatrix") if shutil.which(b) is None
        ]
        if missing:
            raise RuntimeError(
                f"GGUF quantization requires llama.cpp binaries on PATH. "
                f"Missing: {missing}.\n"
                "Install with:\n"
                "  brew install llama.cpp        # macOS\n"
                "  apt install llama-cpp         # Linux (or build from source)\n"
                "Or build from source: https://github.com/ggml-org/llama.cpp"
            )

    @override
    def quantize(self, config: QuantizationConfig) -> QuantizationResult:
        scheme = cast(QuantizationScheme, config.scheme)
        spec = self.schemes[scheme]
        algorithm = spec.resolve_algorithm(
            cast(QuantizationAlgorithm, config.algorithm)
        )
        needs_imatrix = spec.needs_calibration_for(algorithm)

        warn_if_local_gpu_below_inference_capability(
            scheme, spec.min_compute_capability
        )
        assert_output_path_writable(config.output_path)
        out_dir = Path(config.output_path)

        llamacpp_root = ensure_llamacpp_python_tools()

        # Stage 1: HF safetensors → fp16 GGUF.
        fp16_path = out_dir / "model-fp16.gguf"
        self._run_convert(config, fp16_path, llamacpp_root)

        # Stage 2: optional importance matrix.
        imatrix_path: Path | None = None
        if needs_imatrix:
            imatrix_path = out_dir / "imatrix.gguf"
            self._run_imatrix(config, fp16_path, imatrix_path)

        # Stage 3: target quantization.
        target_path = out_dir / f"model-{scheme.name}.gguf"
        self._run_quantize(scheme, fp16_path, target_path, imatrix_path)

        # Clean up the fp16 intermediate (it's typically 2-4× the final size).
        fp16_path.unlink(missing_ok=True)
        if imatrix_path is not None:
            imatrix_path.unlink(missing_ok=True)

        size = get_directory_size(str(out_dir))
        logger.info(f"GGUF quantization complete. Output size: {format_size(size)}")
        return QuantizationResult(
            output_path=str(out_dir),
            backend=self.backend,
            scheme=scheme,
            format_type=self.output_format,
            quantized_size_bytes=size,
        )

    # ------------------------------------------------------------------ stages

    def _run_convert(
        self,
        config: QuantizationConfig,
        fp16_path: Path,
        llamacpp_root: Path,
    ) -> None:
        """Stage 1: HF safetensors → fp16 GGUF."""
        from huggingface_hub import snapshot_download

        # Resolve the source: HF repo id → local snapshot, or pass through a
        # local directory.
        source = config.model.model_name
        src_path = Path(source)
        if not src_path.is_dir():
            logger.info(f"Downloading {source} from Hugging Face...")
            source = snapshot_download(
                repo_id=source,
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.model",
                    "tokenizer*",
                    "*.txt",
                ],
            )

        env = os.environ.copy()
        # Point the convert script at the bootstrapped gguf-py.
        gguf_dir = str(gguf_py_path(llamacpp_root))
        env["PYTHONPATH"] = (
            gguf_dir + os.pathsep + env.get("PYTHONPATH", "")
        ).rstrip(os.pathsep)

        run_subprocess(
            [
                sys.executable,
                str(convert_script_path(llamacpp_root)),
                source,
                "--outfile",
                str(fp16_path),
                "--outtype",
                "f16",
            ],
            log_prefix="convert_hf_to_gguf",
            env=env,
        )

    def _run_imatrix(
        self,
        config: QuantizationConfig,
        fp16_path: Path,
        imatrix_path: Path,
    ) -> None:
        """Stage 2: build importance matrix from the calibration corpus."""
        corpus_path = imatrix_path.parent / "calibration.txt"
        self._write_calibration_corpus(config, corpus_path)

        run_subprocess(
            [
                "llama-imatrix",
                "-m",
                str(fp16_path),
                "-f",
                str(corpus_path),
                "-o",
                str(imatrix_path),
                "--chunks",
                str(min(config.calibration_samples, 64)),
                "-c",
                str(min(config.max_seq_length, 512)),
                "--no-ppl",
            ],
            log_prefix="llama-imatrix",
        )
        corpus_path.unlink(missing_ok=True)

    def _run_quantize(
        self,
        scheme: QuantizationScheme,
        fp16_path: Path,
        target_path: Path,
        imatrix_path: Path | None,
    ) -> None:
        """Stage 3: quantize fp16 GGUF to the target type."""
        cmd = ["llama-quantize"]
        if imatrix_path is not None:
            cmd += ["--imatrix", str(imatrix_path)]
        cmd += [
            str(fp16_path),
            str(target_path),
            _SCHEME_TO_LLAMACPP_NAME[scheme],
        ]
        run_subprocess(cmd, log_prefix="llama-quantize")

    # --------------------------------------------------------------- calibration

    def _write_calibration_corpus(
        self, config: QuantizationConfig, corpus_path: Path
    ) -> None:
        """Stream calibration samples to a plaintext file for llama-imatrix."""
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
                f"Recognized names: {list(recognized)}."
            )
        logger.info(f"Using calibration text column: '{text_column}'")

        def _flatten(value) -> str:
            # Handles chat-style list-of-dicts: extract content fields.
            if isinstance(value, list):
                parts: list[str] = []
                for item in value:
                    if isinstance(item, dict) and "content" in item:
                        parts.append(str(item["content"]))
                    else:
                        parts.append(str(item))
                return "\n".join(parts)
            return str(value)

        with corpus_path.open("w", encoding="utf-8") as f:
            for row in ds:
                text = _flatten(row[text_column]).strip()  # type: ignore[index]
                if text:
                    f.write(text)
                    f.write("\n\n")
