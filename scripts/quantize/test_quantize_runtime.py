#!/usr/bin/env python3
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

"""Exhaustive runtime test for `oumi.quantize`.

Runs every supported (scheme x algorithm) combination end-to-end on a small
model, validates each output directory, exercises the config-level negative
paths, and prints a pass/fail summary.

Requires:
  * A CUDA GPU (any compute capability >= 7.0 will run the quantization
    step; FP8 inference needs SM 8.9 but quantization itself does not).
  * `pip install -e '.[quantization]'` (llmcompressor + bitsandbytes).
  * Hugging Face access. Calibration tests use ``HuggingFaceH4/ultrachat_200k``
    which may require ``huggingface-cli login``. Skip those with
    ``--skip-calibration`` if you don't want the dataset dependency.

Usage:
    # Default: TinyLlama, 32 calibration samples, no cleanup, no reload.
    python scripts/quantize/test_quantize_runtime.py

    # Quick smoke (data-free schemes only, no calibration):
    python scripts/quantize/test_quantize_runtime.py --skip-calibration

    # Full run with output cleanup + reload validation:
    python scripts/quantize/test_quantize_runtime.py --cleanup --reload

    # Single scheme x algorithm:
    python scripts/quantize/test_quantize_runtime.py --only fp8_dynamic:auto

    # List planned tests, don't run anything:
    python scripts/quantize/test_quantize_runtime.py --list-only
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Test matrix
# ---------------------------------------------------------------------------


@dataclass
class Case:
    """One (scheme, algorithm) test case."""

    scheme: str
    algorithm: str  # "auto" or an explicit algorithm name
    needs_calibration: bool
    label: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = f"{self.scheme}:{self.algorithm}"


# Default-algorithm runs (the AUTO path). Covers all 8 schemes.
AUTO_CASES: list[Case] = [
    Case("fp8_dynamic", "auto", needs_calibration=False),
    Case("fp8_block", "auto", needs_calibration=False),
    Case("w4a16", "auto", needs_calibration=True),  # gptq
    Case("w4a16_asym", "auto", needs_calibration=True),  # awq
    Case("w8a16", "auto", needs_calibration=True),  # gptq
    Case("bnb_nf4", "auto", needs_calibration=False),
    Case("bnb_fp4", "auto", needs_calibration=False),
    Case("bnb_int8", "auto", needs_calibration=False),
]

# Explicit algorithm overrides. These exercise the SchemeSpec validation
# path (allowed_algorithms + calibration_required_for overlay).
OVERRIDE_CASES: list[Case] = [
    # FP8_DYNAMIC default is RTN (data-free); GPTQ/AWQ overrides force calibration.
    Case("fp8_dynamic", "gptq", needs_calibration=True),
    Case("fp8_dynamic", "awq", needs_calibration=True),
    # W4A16 default is GPTQ; RTN override still calibrates because the scheme
    # itself sets needs_calibration_default=True.
    Case("w4a16", "rtn", needs_calibration=True),
]


@dataclass
class NegativeCase:
    """Config-level rejection — must raise OumiConfigError at construction time."""

    label: str
    scheme: str
    algorithm: str | None = None
    output_format: str | None = None
    expected_match: str = ""


NEGATIVE_CASES: list[NegativeCase] = [
    NegativeCase(
        label="bnb scheme + gptq algorithm",
        scheme="bnb_nf4",
        algorithm="gptq",
        expected_match="not allowed",
    ),
    NegativeCase(
        label="llmc scheme + bnb algorithm",
        scheme="fp8_dynamic",
        algorithm="bnb",
        expected_match="not allowed",
    ),
    NegativeCase(
        label="bad output_format",
        scheme="fp8_dynamic",
        output_format="gguf",
        expected_match="only supports output format",
    ),
]


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------


@dataclass
class CaseResult:
    case: Case
    passed: bool
    duration_s: float
    output_path: str = ""
    output_size_bytes: int = 0
    error: str = ""
    notes: list[str] = field(default_factory=list)


def _format_size(n: int) -> str:
    s = float(n)
    for u in ("B", "KB", "MB", "GB", "TB"):
        if s < 1024:
            return f"{s:.1f} {u}"
        s /= 1024
    return f"{s:.1f} PB"


def _validate_output_dir(output_path: str) -> tuple[list[str], int]:
    """Sanity-check the saved directory. Returns (notes, total_size_bytes)."""
    p = Path(output_path)
    notes: list[str] = []
    if not p.exists():
        raise FileNotFoundError(f"output directory missing: {output_path}")
    if not p.is_dir():
        raise NotADirectoryError(f"output is not a directory: {output_path}")

    config_json = p / "config.json"
    if not config_json.exists():
        raise FileNotFoundError(f"config.json missing in {output_path}")
    # Confirm it parses.
    with config_json.open() as f:
        cfg_data = json.load(f)
    if "model_type" not in cfg_data:
        notes.append("config.json missing model_type")

    # At least one safetensors shard.
    shards = list(p.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no .safetensors files in {output_path}")
    notes.append(f"{len(shards)} shard(s)")

    # Tokenizer artifacts (any one is enough).
    tok_candidates = [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
    ]
    if not any((p / t).exists() for t in tok_candidates):
        raise FileNotFoundError(f"no tokenizer artifacts in {output_path}")

    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    if total < 1024 * 1024:  # < 1MB suggests a failed save
        raise RuntimeError(f"output suspiciously small: {_format_size(total)}")
    return notes, total


def _try_reload_config(output_path: str) -> str:
    """Reload the saved config (cheap, validates JSON shape)."""
    from transformers import AutoConfig

    AutoConfig.from_pretrained(output_path, trust_remote_code=False)
    return "config reload OK"


def _full_reload(output_path: str) -> str:
    """Reload the quantized model and run a tiny forward pass."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(output_path)
    model = AutoModelForCausalLM.from_pretrained(
        output_path,
        device_map="auto",
        torch_dtype="auto",
    )
    inputs = tok("Hello", return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs)
    note = f"reload OK; logits shape={tuple(out.logits.shape)}"
    del model, tok
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return note


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


def _run_case(
    case: Case,
    *,
    model_name: str,
    output_root: Path,
    calibration_samples: int,
    do_reload: bool,
    cleanup: bool,
) -> CaseResult:
    from oumi import quantize as oumi_quantize
    from oumi.core.configs import QuantizationConfig
    from oumi.core.configs.params.model_params import ModelParams

    output_path = output_root / case.label.replace(":", "_")
    if output_path.exists():
        shutil.rmtree(output_path)

    cfg_kwargs: dict[str, Any] = {
        "model": ModelParams(model_name=model_name),
        "scheme": case.scheme,
        "output_path": str(output_path),
    }
    if case.algorithm != "auto":
        cfg_kwargs["algorithm"] = case.algorithm
    if case.needs_calibration:
        cfg_kwargs["calibration_samples"] = calibration_samples

    cfg = QuantizationConfig(**cfg_kwargs)

    notes: list[str] = []
    t0 = time.perf_counter()
    try:
        result = oumi_quantize(cfg)
        duration = time.perf_counter() - t0

        out_notes, total = _validate_output_dir(str(output_path))
        notes.extend(out_notes)
        # Compare to what oumi reported.
        if abs(total - result.quantized_size_bytes) > 4096:
            notes.append(
                f"size mismatch: dir={_format_size(total)} "
                f"vs result={_format_size(result.quantized_size_bytes)}"
            )

        # Cheap config-only reload always.
        notes.append(_try_reload_config(str(output_path)))
        if do_reload:
            notes.append(_full_reload(str(output_path)))

        passed = True
        error = ""
    except Exception:  # noqa: BLE001 — we want every failure mode reported.
        duration = time.perf_counter() - t0
        passed = False
        error = traceback.format_exc()
        total = 0
    finally:
        if cleanup and output_path.exists():
            shutil.rmtree(output_path, ignore_errors=True)

    return CaseResult(
        case=case,
        passed=passed,
        duration_s=duration,
        output_path=str(output_path),
        output_size_bytes=total,
        error=error,
        notes=notes,
    )


def _run_negative_cases() -> list[tuple[NegativeCase, bool, str]]:
    """Run config-level rejections. No GPU needed."""
    from oumi.core.configs import QuantizationConfig
    from oumi.core.configs.params.model_params import ModelParams
    from oumi.exceptions import OumiConfigError

    results: list[tuple[NegativeCase, bool, str]] = []
    for nc in NEGATIVE_CASES:
        kwargs: dict[str, Any] = {
            "model": ModelParams(model_name="x"),
            "scheme": nc.scheme,
            "output_path": "x",
        }
        if nc.algorithm is not None:
            kwargs["algorithm"] = nc.algorithm
        if nc.output_format is not None:
            kwargs["output_format"] = nc.output_format
        try:
            QuantizationConfig(**kwargs)
            results.append((nc, False, "did not raise"))
        except OumiConfigError as e:
            msg = str(e)
            ok = nc.expected_match in msg
            results.append((nc, ok, msg if not ok else "OK"))
        except Exception as e:  # noqa: BLE001
            results.append(
                (
                    nc,
                    False,
                    f"raised {type(e).__name__} (expected OumiConfigError): {e}",
                )
            )
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument(
        "--model",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HF model id to quantize (default: %(default)s).",
    )
    p.add_argument(
        "--output-dir",
        default="/tmp/oumi-quantize-test",
        help="Root directory for quantized outputs (default: %(default)s).",
    )
    p.add_argument(
        "--calibration-samples",
        type=int,
        default=32,
        help="Calibration samples for GPTQ/AWQ runs (default: %(default)s).",
    )
    p.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip every test case that requires calibration data.",
    )
    p.add_argument(
        "--skip-overrides",
        action="store_true",
        help="Skip the explicit-algorithm-override cases.",
    )
    p.add_argument(
        "--only",
        default="",
        help="Comma-separated list of case labels (e.g. 'fp8_dynamic:auto,bnb_nf4:auto').",
    )
    p.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete each output directory after validation (saves disk).",
    )
    p.add_argument(
        "--reload",
        action="store_true",
        help="Reload the saved model + run a tiny forward pass per case.",
    )
    p.add_argument(
        "--list-only",
        action="store_true",
        help="Print the planned cases and exit.",
    )
    p.add_argument(
        "--no-negative",
        action="store_true",
        help="Skip the config-level negative tests.",
    )
    return p.parse_args()


def _print_env() -> None:
    print("=== Environment ===")
    print(f"python:       {sys.version.split()[0]}")
    try:
        import torch

        print(f"torch:        {torch.__version__}")
        print(f"cuda avail:   {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            print(
                f"gpu:          {torch.cuda.get_device_name(0)} "
                f"(SM {major}.{minor}, count={torch.cuda.device_count()})"
            )
    except Exception as e:  # noqa: BLE001
        print(f"torch:        UNAVAILABLE ({e})")
    for pkg in ("transformers", "llmcompressor", "bitsandbytes", "datasets"):
        try:
            mod = __import__(pkg)
            print(f"{pkg:<13} {getattr(mod, '__version__', '?')}")
        except ImportError:
            print(f"{pkg:<13} NOT INSTALLED")
    try:
        from oumi.quantize import _SCHEME_TO_BACKEND, all_schemes

        print(f"oumi schemes: {sorted(s.value for s in _SCHEME_TO_BACKEND)}")
        assert len(all_schemes()) == len(_SCHEME_TO_BACKEND)
    except Exception as e:  # noqa: BLE001
        print(f"oumi import FAILED: {e}")
    print()


def _filter_cases(cases: list[Case], args: argparse.Namespace) -> list[Case]:
    if args.skip_calibration:
        cases = [c for c in cases if not c.needs_calibration]
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        cases = [c for c in cases if c.label in wanted]
    return cases


def main() -> int:
    args = _parse_args()

    cases: list[Case] = list(AUTO_CASES)
    if not args.skip_overrides:
        cases.extend(OVERRIDE_CASES)
    cases = _filter_cases(cases, args)

    if args.list_only:
        print("Planned cases:")
        for c in cases:
            print(f"  {c.label:<25} calibration={c.needs_calibration}")
        if not args.no_negative:
            print("Negative cases:")
            for nc in NEGATIVE_CASES:
                print(f"  {nc.label}")
        return 0

    _print_env()

    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"Output root: {output_root}")
    print(f"Model:       {args.model}")
    print(f"Cases:       {len(cases)}")
    print()

    results: list[CaseResult] = []
    for i, case in enumerate(cases, start=1):
        print(f"[{i}/{len(cases)}] === {case.label} ===")
        r = _run_case(
            case,
            model_name=args.model,
            output_root=output_root,
            calibration_samples=args.calibration_samples,
            do_reload=args.reload,
            cleanup=args.cleanup,
        )
        results.append(r)
        if r.passed:
            print(
                f"    PASS  {r.duration_s:6.1f}s  "
                f"size={_format_size(r.output_size_bytes)}  "
                f"notes=[{', '.join(r.notes)}]"
            )
        else:
            print(f"    FAIL  {r.duration_s:6.1f}s")
            for line in r.error.splitlines()[-5:]:
                print(f"      {line}")
        print()

        # Free GPU memory between cases.
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    negatives: list[tuple[NegativeCase, bool, str]] = []
    if not args.no_negative:
        print("=== Negative tests ===")
        negatives = _run_negative_cases()
        for nc, ok, info in negatives:
            mark = "PASS" if ok else "FAIL"
            print(f"  {mark}  {nc.label}: {info}")
        print()

    # Summary.
    print("=== Summary ===")
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    total_time = sum(r.duration_s for r in results)
    print(f"Quantize cases: {passed} passed, {failed} failed (of {len(results)})")
    print(f"Total time:     {total_time:.1f}s")
    if negatives:
        n_pass = sum(1 for _, ok, _ in negatives if ok)
        print(f"Negative tests: {n_pass}/{len(negatives)} passed")

    print()
    print(f"{'Case':<28}{'Result':<8}{'Time':<8}Size")
    print("-" * 60)
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        print(
            f"{r.case.label:<28}{mark:<8}"
            f"{r.duration_s:>6.1f}s  {_format_size(r.output_size_bytes)}"
        )

    if failed or any(not ok for _, ok, _ in negatives):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
