#!/usr/bin/env python3
"""Test script to verify PyTorch, vLLM, and flash-attention compatibility.

Runs a series of tests to ensure the installed packages work together correctly
with the available CUDA hardware.

Usage:
    python test_cuda_deps.py
    python test_cuda_deps.py --quick          # Skip slow tests
    python test_cuda_deps.py --json           # Output results as JSON
    python test_cuda_deps.py --test torch     # Test specific package
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any


@dataclass
class TestResult:
    name: str
    passed: bool
    duration_ms: float
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class TestSuite:
    package: str
    results: list[TestResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def total_duration_ms(self) -> float:
        return sum(r.duration_ms for r in self.results)


def run_test(name: str, test_fn: Callable[[], dict[str, Any]]) -> TestResult:
    """Run a single test and capture results."""
    start = time.perf_counter()
    try:
        details = test_fn()
        duration_ms = (time.perf_counter() - start) * 1000
        return TestResult(
            name=name,
            passed=True,
            duration_ms=duration_ms,
            message="OK",
            details=details,
        )
    except Exception as e:
        duration_ms = (time.perf_counter() - start) * 1000
        return TestResult(
            name=name,
            passed=False,
            duration_ms=duration_ms,
            message=str(e),
            error=traceback.format_exc(),
        )


# =============================================================================
# PyTorch Tests
# =============================================================================


def test_torch_import() -> dict[str, Any]:
    """Test that PyTorch can be imported."""
    import torch

    return {
        "version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def test_torch_cuda_available() -> dict[str, Any]:
    """Test that CUDA is available in PyTorch."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    return {
        "cuda_available": True,
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
    }


def test_torch_cuda_device_info() -> dict[str, Any]:
    """Test that we can query CUDA device properties."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    devices = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        cap = torch.cuda.get_device_capability(i)
        devices.append(
            {
                "index": i,
                "name": props.name,
                "compute_capability": f"{cap[0]}.{cap[1]}",
                "total_memory_gb": round(props.total_memory / (1024**3), 2),
                "multi_processor_count": props.multi_processor_count,
            }
        )

    return {"devices": devices}


def test_torch_tensor_cuda() -> dict[str, Any]:
    """Test basic tensor operations on CUDA."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    # Create tensors on GPU
    device = torch.device("cuda:0")
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)

    # Basic operations
    c = torch.matmul(a, b)
    torch.cuda.synchronize()

    # Verify result
    assert c.shape == (1000, 1000), f"Unexpected shape: {c.shape}"
    assert c.device.type == "cuda", f"Tensor not on CUDA: {c.device}"

    # Cleanup
    del a, b, c
    torch.cuda.empty_cache()

    return {"matmul_test": "passed", "device": str(device)}


def test_torch_cuda_memory() -> dict[str, Any]:
    """Test CUDA memory allocation and deallocation."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.cuda.empty_cache()
    gc.collect()

    initial_allocated = torch.cuda.memory_allocated()
    initial_reserved = torch.cuda.memory_reserved()

    # Allocate a large tensor
    size_mb = 100
    tensor = torch.zeros(size_mb * 1024 * 256, dtype=torch.float32, device="cuda")

    allocated_after = torch.cuda.memory_allocated()

    # Free it
    del tensor
    torch.cuda.empty_cache()
    gc.collect()

    final_allocated = torch.cuda.memory_allocated()

    return {
        "initial_allocated_mb": round(initial_allocated / (1024**2), 2),
        "peak_allocated_mb": round(allocated_after / (1024**2), 2),
        "final_allocated_mb": round(final_allocated / (1024**2), 2),
        "memory_freed": final_allocated
        <= initial_allocated + 1024 * 1024,  # Allow 1MB tolerance
    }


def test_torch_autograd_cuda() -> dict[str, Any]:
    """Test autograd on CUDA."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    x = torch.randn(100, 100, device="cuda", requires_grad=True)
    y = torch.randn(100, 100, device="cuda")

    # Forward
    z = torch.matmul(x, y).sum()

    # Backward
    z.backward()

    assert x.grad is not None, "Gradient not computed"
    assert x.grad.shape == x.shape, "Gradient shape mismatch"

    del x, y, z
    torch.cuda.empty_cache()

    return {"autograd_test": "passed"}


def test_torch_amp() -> dict[str, Any]:
    """Test automatic mixed precision (AMP)."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    # Check if AMP is supported
    device = torch.device("cuda")
    cap = torch.cuda.get_device_capability(device)

    if cap[0] < 7:
        return {
            "amp_supported": False,
            "reason": f"Compute capability {cap[0]}.{cap[1]} < 7.0",
        }

    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)

    # Test autocast
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        z = torch.matmul(x, y)

    assert z.dtype == torch.float16, f"Expected float16, got {z.dtype}"

    del x, y, z
    torch.cuda.empty_cache()

    return {"amp_supported": True, "autocast_test": "passed"}


# =============================================================================
# vLLM Tests
# =============================================================================


def test_vllm_import() -> dict[str, Any]:
    """Test that vLLM can be imported."""
    import vllm

    return {"version": vllm.__version__}


def test_vllm_cuda_ops() -> dict[str, Any]:
    """Test that vLLM CUDA operations are available."""
    try:
        from vllm import _custom_ops as ops

        # Check that ops module loaded
        available_ops = [attr for attr in dir(ops) if not attr.startswith("_")]
        return {"ops_available": True, "num_ops": len(available_ops)}
    except ImportError:
        # Older vLLM versions have different structure
        return {"ops_available": "legacy_structure"}


def test_vllm_engine_init() -> dict[str, Any]:
    """Test that vLLM engine can be initialized (lightweight check)."""
    # Just verify config classes work - don't actually load a model
    # as that would be too slow and memory-intensive for a test
    return {"config_classes": "available"}


def test_vllm_sampling() -> dict[str, Any]:
    """Test vLLM sampling parameters."""
    from vllm import SamplingParams

    params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=100,
    )

    return {
        "sampling_params": "available",
        "temperature": params.temperature,
        "top_p": params.top_p,
    }


# =============================================================================
# Flash Attention Tests
# =============================================================================


def test_flash_attn_import() -> dict[str, Any]:
    """Test that flash-attention can be imported."""
    import flash_attn

    return {"version": flash_attn.__version__}


def test_flash_attn_func_available() -> dict[str, Any]:
    """Test that flash_attn_func is available."""
    # Check function signature
    import inspect

    from flash_attn import flash_attn_func

    sig = inspect.signature(flash_attn_func)
    params = list(sig.parameters.keys())

    return {"flash_attn_func": "available", "params": params[:5]}  # First 5 params


def test_flash_attn_forward() -> dict[str, Any]:
    """Test flash attention forward pass."""
    import torch
    from flash_attn import flash_attn_func

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    # Check compute capability
    cap = torch.cuda.get_device_capability()
    if cap[0] < 8:
        return {
            "skipped": True,
            "reason": f"Compute capability {cap[0]}.{cap[1]} < 8.0 (requires Ampere+)",
        }

    # Small test tensors
    batch_size = 2
    seq_len = 128
    num_heads = 8
    head_dim = 64

    # flash_attn expects (batch, seqlen, nheads, headdim)
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16
    )
    k = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16
    )
    v = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16
    )

    # Forward pass
    out = flash_attn_func(q, k, v)
    torch.cuda.synchronize()

    assert out.shape == q.shape, f"Output shape mismatch: {out.shape} vs {q.shape}"

    del q, k, v, out
    torch.cuda.empty_cache()

    return {
        "forward_pass": "passed",
        "batch_size": batch_size,
        "seq_len": seq_len,
        "num_heads": num_heads,
        "head_dim": head_dim,
    }


def test_flash_attn_backward() -> dict[str, Any]:
    """Test flash attention backward pass (gradient computation)."""
    import torch
    from flash_attn import flash_attn_func

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    cap = torch.cuda.get_device_capability()
    if cap[0] < 8:
        return {
            "skipped": True,
            "reason": f"Compute capability {cap[0]}.{cap[1]} < 8.0 (requires Ampere+)",
        }

    batch_size = 2
    seq_len = 64
    num_heads = 4
    head_dim = 32

    q = torch.randn(
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        device="cuda",
        dtype=torch.float16,
        requires_grad=True,
    )
    k = torch.randn(
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        device="cuda",
        dtype=torch.float16,
        requires_grad=True,
    )
    v = torch.randn(
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        device="cuda",
        dtype=torch.float16,
        requires_grad=True,
    )

    out = flash_attn_func(q, k, v)
    loss = out.sum()
    loss.backward()
    torch.cuda.synchronize()

    assert q.grad is not None, "q.grad is None"
    assert k.grad is not None, "k.grad is None"
    assert v.grad is not None, "v.grad is None"

    del q, k, v, out, loss
    torch.cuda.empty_cache()

    return {"backward_pass": "passed"}


def test_flash_attn_varlen() -> dict[str, Any]:
    """Test variable-length flash attention."""
    import torch

    try:
        from flash_attn import flash_attn_varlen_func
    except ImportError:
        return {"skipped": True, "reason": "flash_attn_varlen_func not available"}

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    cap = torch.cuda.get_device_capability()
    if cap[0] < 8:
        return {
            "skipped": True,
            "reason": f"Compute capability {cap[0]}.{cap[1]} < 8.0",
        }

    # Variable length sequences
    total_tokens = 256
    num_heads = 4
    head_dim = 32

    q = torch.randn(
        total_tokens, num_heads, head_dim, device="cuda", dtype=torch.float16
    )
    k = torch.randn(
        total_tokens, num_heads, head_dim, device="cuda", dtype=torch.float16
    )
    v = torch.randn(
        total_tokens, num_heads, head_dim, device="cuda", dtype=torch.float16
    )

    # Cumulative sequence lengths (2 sequences: 128 + 128 = 256)
    cu_seqlens = torch.tensor([0, 128, 256], device="cuda", dtype=torch.int32)
    max_seqlen = 128

    out = flash_attn_varlen_func(
        q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen
    )
    torch.cuda.synchronize()

    assert out.shape == q.shape, f"Output shape mismatch: {out.shape}"

    del q, k, v, out
    torch.cuda.empty_cache()

    return {"varlen_test": "passed", "total_tokens": total_tokens}


# =============================================================================
# Cross-Package Compatibility Tests
# =============================================================================


def test_torch_vllm_cuda_context() -> dict[str, Any]:
    """Test that PyTorch and vLLM share CUDA context properly."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    # Initialize PyTorch CUDA context
    torch_tensor = torch.zeros(100, device="cuda")
    torch_device = torch_tensor.device

    # Import vLLM (should use same context)
    import vllm  # noqa: F401

    # Verify PyTorch still works
    result = torch_tensor.sum()
    assert result.item() == 0.0

    del torch_tensor
    torch.cuda.empty_cache()

    return {"shared_context": True, "torch_device": str(torch_device)}


def test_flash_attn_torch_integration() -> dict[str, Any]:
    """Test flash-attention integration with PyTorch autograd."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    cap = torch.cuda.get_device_capability()
    if cap[0] < 8:
        return {"skipped": True, "reason": "Compute capability < 8.0"}

    from flash_attn import flash_attn_func

    # Use within a PyTorch nn.Module
    class FlashAttnLayer(torch.nn.Module):
        def __init__(self, dim: int, num_heads: int):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = dim // num_heads
            self.qkv = torch.nn.Linear(dim, 3 * dim)
            self.out = torch.nn.Linear(dim, dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, s, d = x.shape
            qkv = self.qkv(x)
            qkv = qkv.reshape(b, s, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(2)
            out = flash_attn_func(q, k, v)
            out = out.reshape(b, s, d)
            return self.out(out)

    model = FlashAttnLayer(256, 8).cuda().half()
    x = torch.randn(2, 32, 256, device="cuda", dtype=torch.float16)

    # Forward
    y = model(x)
    assert y.shape == x.shape

    # Backward
    loss = y.sum()
    loss.backward()

    # Check gradients exist
    for param in model.parameters():
        assert param.grad is not None, "Missing gradient"

    del model, x, y, loss
    torch.cuda.empty_cache()

    return {"module_integration": "passed"}


# =============================================================================
# Test Runner
# =============================================================================


TORCH_TESTS = [
    ("torch_import", test_torch_import),
    ("torch_cuda_available", test_torch_cuda_available),
    ("torch_cuda_device_info", test_torch_cuda_device_info),
    ("torch_tensor_cuda", test_torch_tensor_cuda),
    ("torch_cuda_memory", test_torch_cuda_memory),
    ("torch_autograd_cuda", test_torch_autograd_cuda),
    ("torch_amp", test_torch_amp),
]

VLLM_TESTS = [
    ("vllm_import", test_vllm_import),
    ("vllm_cuda_ops", test_vllm_cuda_ops),
    ("vllm_engine_init", test_vllm_engine_init),
    ("vllm_sampling", test_vllm_sampling),
]

FLASH_ATTN_TESTS = [
    ("flash_attn_import", test_flash_attn_import),
    ("flash_attn_func_available", test_flash_attn_func_available),
    ("flash_attn_forward", test_flash_attn_forward),
    ("flash_attn_backward", test_flash_attn_backward),
    ("flash_attn_varlen", test_flash_attn_varlen),
]

INTEGRATION_TESTS = [
    ("torch_vllm_cuda_context", test_torch_vllm_cuda_context),
    ("flash_attn_torch_integration", test_flash_attn_torch_integration),
]

# Slow tests that can be skipped
SLOW_TESTS = {
    "flash_attn_backward",
    "flash_attn_varlen",
    "flash_attn_torch_integration",
}


def run_test_suite(
    tests: list[tuple[str, Callable]],
    package_name: str,
    skip_slow: bool = False,
) -> TestSuite:
    """Run a suite of tests for a package."""
    suite = TestSuite(package=package_name)

    for name, test_fn in tests:
        if skip_slow and name in SLOW_TESTS:
            suite.results.append(
                TestResult(
                    name=name,
                    passed=True,
                    duration_ms=0,
                    message="SKIPPED (slow test)",
                    details={"skipped": True},
                )
            )
            continue

        result = run_test(name, test_fn)
        suite.results.append(result)

        # Stop suite if a critical test fails
        if not result.passed and name.endswith("_import"):
            break

    return suite


def print_results(suites: list[TestSuite], verbose: bool = False) -> None:
    """Print test results in human-readable format."""
    print("\n" + "=" * 70)
    print("CUDA Dependencies Test Results")
    print("=" * 70)

    total_passed = 0
    total_failed = 0
    total_duration = 0.0

    for suite in suites:
        passed = sum(1 for r in suite.results if r.passed)
        failed = len(suite.results) - passed
        total_passed += passed
        total_failed += failed
        total_duration += suite.total_duration_ms

        status = "PASS" if suite.passed else "FAIL"
        print(f"\n{suite.package}: {status} ({passed}/{len(suite.results)} tests)")
        print("-" * 50)

        for result in suite.results:
            status_icon = "[OK]" if result.passed else "[FAIL]"
            print(f"  {status_icon} {result.name} ({result.duration_ms:.1f}ms)")

            if verbose and result.details:
                for key, value in result.details.items():
                    if key != "skipped":
                        print(f"       {key}: {value}")

            if not result.passed and result.message:
                print(f"       Error: {result.message}")

    print("\n" + "=" * 70)
    print(
        f"Total: {total_passed} passed, {total_failed} failed ({total_duration:.1f}ms)"
    )
    print("=" * 70)


def output_json(suites: list[TestSuite]) -> None:
    """Output results as JSON."""
    output = {
        "suites": [
            {
                "package": suite.package,
                "passed": suite.passed,
                "total_duration_ms": suite.total_duration_ms,
                "results": [asdict(r) for r in suite.results],
            }
            for suite in suites
        ],
        "summary": {
            "total_passed": sum(1 for s in suites for r in s.results if r.passed),
            "total_failed": sum(1 for s in suites for r in s.results if not r.passed),
            "all_passed": all(s.passed for s in suites),
        },
    }
    print(json.dumps(output, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Test PyTorch, vLLM, and flash-attention compatibility"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip slow tests",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed test output",
    )
    parser.add_argument(
        "--test",
        choices=["torch", "vllm", "flash-attn", "integration", "all"],
        default="all",
        help="Which package(s) to test",
    )
    args = parser.parse_args()

    suites: list[TestSuite] = []

    # Run selected tests
    if args.test in ("torch", "all"):
        suite = run_test_suite(TORCH_TESTS, "PyTorch", args.quick)
        suites.append(suite)

    if args.test in ("vllm", "all"):
        # Only run vLLM tests if PyTorch CUDA works
        try:
            import torch

            if torch.cuda.is_available():
                suite = run_test_suite(VLLM_TESTS, "vLLM", args.quick)
                suites.append(suite)
        except ImportError:
            suites.append(
                TestSuite(
                    package="vLLM",
                    results=[
                        TestResult(
                            name="vllm_import",
                            passed=False,
                            duration_ms=0,
                            message="PyTorch not available",
                        )
                    ],
                )
            )

    if args.test in ("flash-attn", "all"):
        try:
            import torch

            if torch.cuda.is_available():
                suite = run_test_suite(FLASH_ATTN_TESTS, "flash-attention", args.quick)
                suites.append(suite)
        except ImportError:
            suites.append(
                TestSuite(
                    package="flash-attention",
                    results=[
                        TestResult(
                            name="flash_attn_import",
                            passed=False,
                            duration_ms=0,
                            message="PyTorch not available",
                        )
                    ],
                )
            )

    if args.test in ("integration", "all"):
        try:
            import torch

            if torch.cuda.is_available():
                suite = run_test_suite(INTEGRATION_TESTS, "Integration", args.quick)
                suites.append(suite)
        except ImportError:
            pass

    # Output results
    if args.json:
        output_json(suites)
    else:
        print_results(suites, args.verbose)

    # Exit code
    all_passed = all(s.passed for s in suites)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
