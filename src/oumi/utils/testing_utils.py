"""Testing utilities for oumi tests."""

import pytest


def requires_gpus(n: int):
    """Decorator to skip tests if not enough GPUs are available.

    Args:
        n: Number of GPUs required.

    Returns:
        pytest decorator that skips the test if condition is not met.
    """
    try:
        import jax

        return pytest.mark.skipif(jax.device_count() < n, reason=f"requires {n} GPUs")
    except ImportError:
        return pytest.mark.skipif(True, reason="JAX not available")
