import asyncio
import re
from unittest.mock import Mock

import pytest

from oumi.core.async_utils import safe_asyncio_run


def test_safe_asyncio_run_nested_safe():
    async def nested():
        return 1

    def method_using_asyncio():
        return asyncio.run(nested())

    def method_using_safe_asyncio_run():
        return safe_asyncio_run(nested())

    with pytest.raises(
        RuntimeError,
        match=re.escape("asyncio.run() cannot be called from a running event loop"),
    ):

        async def main_async():
            return method_using_asyncio()

        # This will raise a RuntimeError because we are trying to run an async function
        # inside a running event loop.
        asyncio.run(main_async())

    async def safe_main():
        return method_using_safe_asyncio_run()

    # Verify using safe_asyncio_run within another safe_asyncio_run context.
    result = safe_asyncio_run(safe_main())
    assert result == 1


def test_safe_asyncio_run_nested_unsafe():
    async def nested():
        return 1

    def method_using_safe_asyncio():
        return safe_asyncio_run(nested())

    async def main():
        return method_using_safe_asyncio()

    # Here we run asyncio.run() at the top level, where the sub-loop is using
    # safe_asyncio_run.
    result = asyncio.run(main())
    assert result == 1


def test_safe_asyncio_run_nested_fails():
    def method_using_asyncio():
        coro = Mock()
        return asyncio.run(coro)

    async def main():
        return method_using_asyncio()

    # Here we run safe_asyncio_run at the top level, where the sub-loop is using
    # asyncio.run(). This will throw an exception as the new loop from safe_run_asyncio
    # is running in the same context as the asyncio.run() call.
    with pytest.raises(
        RuntimeError,
        match=re.escape("asyncio.run() cannot be called from a running event loop"),
    ):
        _ = safe_asyncio_run(main())
