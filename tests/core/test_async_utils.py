import asyncio
import re

import pytest

from oumi.core.async_utils import safe_asyncio_run


def test_safe_asyncio_run_nested():
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

    result = safe_asyncio_run(safe_main())
    assert result == 1
