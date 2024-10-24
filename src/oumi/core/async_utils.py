import asyncio
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

T = TypeVar("T")


def safe_asyncio_run(main: Coroutine[Any, Any, T]) -> T:
    """Run an Awaitable in a new thread. Blocks until the thread is finished.

    This circumvents the issue of running async functions in the main thread when
    an event loop is already running (Jupyter notebooks, for example).

    Prefer using `safe_asyncio_run` over `asyncio.run` to allow upstream callers to
    ignore our dependency on asyncio.

    Args:
        main: The Coroutine to resolve.

    Returns:
        The result of the Coroutine.
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        task = executor.submit(asyncio.run, main)
        return task.result()
