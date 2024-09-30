import asyncio
from multiprocessing.pool import ThreadPool
from typing import Any, Awaitable


def safe_asyncio_run(main: Awaitable[Any]) -> Any:
    """Run a series of Awaitables in a new thread. Blocks until the thread is finished.

    This circumvents the issue of running async functions in the main thread when
    an event loop is already running (Jupyter notebooks, for example).

    Prefer using `safe_asyncio_run` over `asyncio.run` to allow upstream callers to
    ignore our dependency on asyncio.

    Args:
        main: The awaitable to resolve.

    Returns:
        The result of the awaitable.
    """
    pool = ThreadPool(processes=1)
    return pool.apply(asyncio.run, (main,))
