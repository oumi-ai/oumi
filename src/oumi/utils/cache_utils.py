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

"""Cache utilities for handling functions with unhashable arguments."""

import functools
from typing import Any, Callable, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])


def make_hashable(obj: Any) -> Any:
    """Recursively convert unhashable objects to hashable ones.

    Converts:
    - dict -> frozenset of (key, value) pairs
    - list -> tuple
    - set -> frozenset
    - Other types remain unchanged

    Args:
        obj: Object to convert

    Returns:
        Hashable version of the object

    Examples:
        >>> make_hashable({'a': 1, 'b': 2})
        frozenset({('a', 1), ('b', 2)})

        >>> make_hashable({'nested': {'key': 'value'}})
        frozenset({('nested', frozenset({('key', 'value')}))})
    """
    if isinstance(obj, dict):
        return frozenset((k, make_hashable(v)) for k, v in sorted(obj.items()))
    elif isinstance(obj, list):
        return tuple(make_hashable(item) for item in obj)
    elif isinstance(obj, set):
        return frozenset(make_hashable(item) for item in obj)
    elif isinstance(obj, tuple):
        return tuple(make_hashable(item) for item in obj)
    else:
        # For hashable types (str, int, float, bool, None, etc.)
        return obj


def dict_cache(func: F) -> F:
    """Cache decorator that handles unhashable arguments like dictionaries.

    This decorator works like @functools.cache but can handle dictionaries
    and other unhashable types in the function arguments by converting them
    to hashable equivalents.

    Args:
        func: Function to cache

    Returns:
        Cached version of the function

    Example:
        >>> @dict_cache
        ... def process_config(name: str, config: dict):
        ...     print(f"Processing {name} with {config}")
        ...     return len(config)

        >>> # First call executes the function
        >>> process_config("test", {"key": "value"})
        Processing test with {'key': 'value'}
        1

        >>> # Second call with same args returns cached result
        >>> process_config("test", {"key": "value"})
        1
    """
    cache = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Convert args to hashable form
        hashable_args = tuple(make_hashable(arg) for arg in args)

        # Convert kwargs to hashable form
        hashable_kwargs = tuple(
            (k, make_hashable(v)) for k, v in sorted(kwargs.items())
        )

        # Create cache key
        cache_key = (hashable_args, hashable_kwargs)

        # Check cache
        if cache_key not in cache:
            cache[cache_key] = func(*args, **kwargs)

        return cache[cache_key]

    # Add cache control methods similar to functools.lru_cache
    wrapper.cache_clear = lambda: cache.clear()  # type: ignore
    wrapper.cache_info = lambda: f"CacheInfo(hits={len(cache)}, size={len(cache)})"  # type: ignore

    return cast(F, wrapper)


def dict_lru_cache(maxsize: int = 128) -> Callable[[F], F]:
    """LRU cache decorator that handles unhashable arguments like dictionaries.

    This decorator works like @functools.lru_cache but can handle dictionaries
    and other unhashable types in the function arguments.

    Args:
        maxsize: Maximum size of cache. If None, cache can grow without bound.

    Returns:
        Decorator function

    Example:
        >>> @dict_lru_cache(maxsize=32)
        ... def process_config(name: str, config: dict):
        ...     return len(config)
    """

    def decorator(func: F) -> F:
        # Create a wrapper that converts args to hashable JSON strings
        @functools.lru_cache(maxsize=maxsize)
        def cached_wrapper(hashable_args: tuple, hashable_kwargs: tuple):
            # Reconstruct the original args and kwargs
            # (In practice, we just pass through to the original function)
            return func(
                *cached_wrapper._original_args,  # type: ignore
                **cached_wrapper._original_kwargs,  # type: ignore
            )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Store original args/kwargs for the cached function to use
            cached_wrapper._original_args = args  # type: ignore
            cached_wrapper._original_kwargs = kwargs  # type: ignore

            # Convert to hashable form
            hashable_args = tuple(make_hashable(arg) for arg in args)
            hashable_kwargs = tuple(
                (k, make_hashable(v)) for k, v in sorted(kwargs.items())
            )

            return cached_wrapper(hashable_args, hashable_kwargs)

        # Expose cache control methods
        wrapper.cache_clear = cached_wrapper.cache_clear  # type: ignore
        wrapper.cache_info = cached_wrapper.cache_info  # type: ignore

        return cast(F, wrapper)

    return decorator
