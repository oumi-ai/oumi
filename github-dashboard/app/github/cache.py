from __future__ import annotations

import hashlib
import json
import logging
from functools import wraps
from pathlib import Path
import diskcache
from cachetools import TTLCache

logger = logging.getLogger(__name__)

_MEM_CACHE: TTLCache = TTLCache(maxsize=256, ttl=60)
_DISK_CACHE: diskcache.Cache | None = None


def init_cache(cache_dir: str) -> None:
    global _DISK_CACHE
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    _DISK_CACHE = diskcache.Cache(cache_dir)


def _make_key(prefix: str, args: tuple, kwargs: dict) -> str:
    payload = json.dumps({"a": args, "k": sorted(kwargs.items())}, default=str, sort_keys=True)
    h = hashlib.md5(payload.encode()).hexdigest()[:12]
    return f"{prefix}:{h}"


def cached(mem_ttl: int = 60, disk_ttl: int = 300):
    """Decorator for async methods: checks memory → disk → calls real fn."""
    def decorator(fn):
        @wraps(fn)
        async def wrapper(self, *args, **kwargs):
            key = _make_key(fn.__name__, args, kwargs)

            if key in _MEM_CACHE:
                return _MEM_CACHE[key]

            if _DISK_CACHE is not None and key in _DISK_CACHE:
                val = _DISK_CACHE[key]
                _MEM_CACHE[key] = val
                return val

            result = await fn(self, *args, **kwargs)

            try:
                _MEM_CACHE.ttl = mem_ttl  # type: ignore[attr-defined]
                _MEM_CACHE[key] = result
            except Exception:
                pass

            if _DISK_CACHE is not None:
                _DISK_CACHE.set(key, result, expire=disk_ttl)

            return result

        return wrapper
    return decorator
