"""Optional redis.asyncio adapter.

Lazy-imports ``redis`` so that the package works without redis installed.
Entries are stored as JSON under a key prefix.  TTL is enforced natively by
Redis via ``SET ... EX``.

Install: ``pip install fusion-cache[redis]`` or ``pip install redis``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

from .base import Store

try:
    import redis.asyncio as aioredis
except ImportError:  # pragma: no cover
    aioredis = None


class RedisStore(Store):
    """Async Redis-backed store.

    All read/write operations must run inside an event loop with a connected
    ``redis.asyncio`` client.  In the synchronous pipeline the wrapper uses
    these via the loop; standalone use is expected from async code.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        prefix: str = "fusion-cache:",
        client: Optional[Any] = None,
    ) -> None:
        if aioredis is None:
            raise RuntimeError("redis is not installed; run `pip install fusion-cache[redis]`")
        self.url = url
        self.prefix = prefix
        self._client = client
        self._owns_client = client is None

    @property
    def client(self) -> Any:
        if self._client is None:
            if aioredis is None:
                raise RuntimeError("redis is not installed; run `pip install fusion-cache[redis]`")
            self._client = aioredis.from_url(self.url)
        return self._client

    def _k(self, key: str) -> str:
        return f"{self.prefix}{key}"

    async def aget(self, key: str) -> Optional[Dict[str, Any]]:
        raw = await self.client.get(self._k(key))
        if raw is None:
            return None
        return _decode_value(json.loads(raw))

    async def aset(self, key: str, value: Dict[str, Any], ttl: Optional[float] = None) -> None:
        raw = json.dumps(_encode_value(value), ensure_ascii=False)
        if ttl is not None and ttl > 0:
            await self.client.set(self._k(key), raw, ex=int(ttl))
        else:
            await self.client.set(self._k(key), raw)

    async def adelete(self, key: str) -> bool:
        deleted = await self.client.delete(self._k(key))
        return deleted > 0

    async def akeys(self) -> List[str]:
        cursor = 0
        found: List[str] = []
        while True:
            cursor, keys = await self.client.scan(cursor, match=f"{self.prefix}*", count=100)
            found.extend(k.decode() if isinstance(k, bytes) else k for k in keys)
            if cursor == 0:
                break
        return [k[len(self.prefix):] for k in found]

    async def aclear(self) -> None:
        keys = await self.akeys()
        if keys:
            await self.client.delete(*[self._k(k) for k in keys])

    async def asize(self) -> int:
        return len(await self.akeys())

    # --- synchronous protocol (best-effort bridge) ------------------------
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            loop = asyncio_get_event_loop()
        except RuntimeError:
            loop = None
        if loop is not None and loop.is_running():
            raise RuntimeError(
                "RedisStore is async-first; use aget/aset inside an async context "
                "(the FusionCache wrapper drives it via the loop)."
            )
        return self._run(self.aget(key))

    def set(self, key: str, value: Dict[str, Any], ttl: Optional[float] = None) -> None:
        self._run(self.aset(key, value, ttl))

    def delete(self, key: str) -> bool:
        return self._run(self.adelete(key))

    def keys(self) -> Iterable[str]:
        return self._run(self.akeys())

    def clear(self) -> None:
        self._run(self.aclear())

    def size(self) -> int:
        return self._run(self.asize())

    def _run(self, coro: Any) -> Any:
        import asyncio

        return asyncio.run(coro)

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None


def asyncio_get_event_loop():
    import asyncio

    try:
        return asyncio.get_event_loop()
    except Exception:
        return None


def _encode_value(value: Any) -> Any:
    if _is_buffered_stream(value):
        return {
            "__fusion_cache_type__": "BufferedStream",
            "chunks": _encode_value(value.chunks),
            "usage": _encode_value(value.usage),
        }
    if hasattr(value, "model_dump"):
        return _encode_value(value.model_dump(exclude_none=False))
    if isinstance(value, dict):
        return {str(k): _encode_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode_value(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _decode_value(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get("__fusion_cache_type__") == "BufferedStream":
            from fusion_cache.core.pipeline import BufferedStream

            return BufferedStream(
                chunks=_decode_value(value.get("chunks", [])),
                usage=_decode_value(value.get("usage", {})),
            )
        return {k: _decode_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode_value(v) for v in value]
    return value


def _is_buffered_stream(value: Any) -> bool:
    return value.__class__.__name__ == "BufferedStream" and hasattr(value, "chunks") and hasattr(value, "usage")
