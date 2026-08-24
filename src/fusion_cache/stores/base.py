"""Store protocol shared by the memory and redis backends.

A store holds cache entries keyed by string.  Every entry is a dict with at
least ``response`` and ``meta``; the store may expire entries on read
(TTL) and evict by max-size.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional


class Store(ABC):
    """Protocol for the L1 exact store and the L2 semantic store."""

    @abstractmethod
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Return the entry for ``key`` or None (expired entries count as None)."""

    @abstractmethod
    def set(self, key: str, value: Dict[str, Any], ttl: Optional[float] = None) -> None:
        """Store ``value`` under ``key`` with an optional TTL in seconds."""

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Remove ``key``; return True if it existed."""

    @abstractmethod
    def keys(self) -> Iterable[str]:
        """Yield all live keys (used by the semantic scan)."""

    @abstractmethod
    def clear(self) -> None:
        """Drop every entry."""

    @abstractmethod
    def size(self) -> int:
        """Number of live entries."""

    # ---- async variants ----------------------------------------------------
    # The pipeline uses the async variants so a Redis-backed store can run
    # inside an event loop.  The default implementation bridges to the sync
    # methods (fine for in-memory stores).
    async def aget(self, key: str) -> Optional[Dict[str, Any]]:
        return self.get(key)

    async def aset(self, key: str, value: Dict[str, Any], ttl: Optional[float] = None) -> None:
        self.set(key, value, ttl)

    async def adelete(self, key: str) -> bool:
        return self.delete(key)

    async def akeys(self) -> List[str]:
        return list(self.keys())

    async def aclear(self) -> None:
        self.clear()

    async def asize(self) -> int:
        return self.size()

    async def aclose(self) -> None:
        """Release any resources (default: no-op)."""


class InMemoryStoreMixin:
    """Shared TTL logic for in-memory stores."""

    def __init__(self, ttl: Optional[float] = None) -> None:
        self._default_ttl = ttl
        self._data: Dict[str, Any] = {}
        self._expires: Dict[str, float] = {}

    def _is_expired(self, key: str, now: Optional[float] = None) -> bool:
        exp = self._expires.get(key)
        if exp is None:
            return False
        if now is None:
            import time

            now = time.monotonic()
        return now > exp

    def _prune(self) -> None:
        import time

        now = time.monotonic()
        expired = [k for k, e in self._expires.items() if now > e]
        for k in expired:
            self._data.pop(k, None)
            self._expires.pop(k, None)


class MemoryStore(Store, InMemoryStoreMixin):
    """In-memory dict + LRU/TTL store.

    ``max_entries`` bounds the dict; eviction is simple LRU-by-insertion-order
    (approximate LRU, cheap and adequate for the MVP).
    """

    def __init__(self, max_entries: int = 10_000, ttl: Optional[float] = None) -> None:
        super().__init__(ttl=ttl)
        self.max_entries = max_entries
        # _lru holds (key -> monotonic counter) for eviction ordering
        self._lru: Dict[str, float] = {}
        self._clock = 0

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        import time

        now = time.monotonic()
        if self._is_expired(key, now):
            self._data.pop(key, None)
            self._expires.pop(key, None)
            self._lru.pop(key, None)
            return None
        value = self._data.get(key)
        if value is not None:
            self._clock += 1
            self._lru[key] = self._clock
        return value

    def set(self, key: str, value: Dict[str, Any], ttl: Optional[float] = None) -> None:
        import time

        now = time.monotonic()
        if key not in self._data and len(self._data) >= self.max_entries:
            self._evict_one()
        self._data[key] = value
        self._clock += 1
        self._lru[key] = self._clock
        effective_ttl = self._default_ttl if ttl is None else ttl
        if effective_ttl is not None and effective_ttl > 0:
            self._expires[key] = now + effective_ttl
        else:
            self._expires.pop(key, None)

    def _evict_one(self) -> None:
        if not self._data:
            return
        victim = min(self._lru, key=self._lru.get) if self._lru else next(iter(self._data))
        self._data.pop(victim, None)
        self._expires.pop(victim, None)
        self._lru.pop(victim, None)

    def delete(self, key: str) -> bool:
        existed = key in self._data
        self._data.pop(key, None)
        self._expires.pop(key, None)
        self._lru.pop(key, None)
        return existed

    def keys(self) -> List[str]:
        self._prune()
        return list(self._data.keys())

    def clear(self) -> None:
        self._data.clear()
        self._expires.clear()
        self._lru.clear()

    def size(self) -> int:
        self._prune()
        return len(self._data)
