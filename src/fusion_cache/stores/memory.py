"""In-memory LRU/TTL store (default backend)."""

from __future__ import annotations

from .base import InMemoryStoreMixin, Store
from typing import Any, Dict, Iterable, List, Optional


class MemoryStore(Store, InMemoryStoreMixin):
    """Thread-safe in-memory store with LRU eviction and per-key TTL.

    The store is a plain dict guarded by a lock; entries expire lazily on
    read/iteration.  Eviction is LRU-by-recent-use using a monotonic clock.
    """

    def __init__(self, max_entries: int = 10_000, ttl: Optional[float] = None) -> None:
        super().__init__(ttl=ttl)
        self.max_entries = max_entries
        self._lru: Dict[str, float] = {}
        self._clock = 0
        import threading

        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._get_unlocked(key)

    def _get_unlocked(self, key: str) -> Optional[Dict[str, Any]]:
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
        with self._lock:
            self._set_unlocked(key, value, ttl)

    def _set_unlocked(self, key: str, value: Dict[str, Any], ttl: Optional[float] = None) -> None:
        import time

        now = time.monotonic()
        if key not in self._data and len(self._data) >= self.max_entries:
            self._evict_one_unlocked()
        self._data[key] = value
        self._clock += 1
        self._lru[key] = self._clock
        effective_ttl = self._default_ttl if ttl is None else ttl
        if effective_ttl is not None and effective_ttl > 0:
            self._expires[key] = now + effective_ttl
        else:
            self._expires.pop(key, None)

    def _evict_one_unlocked(self) -> None:
        if not self._data:
            return
        victim = min(self._lru, key=self._lru.get) if self._lru else next(iter(self._data))
        self._data.pop(victim, None)
        self._expires.pop(victim, None)
        self._lru.pop(victim, None)

    def delete(self, key: str) -> bool:
        with self._lock:
            existed = key in self._data
            self._data.pop(key, None)
            self._expires.pop(key, None)
            self._lru.pop(key, None)
            return existed

    def keys(self) -> List[str]:
        with self._lock:
            self._prune()
            return list(self._data.keys())

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._expires.clear()
            self._lru.clear()

    def size(self) -> int:
        with self._lock:
            self._prune()
            return len(self._data)
