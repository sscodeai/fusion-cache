"""Simple in-memory rate limiter for the gateway.

Sliding-window counter per key (default: API key or client IP).  Configurable
via env: ``FUSION_RATE_LIMIT`` (requests per minute, 0 = disabled) and
``FUSION_RATE_LIMIT_KEY`` (``api_key`` | ``ip``, default ``api_key``).
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock
from typing import Deque, Dict, Optional


class SlidingWindowRateLimiter:
    """Per-key sliding-window rate limiter (thread-safe)."""

    def __init__(self, limit_per_minute: int, window_s: float = 60.0) -> None:
        self.limit = limit_per_minute
        self.window_s = window_s
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, key: str) -> bool:
        """Record a hit for ``key``; True if within limit."""
        if self.limit <= 0:
            return True
        now = time.monotonic()
        with self._lock:
            dq = self._hits[key]
            # Drop entries outside the window.
            while dq and now - dq[0] >= self.window_s:
                dq.popleft()
            if len(dq) >= self.limit:
                return False
            dq.append(now)
            return True

    def remaining(self, key: str) -> int:
        if self.limit <= 0:
            return -1
        now = time.monotonic()
        with self._lock:
            dq = self._hits[key]
            while dq and now - dq[0] >= self.window_s:
                dq.popleft()
            return max(0, self.limit - len(dq))

    def reset(self, key: Optional[str] = None) -> None:
        with self._lock:
            if key is None:
                self._hits.clear()
            else:
                self._hits.pop(key, None)
