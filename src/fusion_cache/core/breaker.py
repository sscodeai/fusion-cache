"""Circuit breaker for upstream LLM calls.

When the upstream starts failing (5xx, network errors), the breaker opens
and short-circuits further upstream calls for a cooldown window.  During an
open breaker, cached hits (L1/L2) are still served normally — only requests
that would require a fresh upstream call fail fast.  This keeps the cache's
value (serving hits) intact even when the upstream is down.

States:

- **closed**: normal operation; failures counted.
- **open**: upstream calls blocked for ``cooldown_s``; hits still served.
- **half-open** (after cooldown): one probe call allowed; success → closed,
  failure → open again.
"""

from __future__ import annotations

import time
from typing import Optional


class CircuitBreaker:
    """A minimal, thread-safe circuit breaker."""

    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown_s: float = 30.0,
        enabled: bool = True,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self.enabled = enabled
        self._failures = 0
        self._state = "closed"  # closed | open | half-open
        self._opened_at: Optional[float] = None

    @property
    def state(self) -> str:
        if self.enabled and self._state == "open" and self._opened_at is not None:
            if time.monotonic() - self._opened_at >= self.cooldown_s:
                self._state = "half-open"
        return self._state

    def allow_request(self) -> bool:
        """True if an upstream call is allowed right now."""
        if not self.enabled:
            return True
        s = self.state
        if s == "open":
            return False
        if s == "half-open":
            # Allow one probe; mark half-open so concurrent callers wait.
            self._state = "open"
            self._opened_at = time.monotonic()
            return True
        return True

    def record_success(self) -> None:
        if not self.enabled:
            return
        self._failures = 0
        self._state = "closed"
        self._opened_at = None

    def record_failure(self) -> None:
        if not self.enabled:
            return
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._state = "open"
            self._opened_at = time.monotonic()

    def reset(self) -> None:
        self._failures = 0
        self._state = "closed"
        self._opened_at = None
