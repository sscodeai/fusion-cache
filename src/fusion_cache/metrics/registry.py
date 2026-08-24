"""Metrics registry: per-layer hit/miss, hit rate, $ saved, P50/P95 latency.

Thread-safe counters plus a rolling latency buffer.  The registry does not
depend on any external system — the dashboard/prometheus exporters consume
:meth:`snapshot`.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

Layer = str


class MetricsRegistry:
    """In-process metrics for one FusionCache instance."""

    def __init__(self, max_latency_samples: int = 4096) -> None:
        self.max_latency_samples = max_latency_samples
        self._lock = threading.Lock()
        self._latencies: Deque[float] = deque(maxlen=max_latency_samples)
        self._layers: Dict[Layer, Dict[str, Any]] = {}
        self._started = time.monotonic()

    # ---------------------------------------------------------------- record
    def record(
        self,
        layer: str,
        hit: Optional[bool] = None,
        latency_ms: Optional[float] = None,
        cost_saved_usd: float = 0.0,
        tokens: Optional[int] = None,
        reason: Optional[str] = None,
    ) -> None:
        with self._lock:
            entry = self._layers.setdefault(
                layer, {"hits": 0, "misses": 0, "cost_saved_usd": 0.0, "tokens": 0, "errors": 0}
            )
            if hit is True:
                entry["hits"] += 1
            elif hit is False:
                if reason == "upstream_error":
                    entry["errors"] += 1
                entry["misses"] += 1
            if latency_ms is not None:
                self._latencies.append(latency_ms)
            if cost_saved_usd:
                entry["cost_saved_usd"] += cost_saved_usd
            if tokens:
                entry["tokens"] += tokens

    # ---------------------------------------------------------------- queries
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            layers = {k: dict(v) for k, v in self._layers.items()}
            latencies = list(self._latencies)
        total_hits = sum(v["hits"] for v in layers.values())
        total_misses = sum(v["misses"] for v in layers.values())
        total = total_hits + total_misses
        total_saved = sum(v["cost_saved_usd"] for v in layers.values())
        return {
            "layers": layers,
            "total_requests": total,
            "total_hits": total_hits,
            "total_misses": total_misses,
            "hit_rate": (total_hits / total) if total else 0.0,
            "cost_saved_usd": round(total_saved, 6),
            "tokens_cached": sum(v["tokens"] for v in layers.values()),
            "latency": {
                "p50_ms": self._percentile(latencies, 0.50),
                "p95_ms": self._percentile(latencies, 0.95),
                "mean_ms": (sum(latencies) / len(latencies)) if latencies else 0.0,
                "samples": len(latencies),
            },
            "uptime_s": round(time.monotonic() - self._started, 3),
        }

    @staticmethod
    def _percentile(values: List[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = (len(ordered) - 1) * q
        lo = math.floor(idx)
        hi = math.ceil(idx)
        if lo == hi:
            return ordered[lo]
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo)

    def reset(self) -> None:
        with self._lock:
            self._layers.clear()
            self._latencies.clear()

    def latency_series(self, n: int = 60) -> List[float]:
        """Most recent latency samples (oldest → newest), for sparklines."""
        with self._lock:
            xs = list(self._latencies)
        return xs[-n:]

    def layer_summary(self, layer: str) -> Dict[str, Any]:
        with self._lock:
            entry = self._layers.get(layer, {"hits": 0, "misses": 0, "cost_saved_usd": 0.0, "tokens": 0, "errors": 0})
            return dict(entry)

    def __repr__(self) -> str:  # pragma: no cover
        return f"MetricsRegistry({self._layers!r})"
