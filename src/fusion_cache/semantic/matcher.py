"""Semantic matcher: numpy cosine + threshold + false-positive guardrail.

The matcher is deliberately tiny: cosine similarity over float vectors, a
configurable threshold, and a guardrail that refuses matches whose cached
request is *structurally incompatible* with the incoming one (different
model / stream mode).  It is used by the pipeline for the L2 scan; keeping it
standalone makes the behavior unit-testable without an embedder.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

from ..config import FusionCacheConfig
from ..semantic.embedder import Embedder


class SemanticMatcher:
    """Scores stored entries against a query embedding.

    Typical use::

        matcher = SemanticMatcher(config)
        best = matcher.best_match(query_embedding, entries, request=incoming_request)
        if best is not None:
            response = best["response"]
    """

    def __init__(self, config: Optional[FusionCacheConfig] = None) -> None:
        self.config = config or FusionCacheConfig()
        self.threshold = self.config.similarity_threshold

    def score(self, query: List[float], candidate: List[float]) -> float:
        return Embedder.similarity(query, candidate)

    def eligible(self, score: float) -> bool:
        return score >= self.threshold

    def best_match(
        self,
        query_embedding: List[float],
        entries: Iterable[Dict[str, Any]],
        request: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return the best entry above threshold, or None.

        Each ``entries`` item must be a dict with ``embedding`` (and
        optionally ``response``/``meta``/``key``).  When ``request`` is given,
        the false-positive guardrail rejects structurally incompatible
        candidates (different model or stream flag stored in ``meta``).
        """
        best: Optional[Dict[str, Any]] = None
        best_score = 0.0
        for entry in entries:
            embedding = entry.get("embedding")
            if embedding is None:
                continue
            score = self.score(query_embedding, embedding)
            if score <= best_score:
                continue
            if request is not None and not self.guard_ok(entry, request):
                continue
            best_score = score
            best = entry
        if best is None or best_score < self.threshold:
            return None
        best["_score"] = best_score
        return best

    @staticmethod
    def guard_ok(entry: Mapping[str, Any], request: Mapping[str, Any]) -> bool:
        """False-positive guardrail: reject structurally incompatible matches.

        Only *structural* fields are checked — model and stream mode.  Content
        similarity is the cosine threshold's job; the guardrail exists so a
        high-similarity query from a different model can never replay a cached
        response that was generated under different sampling settings.
        """
        meta = entry.get("meta") or {}
        candidate_req = meta.get("_request") if isinstance(meta, dict) else None
        if candidate_req is None:
            return True
        return (
            candidate_req.get("model") == request.get("model")
            and bool(candidate_req.get("stream")) == bool(request.get("stream"))
        )
