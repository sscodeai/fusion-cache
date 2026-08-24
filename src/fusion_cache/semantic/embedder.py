"""OpenAI-compatible embedding client for the L2 semantic layer.

Uses the same ``/embeddings`` REST shape as OpenAI / DeepSeek / SiliconFlow,
with ``base_url`` fully configurable so any compatible endpoint works.
``api_key`` is sent as ``Authorization: Bearer`` only when non-empty.
"""

from __future__ import annotations

import os
from typing import List, Optional

import httpx

from ..config import EmbedderConfig

try:
    import numpy as np  # type: ignore

    HAS_NUMPY = True
except ImportError:  # pragma: no cover
    HAS_NUMPY = False


class Embedder:
    """Async OpenAI-compatible embedder with numpy cosine similarity.

    ``enabled`` is False when no API key is configured; in that case the L2
    layer is effectively skipped (the pipeline already guards on it).
    """

    def __init__(self, config: Optional[EmbedderConfig] = None, client: Optional[httpx.AsyncClient] = None) -> None:
        self.config = config or EmbedderConfig()
        self.api_key = self.config.api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        self.enabled = bool(self.api_key)
        self._client = client
        self._owns_client = client is None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url.rstrip("/"), timeout=self.config.timeout, headers=headers
            )
        return self._client

    def _url(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/embeddings"

    async def embed(self, text: str) -> List[float]:
        """Embed a single text string; returns a float list."""
        if not self.enabled:
            raise RuntimeError("Embedder is not configured (no API key); cannot embed.")
        payload = {"model": self.config.model, "input": text}
        resp = await self.client.post("/embeddings", json=payload)
        resp.raise_for_status()
        data = resp.json()
        try:
            return list(data["data"][0]["embedding"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected embeddings response shape: {str(data)[:200]}") from exc

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not self.enabled:
            raise RuntimeError("Embedder is not configured (no API key); cannot embed.")
        payload = {"model": self.config.model, "input": texts}
        resp = await self.client.post("/embeddings", json=payload)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("data") or []
        by_index = {int(it.get("index", i)): list(it["embedding"]) for i, it in enumerate(items)}
        return [by_index[i] for i in range(len(texts))]

    @staticmethod
    def similarity(a: List[float], b: List[float]) -> float:
        """Cosine similarity between two embedding vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0
        if HAS_NUMPY:
            va = np.asarray(a, dtype=np.float64)
            vb = np.asarray(b, dtype=np.float64)
            denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
            if denom == 0.0:
                return 0.0
            return float(np.dot(va, vb) / denom)
        # pure-python fallback
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        if na * nb == 0.0:
            return 0.0
        return dot / (na * nb)

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None
