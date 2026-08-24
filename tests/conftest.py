"""Shared fixtures: a fake OpenAI-compatible server + async upstream helpers."""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional

import pytest
import respx
from httpx import AsyncClient, Response

from fusion_cache.config import FusionCacheConfig, EmbedderConfig
from fusion_cache.core.pipeline import FusionCache
from fusion_cache.core.replay import replay_chunks
from fusion_cache.semantic.embedder import Embedder
from fusion_cache.stores.memory import MemoryStore


def make_fake_router(
    *,
    chat_response: Optional[Dict[str, Any]] = None,
    chat_status: int = 200,
    embed_status: int = 200,
    embed_model: str = "fake-embed",
):
    """Return a respx router for /chat/completions and /embeddings.

    Chat route returns ``chat_response`` (defaults to a small completion with
    usage).  Embed route returns the embedding of the input using a simple
    bag-of-bigrams hash so similar texts have similar embeddings.
    """
    router = respx.mock(base_url="https://fake.openai.test", assert_all_called=False)

    def chat_handler(request):
        if chat_status != 200:
            return Response(chat_status, json={"error": {"message": "boom"}})
        body = request.content and request.content.decode()
        # include a deterministic echo so the fake server is stateful-ish
        resp = dict(chat_response) if chat_response else _default_chat_response()
        return Response(200, json=resp)

    def embed_handler(request):
        if embed_status != 200:
            return Response(embed_status, json={"error": {"message": "embed boom"}})
        body = request.content.decode()
        import json

        payload = json.loads(body)
        inputs = payload.get("input")
        if isinstance(inputs, str):
            inputs = [inputs]
        data = []
        for i, text in enumerate(inputs):
            emb = _embedding_for(text)
            data.append({"index": i, "object": "embedding", "embedding": emb})
        return Response(200, json={"object": "list", "data": data, "model": embed_model, "usage": {}})

    router.post("https://fake.openai.test/chat/completions").mock(return_value=Response(chat_status, json=_default_chat_response()))
    router.post("https://fake.openai.test/embeddings").mock(side_effect=embed_handler)
    return router


def _default_chat_response() -> Dict[str, Any]:
    return {
        "id": "chatcmpl-fake-1",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "deepseek-chat",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from fake upstream"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 5,
            "total_tokens": 17,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 12,
        },
    }


def _embedding_for(text: str) -> List[float]:
    """Deterministic character-n-gram embedding with L2-normalization.

    Paraphrases share most character trigrams, so similar texts get similar
    vectors — enough to exercise the semantic layer end-to-end without a real
    model.  Unrelated texts share few trigrams → low similarity.
    """
    import math

    text = text.lower()
    ngrams: Dict[str, int] = {}
    # word unigrams + character trigrams
    for tok in text.split():
        ngrams["w:" + tok] = ngrams.get("w:" + tok, 0) + 1
    for i in range(len(text) - 2):
        gram = text[i : i + 3]
        ngrams[gram] = ngrams.get(gram, 0) + 1

    # fixed vocabulary of 128 dims so all embeddings are comparable
    vec = [0.0] * 128
    for gram, count in ngrams.items():
        h = hash(gram) & 127
        vec[h] += count
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def make_embedder() -> Embedder:
    """Embedder pointed at the fake server with a fake key."""
    return Embedder(
        EmbedderConfig(
            base_url="https://fake.openai.test",
            api_key="sk-fake",
            model="fake-embed",
        )
    )


def activate_fake_server():
    """Start the respx router (idempotent) so fake requests never go to the wire."""
    if not getattr(activate_fake_server, "_active", False):
        router = make_fake_router()
        router.start()
        activate_fake_server._active = True


def deactivate_fake_server():
    if getattr(activate_fake_server, "_active", False):
        from httpx import MockTransport  # noqa: F401

        activate_fake_server._active = False
        # stop the respx router started in activate_fake_server
        for m in list(respx._router._routes):
            pass
        try:
            respx.stop_all()
        except Exception:
            pass


def make_cache(**cfg_overrides) -> FusionCache:
    activate_fake_server()
    cfg = FusionCacheConfig(**cfg_overrides)
    return FusionCache(config=cfg, embedder=make_embedder())


def make_cache_no_embed(**cfg_overrides) -> FusionCache:
    """Cache with the embedder disabled (no API key) — tests L1/L3 only."""
    cfg = FusionCacheConfig(**cfg_overrides)
    embedder = Embedder(EmbedderConfig(base_url="https://fake.openai.test", api_key="", model="fake-embed"))
    return FusionCache(config=cfg, embedder=embedder)


def chat_request(**overrides) -> Dict[str, Any]:
    req = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "Hello there"}],
        "temperature": 0.2,
    }
    req.update(overrides)
    return req


async def fake_upstream(**kwargs) -> Any:
    """Default upstream: return a dict-shaped completion."""
    return _default_chat_response()


def fake_stream_upstream(**kwargs):
    """Return a sync generator of chunk dicts (buffered by the pipeline)."""

    def gen():
        for text in ["Hello", " ", "world", "!"]:
            yield {
                "id": "chatcmpl-fake-stream",
                "object": "chat.completion.chunk",
                "created": 1_700_000_000,
                "model": kwargs.get("model", "deepseek-chat"),
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
            }
        yield {
            "id": "chatcmpl-fake-stream",
            "object": "chat.completion.chunk",
            "created": 1_700_000_000,
            "model": kwargs.get("model", "deepseek-chat"),
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
                "prompt_cache_hit_tokens": 8,
                "prompt_cache_miss_tokens": 2,
            },
        }

    return gen()


class FakeCompletions:
    """Minimal fake ``client.chat.completions`` used by wrapper tests."""

    def __init__(self, responses=None, stream_chunks=None, calls=None):
        self._responses = responses or []
        self._stream_chunks = stream_chunks
        self.calls = calls if calls is not None else []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            if self._stream_chunks is None:
                return fake_stream_upstream(**kwargs)
            return self._stream_chunks(**kwargs)
        if self._responses:
            return self._responses.pop(0)
        return _default_chat_response()


class FakeChat:
    def __init__(self, completions):
        self.completions = completions


class FakeClient:
    def __init__(self, responses=None, stream_chunks=None, calls=None):
        self.chat = FakeChat(FakeCompletions(responses, stream_chunks, calls))


class FakeAsyncCompletions:
    def __init__(self, responses=None, stream_chunks=None, calls=None):
        self._responses = responses or []
        self._stream_chunks = stream_chunks
        self.calls = calls if calls is not None else []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            chunks = self._stream_chunks(**kwargs) if self._stream_chunks else fake_stream_upstream(**kwargs)
            # the pipeline's astream() normalizes sync iterators; return as-is
            return chunks
        if self._responses:
            return self._responses.pop(0)
        return _default_chat_response()


class FakeAsyncChat:
    def __init__(self, completions):
        self.completions = completions


class FakeAsyncClient:
    def __init__(self, responses=None, stream_chunks=None, calls=None):
        self.chat = FakeAsyncChat(FakeAsyncCompletions(responses, stream_chunks, calls))
