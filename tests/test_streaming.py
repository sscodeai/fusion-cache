"""Streaming tests: buffer-then-replay, chunk integrity, usage preservation."""

from __future__ import annotations

import pytest

from fusion_cache.core.pipeline import BufferedStream, FusionCache
from fusion_cache.core.replay import (
    buffer_stream,
    replay_chunks,
    replay_chunks_sync,
    stream_text,
)
from fusion_cache.wrapper.openai import CachedOpenAI

from conftest import (
    FakeClient,
    FakeAsyncClient,
    chat_request,
    fake_stream_upstream,
    make_cache_no_embed,
)


class TestReplayHelpers:
    @pytest.mark.asyncio
    async def test_buffer_and_replay_roundtrip(self):
        chunks, meta = await buffer_stream(fake_stream_upstream())
        assert len(chunks) == 5
        assert meta["id"] == "chatcmpl-fake-stream"
        assert meta["usage"]["prompt_cache_hit_tokens"] == 8

        replayed = [c async for c in replay_chunks(chunks)]
        assert replayed == chunks

    @pytest.mark.asyncio
    async def test_sync_replay(self):
        chunks, _ = await buffer_stream(fake_stream_upstream())
        replayed = list(replay_chunks_sync(chunks))
        assert replayed == chunks

    def test_stream_text(self):
        chunks = [
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": " world"}}]},
            {"choices": [{"delta": {"content": "!"}}]},
        ]
        assert stream_text(chunks) == "Hello world!"

    @pytest.mark.asyncio
    async def test_empty_stream_raises(self):
        async def empty():
            if False:
                yield {}

        with pytest.raises(RuntimeError):
            await buffer_stream(empty())


class TestStreamingPipeline:
    @pytest.mark.asyncio
    async def test_stream_buffers_and_caches(self):
        cache = make_cache_no_embed()
        calls = 0

        async def upstream(**kwargs):
            nonlocal calls
            calls += 1
            return fake_stream_upstream(**kwargs)

        req = chat_request(messages=[{"role": "user", "content": "stream me"}])
        r1 = await cache.chat_completion(request=req, upstream=upstream, stream=True)
        assert isinstance(r1.response, BufferedStream)
        assert len(r1.response.chunks) == 5
        assert calls == 1

        # second identical request → exact hit, same chunks
        r2 = await cache.chat_completion(request=req, upstream=upstream, stream=True)
        assert r2.layer == "exact"
        assert isinstance(r2.response, BufferedStream)
        assert r2.response.chunks == r1.response.chunks
        assert calls == 1

    @pytest.mark.asyncio
    async def test_stream_usage_captured(self):
        cache = make_cache_no_embed()

        async def upstream(**kwargs):
            return fake_stream_upstream(**kwargs)

        r = await cache.chat_completion(
            request=chat_request(messages=[{"role": "user", "content": "hi"}]),
            upstream=upstream,
            stream=True,
        )
        assert r.usage["prompt_cache_hit_tokens"] == 8

    @pytest.mark.asyncio
    async def test_stream_and_nonstream_are_separate_keys(self):
        cache = make_cache_no_embed()
        calls = 0

        async def upstream(**kwargs):
            nonlocal calls
            calls += 1
            return fake_stream_upstream(**kwargs) if kwargs.get("stream") else {"choices": [], "usage": {}}

        req = chat_request(messages=[{"role": "user", "content": "same prompt"}])
        await cache.chat_completion(request=req, upstream=upstream, stream=False)
        r2 = await cache.chat_completion(request=req, upstream=upstream, stream=True)
        assert r2.layer == "miss"  # stream key differs from non-stream key
        assert calls == 2


class TestWrapperStreaming:
    def test_sync_wrapper_stream_replay(self):
        client = FakeClient()
        wrapper = CachedOpenAI(client, cache=make_cache_no_embed())
        req = chat_request(messages=[{"role": "user", "content": "hello stream"}], stream=True)

        gen1 = wrapper.chat.completions.create(**req)
        texts = [c["choices"][0]["delta"]["content"] for c in gen1 if c["choices"][0]["delta"].get("content")]
        assert "".join(texts) == "Hello world!"

        gen2 = wrapper.chat.completions.create(**req)
        assert len(list(gen2)) == 5
        assert len(client.chat.completions.calls) == 1  # cached, upstream called once

    def test_sync_wrapper_stream_usage_on_last_chunk(self):
        client = FakeClient()
        wrapper = CachedOpenAI(client, cache=make_cache_no_embed())
        req = chat_request(messages=[{"role": "user", "content": "hello stream"}], stream=True)

        chunks = list(wrapper.chat.completions.create(**req))
        last = chunks[-1]
        assert last["choices"][0]["finish_reason"] == "stop"
        assert last.get("usage", {}).get("prompt_cache_hit_tokens") == 8

    @pytest.mark.asyncio
    async def test_async_wrapper_stream_replay(self):
        client = FakeAsyncClient()
        wrapper = CachedOpenAI(client, cache=make_cache_no_embed())
        req = chat_request(messages=[{"role": "user", "content": "hello stream"}], stream=True)

        gen1 = await wrapper.chat.completions.create(**req)
        texts = []
        async for c in gen1:
            texts.append(c["choices"][0]["delta"].get("content", ""))
        assert "".join(texts) == "Hello world!"

        gen2 = await wrapper.chat.completions.create(**req)
        count = 0
        async for _ in gen2:
            count += 1
        assert count == 5
        assert len(client.chat.completions.calls) == 1

    @pytest.mark.asyncio
    async def test_async_wrapper_nonstream(self):
        client = FakeAsyncClient()
        wrapper = CachedOpenAI(client, cache=make_cache_no_embed())
        req = chat_request(messages=[{"role": "user", "content": "hello non-stream"}])

        resp1 = await wrapper.chat.completions.create(**req)
        assert resp1["choices"][0]["message"]["content"] == "Hello from fake upstream"
        resp2 = await wrapper.chat.completions.create(**req)
        assert resp2 == resp1
        assert len(client.chat.completions.calls) == 1

    def test_sync_wrapper_nonstream(self):
        client = FakeClient()
        wrapper = CachedOpenAI(client, cache=make_cache_no_embed())
        req = chat_request(messages=[{"role": "user", "content": "hello non-stream"}])

        resp1 = wrapper.chat.completions.create(**req)
        assert resp1["choices"][0]["message"]["content"] == "Hello from fake upstream"
        resp2 = wrapper.chat.completions.create(**req)
        assert resp2 == resp1
        assert len(client.chat.completions.calls) == 1
