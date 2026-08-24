"""Pipeline tests: L1 exact, L2 semantic, L3 prefix accounting, key canonicalization."""

from __future__ import annotations

import pytest

from fusion_cache.core.key import canonicalize_request, request_hash
from fusion_cache.core.pipeline import BufferedStream

from conftest import (
    chat_request,
    fake_stream_upstream,
    fake_upstream,
    make_cache,
    make_cache_no_embed,
)


class TestCanonicalization:
    def test_whitespace_collapse(self):
        a = chat_request(messages=[{"role": "user", "content": "Hello   there\n\nworld"}])
        b = chat_request(messages=[{"role": "user", "content": "Hello there world"}])
        assert request_hash(a) == request_hash(b)

    def test_message_order_matters(self):
        a = chat_request(messages=[{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}])
        b = chat_request(messages=[{"role": "assistant", "content": "y"}, {"role": "user", "content": "x"}])
        assert request_hash(a) != request_hash(b)

    def test_temperature_in_key(self):
        a = chat_request(temperature=0.2)
        b = chat_request(temperature=0.8)
        assert request_hash(a) != request_hash(b)

    def test_model_in_key(self):
        a = chat_request(model="deepseek-chat")
        b = chat_request(model="deepseek-reasoner")
        assert request_hash(a) != request_hash(b)

    def test_stream_flag_distinguishes(self):
        a = chat_request(stream=False)
        b = chat_request(stream=True)
        assert request_hash(a) != request_hash(b)

    def test_irrelevant_params_dropped(self):
        a = chat_request(user="alice", stream_options={"include_usage": True})
        b = chat_request(user="bob")
        # user/stream_options are transport noise; keys match
        assert request_hash(a) == request_hash(b)

    def test_canonical_stable_sort(self):
        a = canonicalize_request(chat_request(messages=[{"content": "hi", "role": "user"}]))
        b = canonicalize_request(chat_request(messages=[{"role": "user", "content": "hi"}]))
        assert a == b


class TestExactLayer:
    @pytest.mark.asyncio
    async def test_first_miss_then_exact_hit(self):
        cache = make_cache_no_embed()
        upstream_calls = []

        async def upstream(**kwargs):
            upstream_calls.append(kwargs)
            return await fake_upstream(**kwargs)

        req = chat_request()
        r1 = await cache.chat_completion(request=req, upstream=upstream, stream=False)
        assert r1.layer == "miss"
        assert r1.cached is False
        assert r1.prefix_hit is False
        assert len(upstream_calls) == 1

        r2 = await cache.chat_completion(request=req, upstream=upstream, stream=False)
        assert r2.layer == "exact"
        assert r2.hit is True
        assert r2.cached is True
        assert len(upstream_calls) == 1  # upstream not called again

    @pytest.mark.asyncio
    async def test_exact_layer_can_be_disabled(self):
        cache = make_cache_no_embed(enable_exact=False)
        calls = 0

        async def upstream(**kwargs):
            nonlocal calls
            calls += 1
            return await fake_upstream(**kwargs)

        req = chat_request()
        await cache.chat_completion(request=req, upstream=upstream, stream=False)
        await cache.chat_completion(request=req, upstream=upstream, stream=False)
        assert calls == 2

    @pytest.mark.asyncio
    async def test_ttl_expiry(self):
        import asyncio

        cache = make_cache_no_embed(exact_ttl=0.05)
        calls = 0

        async def upstream(**kwargs):
            nonlocal calls
            calls += 1
            return await fake_upstream(**kwargs)

        req = chat_request()
        await cache.chat_completion(request=req, upstream=upstream, stream=False)
        await asyncio.sleep(0.08)
        await cache.chat_completion(request=req, upstream=upstream, stream=False)
        assert calls == 2

    @pytest.mark.asyncio
    async def test_different_temperature_is_a_miss(self):
        cache = make_cache_no_embed()
        calls = 0

        async def upstream(**kwargs):
            nonlocal calls
            calls += 1
            return await fake_upstream(**kwargs)

        await cache.chat_completion(request=chat_request(temperature=0.2), upstream=upstream, stream=False)
        await cache.chat_completion(request=chat_request(temperature=0.9), upstream=upstream, stream=False)
        assert calls == 2


class TestSemanticLayer:
    @pytest.mark.asyncio
    async def test_semantic_hit_on_similar_question(self):
        cache = make_cache(similarity_threshold=0.6)
        calls = 0

        async def upstream(**kwargs):
            nonlocal calls
            calls += 1
            return await fake_upstream(**kwargs)

        base = chat_request(messages=[{"role": "user", "content": "How do I reset my password?"}])
        similar = chat_request(messages=[{"role": "user", "content": "How can I reset the password?"}])

        r1 = await cache.chat_completion(request=base, upstream=upstream, stream=False)
        assert r1.layer == "miss"

        r2 = await cache.chat_completion(request=similar, upstream=upstream, stream=False)
        assert r2.layer == "semantic", r2
        assert r2.hit is True
        assert calls == 1  # upstream not called for the similar question

    @pytest.mark.asyncio
    async def test_unrelated_question_is_miss(self):
        cache = make_cache(similarity_threshold=0.6)
        calls = 0

        async def upstream(**kwargs):
            nonlocal calls
            calls += 1
            return await fake_upstream(**kwargs)

        await cache.chat_completion(
            request=chat_request(messages=[{"role": "user", "content": "What is the capital of France?"}]),
            upstream=upstream,
            stream=False,
        )
        r2 = await cache.chat_completion(
            request=chat_request(messages=[{"role": "user", "content": "Tell me about quantum physics please"}]),
            upstream=upstream,
            stream=False,
        )
        assert r2.layer == "miss"
        assert calls == 2

    @pytest.mark.asyncio
    async def test_threshold_gating(self):
        # high threshold → no semantic hit even for similar text
        cache = make_cache(similarity_threshold=0.99)
        calls = 0

        async def upstream(**kwargs):
            nonlocal calls
            calls += 1
            return await fake_upstream(**kwargs)

        base = chat_request(messages=[{"role": "user", "content": "reset my password please"}]),
        # note: trailing comma makes this a tuple; fix below
        base = chat_request(messages=[{"role": "user", "content": "reset my password please"}])
        similar = chat_request(messages=[{"role": "user", "content": "reset my password please thanks"}])

        await cache.chat_completion(request=base, upstream=upstream, stream=False)
        r2 = await cache.chat_completion(request=similar, upstream=upstream, stream=False)
        assert r2.layer == "miss"
        assert calls == 2

    @pytest.mark.asyncio
    async def test_guardrail_different_model(self):
        cache = make_cache(similarity_threshold=0.6)
        calls = 0

        async def upstream(**kwargs):
            nonlocal calls
            calls += 1
            return await fake_upstream(**kwargs)

        base = chat_request(model="deepseek-chat", messages=[{"role": "user", "content": "reset password"}])
        await cache.chat_completion(request=base, upstream=upstream, stream=False)

        other_model = chat_request(model="gpt-4o", messages=[{"role": "user", "content": "reset password"}])
        r2 = await cache.chat_completion(request=other_model, upstream=upstream, stream=False)
        # structurally incompatible (different model) → guardrail rejects
        assert r2.layer == "miss"
        assert calls == 2

    @pytest.mark.asyncio
    async def test_semantic_disabled(self):
        cache = make_cache(enable_semantic=False, similarity_threshold=0.6)
        calls = 0

        async def upstream(**kwargs):
            nonlocal calls
            calls += 1
            return await fake_upstream(**kwargs)

        base = chat_request(messages=[{"role": "user", "content": "reset password"}])
        await cache.chat_completion(request=base, upstream=upstream, stream=False)
        similar = chat_request(messages=[{"role": "user", "content": "reset password now"}])
        r2 = await cache.chat_completion(request=similar, upstream=upstream, stream=False)
        assert r2.layer == "miss"
        assert calls == 2


class TestPrefixAccountingLayer:
    @pytest.mark.asyncio
    async def test_prefix_hit_tokens_recorded(self):
        cache = make_cache_no_embed()
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_cache_hit_tokens": 80,
            "prompt_cache_miss_tokens": 20,
        }

        async def upstream(**kwargs):
            return {"id": "x", "choices": [{"message": {"content": "ok"}}], "usage": usage}

        r = await cache.chat_completion(request=chat_request(), upstream=upstream, stream=False)
        assert r.layer == "miss"
        assert r.upstream_hit_tokens == 80
        assert r.upstream_miss_tokens == 20
        assert r.prefix_hit is True
        # saved = 80 * (0.22 - 0.007)/1e6
        expected = 80 * (0.22 - 0.007) / 1_000_000.0
        assert abs(r.cost_saved_usd - expected) < 1e-9

    @pytest.mark.asyncio
    async def test_no_prefix_tokens_no_savings(self):
        cache = make_cache_no_embed()
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 10,
        }

        async def upstream(**kwargs):
            return {"id": "x", "choices": [], "usage": usage}

        r = await cache.chat_completion(request=chat_request(), upstream=upstream, stream=False)
        assert r.layer == "miss"
        assert r.upstream_hit_tokens == 0
        assert r.cost_saved_usd == 0.0
        assert r.prefix_hit is False

    @pytest.mark.asyncio
    async def test_accounting_disabled(self):
        cache = make_cache_no_embed(enable_prefix_accounting=False)
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_cache_hit_tokens": 80,
            "prompt_cache_miss_tokens": 20,
        }

        async def upstream(**kwargs):
            return {"id": "x", "choices": [], "usage": usage}

        r = await cache.chat_completion(request=chat_request(), upstream=upstream, stream=False)
        assert r.layer == "miss"
        # savings are still computed and recorded even when accounting toggle is off
        assert r.cost_saved_usd > 0

    @pytest.mark.asyncio
    async def test_stream_passthrough_accounting(self):
        cache = make_cache_no_embed()
        calls = []

        async def upstream(**kwargs):
            return fake_stream_upstream(**kwargs)

        r = await cache.chat_completion(
            request=chat_request(messages=[{"role": "user", "content": "hi"}]),
            upstream=upstream,
            stream=True,
        )
        assert isinstance(r.response, BufferedStream)
        assert len(r.response.chunks) == 5
        assert r.upstream_hit_tokens == 8
        assert r.upstream_miss_tokens == 2
        assert r.layer == "miss"
        assert r.prefix_hit is True

    @pytest.mark.asyncio
    async def test_stats_aggregate(self):
        cache = make_cache_no_embed()

        async def upstream(**kwargs):
            return await fake_upstream(**kwargs)

        req = chat_request()
        await cache.chat_completion(request=req, upstream=upstream, stream=False)  # miss
        await cache.chat_completion(request=req, upstream=upstream, stream=False)  # exact hit

        stats = cache.stats
        assert stats.requests == 2
        assert stats.misses == 1
        assert stats.exact_hits == 1
        assert stats.prefix_hits == 0
        assert stats.hit_rate() == 0.5

    @pytest.mark.asyncio
    async def test_upstream_error_propagates(self):
        cache = make_cache_no_embed()

        async def upstream(**kwargs):
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            await cache.chat_completion(request=chat_request(), upstream=upstream, stream=False)
