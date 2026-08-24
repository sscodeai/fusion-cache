"""Metrics tests: per-layer hit/miss, hit rate, $ saved, P50/P95, accounting math."""

from __future__ import annotations

import pytest

from fusion_cache.config import DeepSeekPriceModel
from fusion_cache.metrics.registry import MetricsRegistry
from fusion_cache.prefix.accounting import (
    UsageBreakdown,
    account_prefix_costs,
    summarize_usage,
)

from conftest import chat_request, fake_upstream, make_cache_no_embed


class TestMetricsRegistry:
    def test_record_hit_miss(self):
        m = MetricsRegistry()
        m.record("exact", hit=True)
        m.record("exact", hit=True)
        m.record("semantic", hit=True)
        m.record("miss", hit=False)

        snap = m.snapshot()
        assert snap["total_requests"] == 4
        assert snap["total_hits"] == 3
        assert snap["total_misses"] == 1
        assert snap["hit_rate"] == 0.75
        assert snap["layers"]["exact"]["hits"] == 2

    def test_latency_percentiles(self):
        m = MetricsRegistry()
        for i in range(1, 101):
            m.record("miss", hit=False, latency_ms=float(i))
        snap = m.snapshot()
        assert snap["latency"]["p50_ms"] == pytest.approx(50.5, abs=0.1)
        assert snap["latency"]["p95_ms"] == pytest.approx(95.05, abs=0.1)
        assert snap["latency"]["mean_ms"] == pytest.approx(50.5, abs=0.1)

    def test_percentile_empty(self):
        assert MetricsRegistry().snapshot()["latency"]["p50_ms"] == 0.0

    def test_cost_saved_aggregation(self):
        m = MetricsRegistry()
        m.record("miss", hit=False, cost_saved_usd=0.001)
        m.record("miss", hit=False, cost_saved_usd=0.002)
        assert m.snapshot()["cost_saved_usd"] == pytest.approx(0.003)

    def test_tokens(self):
        m = MetricsRegistry()
        m.record("prefix", hit=True, tokens=80)
        m.record("prefix", hit=True, tokens=20)
        assert m.snapshot()["tokens_cached"] == 100

    def test_reset(self):
        m = MetricsRegistry()
        m.record("exact", hit=True)
        m.reset()
        assert m.snapshot()["total_requests"] == 0


class TestAccountingMath:
    def test_savings_formula(self):
        b = UsageBreakdown(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            prompt_cache_hit_tokens=80,
            prompt_cache_miss_tokens=20,
        )
        pm = DeepSeekPriceModel(input_miss_per_1m=0.22, input_hit_per_1m=0.007, output_per_1m=1.1)
        costs = account_prefix_costs(b, pm)
        # saved = 80 * (0.22 - 0.007) / 1e6
        assert costs.saved_usd == pytest.approx(80 * (0.22 - 0.007) / 1e6)
        # input hit billed at 0.007/1e6 per token
        assert costs.input_hit_usd == pytest.approx(80 * 0.007 / 1e6)

    def test_no_hit_tokens_no_savings(self):
        b = UsageBreakdown(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            prompt_cache_hit_tokens=0,
            prompt_cache_miss_tokens=10,
        )
        costs = account_prefix_costs(b)
        assert costs.saved_usd == 0.0

    def test_attribution_fallback(self):
        # provider reports prompt_tokens but no cache fields → all prompt is miss
        b = UsageBreakdown(prompt_tokens=50, completion_tokens=10, total_tokens=60)
        costs = account_prefix_costs(b)
        assert costs.input_miss_usd == pytest.approx(50 * 0.22 / 1e6)
        assert costs.saved_usd == 0.0

    def test_hit_rate(self):
        b = UsageBreakdown(prompt_tokens=100, prompt_cache_hit_tokens=75, prompt_cache_miss_tokens=25)
        assert b.prefix_cache_hit_rate == pytest.approx(0.75)

    def test_summarize_usage_from_dict(self):
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_cache_hit_tokens": 80,
            "prompt_cache_miss_tokens": 20,
        }
        summary = summarize_usage(usage)
        assert summary["usage"]["prompt_cache_hit_tokens"] == 80
        assert summary["cost"]["saved_usd"] > 0

    def test_summarize_usage_from_object(self):
        class Usage:
            prompt_tokens = 100
            completion_tokens = 20
            total_tokens = 120
            prompt_cache_hit_tokens = 80
            prompt_cache_miss_tokens = 20

        class Resp:
            usage = Usage()

        summary = summarize_usage(Resp())
        assert summary["usage"]["prefix_cache_hit_rate"] == pytest.approx(0.8)

    def test_nested_details_cached_tokens(self):
        # OpenAI-style prompt_tokens_details.cached_tokens
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 60},
        }
        b = UsageBreakdown.from_dict(usage)
        assert b.prompt_cache_hit_tokens == 60
        assert b.prompt_cache_miss_tokens == 0


class TestPipelineMetrics:
    @pytest.mark.asyncio
    async def test_metrics_follow_pipeline(self):
        cache = make_cache_no_embed()

        async def upstream(**kwargs):
            return await fake_upstream(**kwargs)

        req = chat_request()
        await cache.chat_completion(request=req, upstream=upstream, stream=False)  # miss
        await cache.chat_completion(request=req, upstream=upstream, stream=False)  # exact

        snap = cache.metrics.snapshot()
        assert snap["layers"]["exact"]["hits"] == 1
        assert snap["layers"]["miss"]["misses"] == 1
        assert snap["hit_rate"] == 0.5

        stats = cache.stats_dict()
        assert stats["exact_hits"] == 1
        assert stats["misses"] == 1
