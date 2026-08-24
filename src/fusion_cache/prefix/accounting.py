"""L3 prefix-cache accounting: prompt_cache_hit/miss_tokens → $ saved.

DeepSeek (and other OpenAI-compatible providers with automatic prefix
caching) return ``usage.prompt_cache_hit_tokens`` and
``usage.prompt_cache_miss_tokens``.  The hit tokens are billed at the
cache-hit input price; the miss tokens at the regular input price.  The
*difference* between those two billings is the money the upstream prefix
cache saved us — and we report it as the L3 layer's contribution.

Savings formula::

    saved_usd = prompt_cache_hit_tokens * (input_miss_per_1m - input_hit_per_1m) / 1_000_000
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Union

from ..config import DeepSeekPriceModel


@dataclass
class UsageBreakdown:
    """Token usage extracted from an upstream response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_cache_hit_tokens: int = 0
    prompt_cache_miss_tokens: int = 0

    @classmethod
    def from_dict(cls, usage: Mapping[str, Any]) -> "UsageBreakdown":
        """Extract usage fields from a usage dict (already extracted)."""
        details = usage.get("prompt_tokens_details") or {}
        return cls(
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
            prompt_cache_hit_tokens=int(
                usage.get("prompt_cache_hit_tokens") or details.get("cached_tokens") or 0
            ),
            prompt_cache_miss_tokens=int(usage.get("prompt_cache_miss_tokens") or 0),
        )

    @classmethod
    def from_response(cls, response: Any) -> "UsageBreakdown":
        """Extract usage fields from an OpenAI-style response object or dict.

        Also accepts a bare usage dict (e.g. what ``summarize_usage`` receives
        when the caller already extracted usage).
        """
        if isinstance(response, Mapping) and "prompt_tokens" in response:
            # it's a bare usage dict
            return cls.from_dict(response)
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, Mapping):
            usage = response.get("usage")
        if usage is None:
            return cls()
        if isinstance(usage, Mapping):
            details = usage.get("prompt_tokens_details") or {}
            return cls(
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens") or 0),
                total_tokens=int(usage.get("total_tokens") or 0),
                prompt_cache_hit_tokens=int(
                    usage.get("prompt_cache_hit_tokens") or details.get("cached_tokens") or 0
                ),
                prompt_cache_miss_tokens=int(usage.get("prompt_cache_miss_tokens") or 0),
            )
        details = getattr(usage, "prompt_tokens_details", None) or {}
        return cls(
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
            prompt_cache_hit_tokens=int(
                getattr(usage, "prompt_cache_hit_tokens", 0) or getattr(details, "cached_tokens", 0) or 0
            ),
            prompt_cache_miss_tokens=int(getattr(usage, "prompt_cache_miss_tokens", 0) or 0),
        )

    @property
    def prefix_cache_hit_rate(self) -> float:
        """Fraction of prompt tokens that hit the upstream prefix cache."""
        prompt = self.prompt_tokens
        if prompt <= 0:
            return 0.0
        return min(1.0, self.prompt_cache_hit_tokens / prompt)


@dataclass
class CostAccounting:
    """Money saved via prefix-cache hits, plus a raw-cost estimate."""

    saved_usd: float
    input_miss_usd: float
    input_hit_usd: float
    output_usd: float
    total_usd: float

    def as_dict(self) -> Dict[str, Union[float, int]]:
        return {
            "saved_usd": round(self.saved_usd, 6),
            "input_miss_usd": round(self.input_miss_usd, 6),
            "input_hit_usd": round(self.input_hit_usd, 6),
            "output_usd": round(self.output_usd, 6),
            "total_usd": round(self.total_usd, 6),
        }


def account_prefix_costs(
    breakdown: UsageBreakdown,
    price_model: Optional[DeepSeekPriceModel] = None,
) -> CostAccounting:
    """Compute $ saved and cost components for one upstream response."""
    pm = price_model or DeepSeekPriceModel.deepseek_default()
    per_miss = pm.input_miss_per_1m / 1_000_000.0
    per_hit = pm.input_hit_per_1m / 1_000_000.0
    per_out = pm.output_per_1m / 1_000_000.0

    miss_tokens = breakdown.prompt_cache_miss_tokens
    # If the provider only reports prompt_tokens (no cache fields), attribute
    # everything to miss — honest and conservative.
    if miss_tokens == 0 and breakdown.prompt_tokens > 0 and breakdown.prompt_cache_hit_tokens == 0:
        miss_tokens = breakdown.prompt_tokens

    input_miss_usd = miss_tokens * per_miss
    input_hit_usd = breakdown.prompt_cache_hit_tokens * per_hit
    output_usd = breakdown.completion_tokens * per_out
    total_usd = input_miss_usd + input_hit_usd + output_usd

    # What we *would* have paid if every prompt token was a miss:
    would_have_been = (breakdown.prompt_cache_hit_tokens + miss_tokens) * per_miss + output_usd
    saved_usd = max(0.0, would_have_been - total_usd)
    return CostAccounting(
        saved_usd=saved_usd,
        input_miss_usd=input_miss_usd,
        input_hit_usd=input_hit_usd,
        output_usd=output_usd,
        total_usd=total_usd,
    )


def summarize_usage(response: Any, price_model: Optional[DeepSeekPriceModel] = None) -> Dict[str, Any]:
    """One-call summary: breakdown + accounting for a raw upstream response."""
    breakdown = UsageBreakdown.from_response(response)
    costs = account_prefix_costs(breakdown, price_model)
    return {
        "usage": {
            "prompt_tokens": breakdown.prompt_tokens,
            "completion_tokens": breakdown.completion_tokens,
            "total_tokens": breakdown.total_tokens,
            "prompt_cache_hit_tokens": breakdown.prompt_cache_hit_tokens,
            "prompt_cache_miss_tokens": breakdown.prompt_cache_miss_tokens,
            "prefix_cache_hit_rate": round(breakdown.prefix_cache_hit_rate, 4),
        },
        "cost": costs.as_dict(),
    }
