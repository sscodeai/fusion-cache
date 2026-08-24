"""pydantic v2 configuration for fusion-cache.

Everything a user is likely to tune lives here: layer toggles, the semantic
similarity threshold, TTLs, the DeepSeek price model used by the L3 prefix
accounting, and the embedder used by the L2 semantic layer.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional

from pydantic import BaseModel, Field, field_validator

DEFAULT_PRICE_MODEL: Dict[str, float] = {
    # DeepSeek off-peak pricing snapshot (2026-08, USD per 1M tokens).
    # Cache-hit input tokens are priced separately from miss tokens.
    "input_miss_per_1m": 0.22,
    "input_hit_per_1m": 0.007,
    "output_per_1m": 1.1,
}


class DeepSeekPriceModel(BaseModel):
    """Price snapshot used by :mod:`fusion_cache.prefix.accounting`.

    ``input_hit`` is the per-token price of *cache-hit* input tokens
    (``prompt_cache_hit_tokens``) while ``input_miss`` is the price of
    uncached input tokens (``prompt_cache_miss_tokens``).
    """

    input_miss_per_1m: float = Field(default=0.22, gt=0.0)
    input_hit_per_1m: float = Field(default=0.007, gt=0.0)
    output_per_1m: float = Field(default=1.1, gt=0.0)

    @classmethod
    def deepseek_default(cls) -> "DeepSeekPriceModel":
        return cls(**DEFAULT_PRICE_MODEL)


class EmbedderConfig(BaseModel):
    """Settings for the OpenAI-compatible embedding endpoint (L2 semantic)."""

    base_url: str = Field(default="https://api.deepseek.com", description="Any OpenAI-compatible base URL.")
    api_key: str = Field(default="", description="Defaults to the DEEPSEEK_API_KEY / OPENAI_API_KEY env var.")
    model: str = Field(default="deepseek-embedding", description="Embedding model name served at /embeddings.")
    timeout: float = Field(default=10.0, gt=0.0)


class FusionCacheConfig(BaseModel):
    """Top-level configuration for :class:`fusion_cache.core.pipeline.FusionCache`."""

    # --- layer toggles ---------------------------------------------------
    enable_exact: bool = True
    enable_semantic: bool = True
    enable_prefix_accounting: bool = True

    # --- TTLs (seconds) --------------------------------------------------
    exact_ttl: float = Field(default=3600.0, ge=0.0)
    semantic_ttl: float = Field(default=7200.0, ge=0.0)
    semantic_grace_ttl: float = Field(default=3600.0, ge=0.0)

    # --- L2 semantic ------------------------------------------------------
    similarity_threshold: float = Field(default=0.93, ge=0.0, le=1.0)
    semantic_max_entries: int = Field(default=10_000, ge=1)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)

    # --- L1 exact ----------------------------------------------------------
    max_entries: int = Field(default=10_000, ge=1)

    # --- L3 prefix accounting ------------------------------------------------
    price_model: DeepSeekPriceModel = Field(default_factory=DeepSeekPriceModel.deepseek_default)

    # --- upstream ------------------------------------------------------------
    upstream_timeout: float = Field(default=60.0, gt=0.0)

    @field_validator("embedder", mode="before")
    @classmethod
    def _embedder_dict(cls, v):
        return EmbedderConfig(**v) if isinstance(v, dict) else v

    @field_validator("price_model", mode="before")
    @classmethod
    def _price_dict(cls, v):
        return DeepSeekPriceModel(**v) if isinstance(v, dict) else v

    @lru_cache(maxsize=1)
    def cache_key(self) -> str:
        """Stable fingerprint of tuning-relevant fields (used by tests/benchmarks)."""
        return (
            f"{self.enable_exact}|{self.enable_semantic}|{self.enable_prefix_accounting}|"
            f"{self.exact_ttl}|{self.semantic_ttl}|{self.similarity_threshold}|"
            f"{self.price_model.input_miss_per_1m}|{self.price_model.input_hit_per_1m}|"
            f"{self.price_model.output_per_1m}"
        )

    @classmethod
    def from_env(cls) -> "FusionCacheConfig":
        """Build a config from environment variables (best-effort, no hard deps)."""
        import os

        def _f(name: str, default: float) -> float:
            raw = os.getenv(name)
            try:
                return float(raw) if raw is not None else default
            except ValueError:
                return default

        embedder = EmbedderConfig(
            base_url=os.getenv("FUSION_EMBED_BASE_URL", EmbedderConfig().base_url),
            api_key=os.getenv("FUSION_EMBED_API_KEY", os.getenv("DEEPSEEK_API_KEY", os.getenv("OPENAI_API_KEY", ""))),
            model=os.getenv("FUSION_EMBED_MODEL", EmbedderConfig().model),
            timeout=_f("FUSION_EMBED_TIMEOUT", EmbedderConfig().timeout),
        )
        return cls(
            enable_exact=os.getenv("FUSION_ENABLE_EXACT", "1").lower() not in {"0", "false", "off", "no"},
            enable_semantic=os.getenv("FUSION_ENABLE_SEMANTIC", "1").lower() not in {"0", "false", "off", "no"},
            enable_prefix_accounting=os.getenv("FUSION_ENABLE_PREFIX_ACCOUNTING", "1").lower()
            not in {"0", "false", "off", "no"},
            exact_ttl=_f("FUSION_EXACT_TTL", 3600.0),
            semantic_ttl=_f("FUSION_SEMANTIC_TTL", 7200.0),
            semantic_grace_ttl=_f("FUSION_SEMANTIC_GRACE_TTL", 3600.0),
            similarity_threshold=_f("FUSION_SIM_THRESHOLD", 0.93),
            max_entries=int(_f("FUSION_MAX_ENTRIES", 10_000)),
            semantic_max_entries=int(_f("FUSION_SEMANTIC_MAX_ENTRIES", 10_000)),
            embedder=embedder,
            upstream_timeout=_f("FUSION_UPSTREAM_TIMEOUT", 60.0),
        )

    @property
    def effective_embed_base_url(self) -> Optional[str]:
        return self.embedder.base_url or None
