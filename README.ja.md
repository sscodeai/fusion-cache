# fusion-cache

[English](README.md) | [日本語](README.ja.md)

LLM API のためのフレームワーク非依存キャッシュレイヤーです。OpenAI 互換クライアントをラップするか、OpenAI 互換ゲートウェイとして配置するだけで、繰り返しリクエストのコストとレイテンシを削減できます。

fusion-cache は 3 つの層を順番に使います。

| レイヤー | 役割 | ヒット時のコスト |
|---|---|---|
| L1 exact | 完全一致するリクエストを即時リプレイ | ほぼ 0 ms |
| L2 semantic | 言い換えられた近いリクエストを構造ガード付きで再利用 | 埋め込み 1 回 |
| L3 prefix | 上流プロバイダの prefix cache 割引を記録して可視化 | 追加呼び出しなし |

## 特長

- OpenAI / DeepSeek などの OpenAI 互換 API に対応
- 同一リクエストは canonicalized SHA-256 key で正確にキャッシュ
- 近い意味のリクエストは embedding + cosine similarity で再利用
- model、stream、tools、response_format など、出力に影響する設定を guardrail で分離
- `stream=True` も buffer-then-replay 方式でキャッシュ可能
- RedisStore による複数インスタンス共有キャッシュに対応
- FastAPI ゲートウェイ、CLI、MCP server、Prometheus 形式メトリクスを同梱

## インストール

```bash
pip install fusion-cache
```

開発環境ではリポジトリから editable install できます。

```bash
pip install -e .
pip install -e ".[test,gateway,redis,mcp]"
```

Python 3.10 以上が必要です。

## クイックスタート

```python
import openai
from fusion_cache import FusionCache
from fusion_cache.wrapper.openai import CachedOpenAI

cache = FusionCache()
client = CachedOpenAI(
    openai.OpenAI(
        api_key="sk-...",
        base_url="https://api.deepseek.com",
    ),
    cache=cache,
)

resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "Explain prefix caching"}],
)

print(cache.stats_dict())
```

1 回目は上流 API に送信され、2 回目以降の同一リクエストは L1 exact cache から即時に返されます。L2 semantic cache を有効にするには、OpenAI 互換の `/embeddings` endpoint と API key を設定します。

## L2 semantic cache

```python
from fusion_cache import FusionCache, FusionCacheConfig

cache = FusionCache(FusionCacheConfig(
    similarity_threshold=0.93,
    embedder={
        "base_url": "https://api.deepseek.com",
        "api_key": "sk-...",
        "model": "deepseek-embedding",
    },
))
```

`similarity_threshold` を高くすると、ヒットは少なくなりますが誤ヒットのリスクを下げられます。埋め込み API key がない場合、L2 は自動的にスキップされ、L1 と L3 はそのまま動作します。

## ゲートウェイとして使う

fusion-cache は OpenAI 互換の reverse proxy としても動作します。

```bash
fusion-cache serve --port 8000
```

Redis を使うと複数インスタンス間でキャッシュを共有できます。

```bash
fusion-cache serve --port 8000 --redis redis://localhost:6379/0
```

主な endpoint:

| Endpoint | 説明 |
|---|---|
| `POST /v1/chat/completions` | キャッシュ付き OpenAI 互換 chat completions |
| `GET /v1/models` | 上流 models endpoint の passthrough |
| `GET /metrics` | Prometheus 形式または JSON のメトリクス |
| `GET /dashboard` | 簡易 HTML dashboard |
| `POST /v1/cache/invalidate` | キャッシュの削除 |

## 設定

主要な設定は `FusionCacheConfig` にまとまっています。

```python
FusionCacheConfig(
    enable_exact=True,
    enable_semantic=True,
    enable_prefix_accounting=True,
    exact_ttl=3600.0,
    semantic_ttl=7200.0,
    similarity_threshold=0.93,
    max_entries=10_000,
    semantic_max_entries=10_000,
)
```

環境変数でも設定できます。代表例:

| 環境変数 | 説明 |
|---|---|
| `FUSION_UPSTREAM_BASE_URL` | 上流 API base URL |
| `FUSION_UPSTREAM_API_KEY` | 上流 API key |
| `FUSION_GATEWAY_API_KEY` | ゲートウェイ認証 key |
| `FUSION_ENABLE_SEMANTIC` | L2 semantic cache の有効化 |
| `FUSION_SIM_THRESHOLD` | 類似度しきい値 |
| `REDIS_URL` | RedisStore の接続先 |

## ストリーミング

`stream=True` のリクエストは、上流 stream を一度最後まで buffer してから保存し、呼び出し元には chunk iterator としてリプレイします。初回の TTFB は上がりますが、以降の同一 stream リクエストはキャッシュから再生できます。

## AI agent との連携

AI coding agent やツールから使う場合は、agent の `OPENAI_BASE_URL` を fusion-cache gateway に向けるだけで、既存コードを大きく変えずにキャッシュを挟めます。MCP server も同梱しており、agent から cache stats や invalidate を扱えます。

```bash
pip install "fusion-cache[mcp]"
fusion-cache-mcp
```

詳しくは [Agent Integration](docs/agent-integration.md) を参照してください。

## ベンチマーク

README の英語版には、実ワークロードでのヒット率、レイテンシ、prefix-cache accounting の詳細なベンチマークが記載されています。代表値として、L1/L2 によって 60 リクエスト中 58 件をキャッシュから返し、P95 latency を秒単位からミリ秒未満へ下げる結果が示されています。

## ライセンス

Apache-2.0
