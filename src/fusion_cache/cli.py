"""Command-line interface for fusion-cache.

Subcommands:

- ``fusion-cache serve``  — start the FastAPI gateway (host/port/redis).
- ``fusion-cache stats``  — print cache stats (from a running gateway or a
  fresh in-process cache).
- ``fusion-cache check``  — configuration / environment health check.

Example::

    fusion-cache serve --port 8000 --redis redis://localhost:6379/0
    fusion-cache stats --url http://localhost:8000
    fusion-cache check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fusion-cache", description="fusion-cache: LLM API fusion caching layer")
    sub = p.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="start the FastAPI gateway")
    serve.add_argument("--host", default=os.environ.get("FUSION_HOST", "0.0.0.0"))
    serve.add_argument("--port", type=int, default=int(os.environ.get("FUSION_PORT", "8000")))
    serve.add_argument("--redis", default=os.environ.get("REDIS_URL"), help="Redis URL (e.g. redis://localhost:6379/0)")
    serve.add_argument("--upstream", default=os.environ.get("FUSION_UPSTREAM_BASE_URL", "https://api.deepseek.com"))

    stats = sub.add_parser("stats", help="print cache stats")
    stats.add_argument("--url", default=os.environ.get("FUSION_GATEWAY_URL", "http://localhost:8000"), help="gateway base URL")

    sub.add_parser("check", help="configuration / environment health check")
    return p


def _cmd_serve(args: argparse.Namespace) -> int:
    if args.redis:
        os.environ["REDIS_URL"] = args.redis
    if args.upstream:
        os.environ["FUSION_UPSTREAM_BASE_URL"] = args.upstream
    from fusion_cache.gateway.app import create_app

    import uvicorn

    app = create_app()
    print(f"fusion-cache gateway listening on http://{args.host}:{args.port} (upstream={app.state.upstream_base_url})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    import urllib.request

    url = args.url.rstrip("/") + "/metrics"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310
            body = resp.read().decode()
    except Exception as exc:
        print(f"failed to fetch {url}: {exc}", file=sys.stderr)
        return 1
    # Try JSON first (Accept: application/json), fall back to Prometheus text.
    import httpx

    try:
        r = httpx.get(args.url.rstrip("/") + "/metrics", headers={"Accept": "application/json"}, timeout=10)
        if r.status_code == 200:
            try:
                data = r.json()
                print(json.dumps(data, indent=2, ensure_ascii=False))
                return 0
            except Exception:
                pass
    except Exception:
        pass
    print(body)
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    ok = True

    def report(name: str, status: bool, detail: str = "") -> None:
        nonlocal ok
        mark = "✅" if status else "❌"
        print(f"{mark} {name}" + (f" — {detail}" if detail else ""))
        ok = ok and status

    try:
        from fusion_cache import __version__
        report("fusion-cache version", True, __version__)
    except Exception as exc:
        report("fusion-cache import", False, str(exc))

    try:
        import fastapi  # noqa: F401
        report("fastapi", True)
    except Exception:
        report("fastapi", False, "install with: pip install fusion-cache[gateway]")

    try:
        import uvicorn  # noqa: F401
        report("uvicorn", True)
    except Exception:
        report("uvicorn", False, "install with: pip install fusion-cache[gateway]")

    try:
        import redis  # noqa: F401
        report("redis client", True)
    except Exception:
        report("redis client", False, "optional — install with: pip install fusion-cache[redis]")

    try:
        import numpy  # noqa: F401
        report("numpy", True)
    except Exception:
        report("numpy", False, "install with: pip install numpy")

    upstream = os.environ.get("FUSION_UPSTREAM_BASE_URL", "https://api.deepseek.com")
    report("upstream base url", bool(upstream), upstream)

    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("FUSION_UPSTREAM_API_KEY")
    report("upstream api key", bool(api_key), "set" if api_key else "not set (optional for cache-only use)")

    print("\n" + ("All checks passed." if ok else "Some checks failed."))
    return 0 if ok else 1


def main(argv: Optional[list[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.cmd == "serve":
        return _cmd_serve(args)
    if args.cmd == "stats":
        return _cmd_stats(args)
    if args.cmd == "check":
        return _cmd_check(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
