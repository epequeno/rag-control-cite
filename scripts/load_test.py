"""Load test the /query endpoint — p50/p95/p99 under sustained concurrency.

Usage:
  python scripts/load_test.py http://IP:8000 --rps 2 --duration 120
  python scripts/load_test.py http://IP:8000 --tenants eng-low,priv-high

Writes results/load_<timestamp>.json with the exact command line, git SHA,
instance sizes, and per-status counts. 404s are recorded separately from
200s, and the latency curve is computed over 200s only. Requests round-robin
over the tenant set (all four by default) so the ACL path — including
priv-high, the small-entitlement tenant — is exercised.

Auth: /query requires a bearer token when the service enforces auth. Supply
them via --tokens (JSON {"tenant": "token"}) or RCC_TENANT_TOKENS.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import httpx

TENANTS = ["eng-low", "sec-moderate", "priv-high", "sec-high"]
QUERIES = [
    "What does control AC-2 require?",
    "What are the audit and accountability requirements?",
    "What does control SC-8 require?",
    "What is the system and information integrity policy?",
    "What are the configuration management requirements?",
]


def one_request(
    client: httpx.Client, url: str, i: int, tenants: list[str], tokens: dict[str, str]
) -> tuple[float, int]:
    """One request; returns (latency_ms, status). 0 = transport error."""
    q = QUERIES[i % len(QUERIES)]
    tenant = tenants[i % len(tenants)]
    headers = {"Authorization": f"Bearer {tokens[tenant]}"} if tenant in tokens else {}
    t0 = time.perf_counter()
    try:
        resp = client.post(
            url,
            json={"query": q, "tenant_id": tenant},
            headers=headers,
            timeout=120.0,
        )
        return (time.perf_counter() - t0) * 1000, resp.status_code
    except httpx.HTTPError, OSError:
        return (time.perf_counter() - t0) * 1000, 0


def percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    return vals[min(int(len(vals) * p / 100), len(vals) - 1)]


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError, FileNotFoundError:
        return "no-git"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="e.g. http://1.2.3.4:8000")
    parser.add_argument("--rps", type=float, default=5.0)
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--tenants", default=",".join(TENANTS))
    parser.add_argument("--tokens", default=os.environ.get("RCC_TENANT_TOKENS", ""))
    parser.add_argument(
        "--instance-size",
        default=os.environ.get("RCC_INSTANCE_SIZE", "0.25 vCPU / 1 GB (Fargate)"),
    )
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    tenants = [t.strip() for t in args.tenants.split(",") if t.strip()]
    tokens = json.loads(args.tokens) if args.tokens else {}
    url = args.url.rstrip("/") + "/query"
    n = int(args.rps * args.duration)
    workers = max(4, int(args.rps * 6))

    print(f"load test: {n} requests @ {args.rps} rps for {args.duration}s → {url}")
    print(f"tenants: {tenants}   tokens: {'yes' if tokens else 'no (auth disabled?)'}")

    samples: list[tuple[float, int]] = []
    t_start = time.perf_counter()
    with httpx.Client() as client, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one_request, client, url, i, tenants, tokens) for i in range(n)]
        for fut in futures:
            samples.append(fut.result())
    wall = time.perf_counter() - t_start

    codes: dict[int, int] = {}
    for _, code in samples:
        codes[code] = codes.get(code, 0) + 1
    latencies_200 = [lat for lat, code in samples if code == 200]

    summary = {
        "command": " ".join(sys.argv),
        "git_sha": git_sha(),
        "instance_size": args.instance_size,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "url": url,
        "rps": args.rps,
        "duration_s": args.duration,
        "tenants": tenants,
        "n_requests": len(samples),
        "wall_seconds": round(wall, 1),
        "status_codes": {str(k): v for k, v in sorted(codes.items())},
        "n_200": codes.get(200, 0),
        "n_404": codes.get(404, 0),
        "n_errors": sum(v for k, v in codes.items() if k == 0 or k >= 500),
        "p50_ms": round(percentile(latencies_200, 50), 1),
        "p90_ms": round(percentile(latencies_200, 90), 1),
        "p95_ms": round(percentile(latencies_200, 95), 1),
        "p99_ms": round(percentile(latencies_200, 99), 1),
    }

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"load_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print()
    print(json.dumps(summary, indent=2))
    print(f"\nsaved: {out_path}")

    sys.exit(0 if summary["n_errors"] == 0 else 1)


if __name__ == "__main__":
    main()
