#!/usr/bin/env python3
"""
Benchmark script for PaddleOCR-VL throughput testing.

Tests different concurrency levels to find optimal batching.

Usage:
    python benchmark.py --url https://your-modal-url.modal.run --dir images/
"""

import argparse
import asyncio
import base64
import time
from pathlib import Path

import httpx


async def send_request(
    client: httpx.AsyncClient,
    base_url: str,
    image_b64: str,
    request_id: int,
) -> tuple[int, float, bool]:
    """Send a single request and return (id, latency, success)."""
    start = time.perf_counter()

    payload = {
        "model": "PaddleOCR-VL",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                    {"type": "text", "text": "Table Recognition:"},
                ],
            }
        ],
        "max_tokens": 4096,
        "temperature": 0.0,
    }

    try:
        response = await client.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=300.0,
        )
        response.raise_for_status()
        latency = time.perf_counter() - start
        return request_id, latency, True
    except Exception as e:
        latency = time.perf_counter() - start
        print(f"Request {request_id} failed: {e}")
        return request_id, latency, False


async def benchmark_concurrency(
    base_url: str,
    images_b64: list[str],
    concurrency: int,
    num_requests: int,
) -> dict:
    """Run benchmark with specific concurrency level."""
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_request(client, img_b64, req_id):
        async with semaphore:
            return await send_request(client, base_url, img_b64, req_id)

    async with httpx.AsyncClient(follow_redirects=True) as client:
        start = time.perf_counter()

        tasks = []
        for i in range(num_requests):
            img_b64 = images_b64[i % len(images_b64)]
            tasks.append(bounded_request(client, img_b64, i))

        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start

    # Calculate stats
    latencies = [r[1] for r in results if r[2]]
    successes = sum(1 for r in results if r[2])

    return {
        "concurrency": concurrency,
        "num_requests": num_requests,
        "successes": successes,
        "failures": num_requests - successes,
        "total_time": total_time,
        "throughput": successes / total_time if total_time > 0 else 0,
        "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
        "min_latency": min(latencies) if latencies else 0,
        "max_latency": max(latencies) if latencies else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark PaddleOCR-VL throughput")
    parser.add_argument("--url", type=str, required=True, help="Modal server URL")
    parser.add_argument("--dir", type=str, required=True, help="Directory of test images")
    parser.add_argument("--requests", type=int, default=50, help="Total requests to send")
    parser.add_argument(
        "--concurrency",
        type=str,
        default="1,10,25,50,100",
        help="Comma-separated concurrency levels to test",
    )
    args = parser.parse_args()

    # Load images
    extensions = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")
    images_b64 = []
    for path in sorted(Path(args.dir).iterdir()):
        if path.suffix.lower() in extensions:
            with open(path, "rb") as f:
                images_b64.append(base64.b64encode(f.read()).decode())

    if not images_b64:
        print(f"No images found in {args.dir}")
        return

    print(f"Loaded {len(images_b64)} images for benchmarking")
    print(f"Server: {args.url}")
    print(f"Total requests per test: {args.requests}")
    print()

    concurrency_levels = [int(c) for c in args.concurrency.split(",")]

    print("=" * 70)
    print(f"{'Concurrency':>12} {'Throughput':>12} {'Avg Latency':>12} {'Min':>8} {'Max':>8}")
    print("=" * 70)

    results = []
    for concurrency in concurrency_levels:
        print(f"Testing concurrency={concurrency}...", end=" ", flush=True)
        result = asyncio.run(
            benchmark_concurrency(args.url, images_b64, concurrency, args.requests)
        )
        results.append(result)

        print(
            f"\r{concurrency:>12} "
            f"{result['throughput']:>10.2f}/s "
            f"{result['avg_latency']:>10.2f}s "
            f"{result['min_latency']:>7.2f}s "
            f"{result['max_latency']:>7.2f}s"
        )

    print("=" * 70)

    # Find optimal
    best = max(results, key=lambda r: r["throughput"])
    print(f"\nOptimal concurrency: {best['concurrency']}")
    print(f"Peak throughput: {best['throughput']:.2f} pages/sec")
    print(f"At this rate, 1000 pages would take: {1000 / best['throughput']:.1f} seconds")

    # Cost estimate (L4 = $0.80/hr)
    cost_per_1000 = (1000 / best["throughput"]) * (0.80 / 3600)
    print(f"Estimated cost for 1000 pages: ${cost_per_1000:.4f}")


if __name__ == "__main__":
    main()
