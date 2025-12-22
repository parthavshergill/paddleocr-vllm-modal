#!/usr/bin/env python3
"""
Async concurrent client for PaddleOCR-VL table extraction.

Fires all requests concurrently to maximize vLLM's batching efficiency.

Usage:
    # First deploy the server
    modal deploy table_extractor.py

    # Then run extraction (get URL from deploy output)
    python extract_tables.py --url https://your-modal-url.modal.run --dir images/
    python extract_tables.py --url https://your-modal-url.modal.run --image images/table.png
"""

import argparse
import asyncio
import base64
import os
import time
from pathlib import Path

import httpx


async def extract_table(
    client: httpx.AsyncClient,
    base_url: str,
    image_data: bytes,
    image_name: str,
) -> tuple[str, str]:
    """Extract table from a single image."""
    image_b64 = base64.b64encode(image_data).decode()

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

    response = await client.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        timeout=300.0,  # 5 min timeout per request
    )
    response.raise_for_status()

    result = response.json()
    content = result["choices"][0]["message"]["content"]
    return image_name, content


async def extract_tables_concurrent(
    base_url: str,
    images: list[tuple[str, bytes]],  # (name, data) pairs
    max_concurrent: int = 100,
) -> list[tuple[str, str]]:
    """Extract tables from multiple images concurrently."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_extract(client, name, data):
        async with semaphore:
            return await extract_table(client, base_url, data, name)

    async with httpx.AsyncClient(follow_redirects=True) as client:
        tasks = [bounded_extract(client, name, data) for name, data in images]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions and report them
    successful = []
    for result in results:
        if isinstance(result, Exception):
            print(f"Error: {result}")
        else:
            successful.append(result)

    return successful


def main():
    parser = argparse.ArgumentParser(description="Extract tables using PaddleOCR-VL")
    parser.add_argument("--url", type=str, required=True, help="Modal server URL")
    parser.add_argument("--image", type=str, help="Single image path")
    parser.add_argument("--dir", type=str, help="Directory of images")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--concurrent", type=int, default=100, help="Max concurrent requests")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Collect images
    images = []
    if args.dir:
        extensions = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")
        for path in sorted(Path(args.dir).iterdir()):
            if path.suffix.lower() in extensions:
                with open(path, "rb") as f:
                    images.append((path.stem, f.read()))
    elif args.image:
        path = Path(args.image)
        with open(path, "rb") as f:
            images.append((path.stem, f.read()))
    else:
        print("Provide --image or --dir")
        return

    print(f"Processing {len(images)} images with {args.concurrent} concurrent requests...")
    print(f"Server: {args.url}")

    # Run extraction
    start = time.perf_counter()
    results = asyncio.run(
        extract_tables_concurrent(args.url, images, args.concurrent)
    )
    elapsed = time.perf_counter() - start

    # Save results
    for name, content in results:
        output_path = Path(args.output) / f"{name}.md"
        with open(output_path, "w") as f:
            f.write(content)
        print(f"Saved: {output_path}")

    # Print stats
    print(f"\n{'='*50}")
    print(f"Completed: {len(results)}/{len(images)} images")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Throughput: {len(results) / elapsed:.2f} pages/sec")
    print(f"Avg latency: {elapsed / len(results):.2f}s per image")


if __name__ == "__main__":
    main()
