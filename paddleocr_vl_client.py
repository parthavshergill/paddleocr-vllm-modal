#!/usr/bin/env python3
"""
PaddleOCR-VL Client - Connect to the deployed warm service.

This script connects to the deployed Modal app and calls the warm service.
First call will be cold (~170s), subsequent calls within 15 min will be warm (~30-40s).

Prerequisites:
    modal deploy paddleocr_vl_service.py

Usage:
    # Basic usage - process all images in a directory
    python paddleocr_vl_client.py --input-dir images/

    # Filter images by filename substring
    python paddleocr_vl_client.py --input-dir images-70/ --filter SFMA

    # Limit number of images processed
    python paddleocr_vl_client.py --input-dir images/ --limit 10

    # Extract only table regions
    python paddleocr_vl_client.py --input-dir images/ --region-type table

    # Process in batches of 10 (helps identify slow images)
    python paddleocr_vl_client.py --input-dir images/ --batch-size 10

    # Full example with all options
    python paddleocr_vl_client.py \\
        --input-dir images-70/ \\
        --output-dir output/ \\
        --filter SFMA \\
        --limit 20 \\
        --region-type table \\
        --batch-size 10

Note: Concurrency is fixed at 192 (matching vLLM's --max-num-seqs).
"""

import argparse
import modal
import time
from pathlib import Path
from typing import Any


def load_images(input_dir: str, filter_str: str, limit: int) -> list[tuple[str, bytes]]:
    """
    Load images from directory with optional filtering and limiting.
    
    Args:
        input_dir: Directory containing input images
        filter_str: Only include images with this substring in filename
        limit: Maximum number of images to load (0 = no limit)
        
    Returns:
        List of (filename_stem, image_bytes) tuples
    """
    extensions = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")
    
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist")
    
    images = []
    for path in sorted(input_path.iterdir()):
        if path.suffix.lower() in extensions:
            if filter_str and filter_str not in path.name:
                continue
            with open(path, "rb") as f:
                images.append((path.stem, f.read()))
    
    if limit > 0:
        images = images[:limit]
    
    return images


def main():
    parser = argparse.ArgumentParser(
        description="PaddleOCR-VL Client - Extract text/tables from images using deployed warm service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input-dir images/
  %(prog)s --input-dir images/ --region-type table
  %(prog)s --input-dir images-70/ --filter SFMA --output-dir sfma-test
  %(prog)s --input-dir images-70/ --batch-size 10 --region-type table
        """,
    )
    
    # Input/Output arguments
    parser.add_argument(
        "--input-dir",
        default="images/",
        help="Directory containing input images (default: images/)",
    )
    parser.add_argument(
        "--output-dir",
        default="output-warm",
        help="Directory to save Markdown output (default: output-warm)",
    )
    
    # Filtering arguments
    parser.add_argument(
        "--filter",
        default="",
        help="Only process images with this substring in filename",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of images to process (0 = no limit)",
    )
    parser.add_argument(
        "--region-type",
        default="",
        help='Only process regions of this type (e.g., "table" or "table,header")',
    )
    
    # Batch processing
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Process images in batches of this size (0 = all at once). Helps identify slow images.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Number of retries per batch on failure (default: 1).",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=5.0,
        help="Seconds to wait between retries (multiplied by attempt number). Default: 5.0",
    )
    parser.add_argument(
        "--split-on-failure",
        action="store_true",
        help="If a batch fails after retries, split it into halves and try again to isolate problematic pages.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort the run immediately on the first batch failure.",
    )
    
    args = parser.parse_args()

    # Load images
    try:
        images = load_images(args.input_dir, args.filter, args.limit)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    if not images:
        print(f"No images found in {args.input_dir}")
        if args.filter:
            print(f"  (filter: '{args.filter}')")
        return

    # Print configuration
    print("=" * 60)
    print("PaddleOCR-VL Client")
    print("=" * 60)
    print(f"Input directory:  {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Images found:     {len(images)}")
    if args.filter:
        print(f"Filter:           '{args.filter}'")
    if args.limit > 0:
        print(f"Limit:            {args.limit}")
    if args.region_type:
        print(f"Region type:      '{args.region_type}'")
    if args.batch_size > 0:
        print(f"Batch size:       {args.batch_size}")
    print("Concurrency:      128 (fixed at container startup)")
    print("=" * 60)

    print("\nConnecting to deployed warm service...")
    
    # Connect to the deployed app
    try:
        PaddleOCRVLService = modal.Cls.from_name("paddleocr-vl-warm", "PaddleOCRVLService")
    except modal.exception.NotFoundError:
        print("\nError: Deployed app 'paddleocr-vl-warm' not found!")
        print("Please deploy first with: modal deploy paddleocr_vl_service.py")
        return

    service = PaddleOCRVLService()
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    overall_start = time.perf_counter()
    all_results = []
    batch_times = []
    warmup_time = 0
    
    def fmt_bytes(b):
        if b >= 1024 * 1024:
            return f"{b / (1024*1024):.2f}MB"
        elif b >= 1024:
            return f"{b / 1024:.2f}KB"
        return f"{b}B"

    def _call_batch_with_retries(
        batch: list[tuple[str, bytes]],
        *,
        batch_label: str,
        attempt: int = 0,
    ) -> tuple[list[tuple[str, str]], dict | None, dict | None]:
        """
        Returns (results, stats, error_payload).
        - error_payload is a plain dict (stringifiable) when failures occur.
        """
        try:
            response = service.process_images.remote(
                batch,
                region_type=args.region_type,
            )
        except modal.exception.ExecutionError as e:
            # NOTE: This can include "Could not deserialize remote exception..."
            # when the remote exception class isn't installed locally.
            err_payload = {"type": "ExecutionError", "message": str(e)}
            if attempt < args.retries:
                wait_s = args.retry_backoff * (attempt + 1)
                print(f"[ERROR] {batch_label} failed ({err_payload['type']}), retrying in {wait_s:.1f}s...")
                time.sleep(wait_s)
                return _call_batch_with_retries(batch, batch_label=batch_label, attempt=attempt + 1)
            return [], None, err_payload
        except Exception as e:
            err_payload = {"type": type(e).__name__, "message": str(e)}
            if attempt < args.retries:
                wait_s = args.retry_backoff * (attempt + 1)
                print(f"[ERROR] {batch_label} failed ({err_payload['type']}), retrying in {wait_s:.1f}s...")
                time.sleep(wait_s)
                return _call_batch_with_retries(batch, batch_label=batch_label, attempt=attempt + 1)
            return [], None, err_payload

        # Service may return structured errors instead of raising (preferred).
        if isinstance(response, dict) and response.get("error"):
            err_payload = response.get("error")
            if attempt < args.retries:
                wait_s = args.retry_backoff * (attempt + 1)
                print(f"[ERROR] {batch_label} returned error, retrying in {wait_s:.1f}s...")
                print(f"        {err_payload}")
                time.sleep(wait_s)
                return _call_batch_with_retries(batch, batch_label=batch_label, attempt=attempt + 1)
            return response.get("results", []), response.get("stats"), err_payload

        return response["results"], response.get("stats"), None

    def _process_batch_maybe_split(
        batch: list[tuple[str, bytes]],
        *,
        batch_label: str,
    ) -> tuple[list[tuple[str, str]], dict | None, dict | None]:
        results, stats, err = _call_batch_with_retries(batch, batch_label=batch_label)
        if not err:
            return results, stats, None

        # Optionally split and retry to isolate problematic pages.
        if args.split_on_failure and len(batch) > 1:
            mid = len(batch) // 2
            left, right = batch[:mid], batch[mid:]
            print(f"[WARN] {batch_label} failed; splitting into {len(left)} + {len(right)} and retrying...")
            left_res, left_stats, left_err = _process_batch_maybe_split(left, batch_label=f"{batch_label} (left)")
            right_res, right_stats, right_err = _process_batch_maybe_split(right, batch_label=f"{batch_label} (right)")

            combined_res = []
            combined_res.extend(left_res)
            combined_res.extend(right_res)
            # Stats are not meaningfully mergeable here; return None and bubble up errors if any.
            combined_err: dict[str, Any] | None = None
            if left_err or right_err:
                combined_err = {"left_error": left_err, "right_error": right_err}
            return combined_res, None, combined_err

        return results, stats, err
    
    # Batch processing mode
    if args.batch_size > 0:
        # Warmup request with first image
        print("\n[WARMUP] Sending 1 image to warm up server...")
        warmup_start = time.perf_counter()
        warmup_response = service.process_images.remote(images[:1], region_type=args.region_type)
        warmup_time = time.perf_counter() - warmup_start
        warmup_results = warmup_response["results"]
        warmup_stats = warmup_response.get("stats")
        warmup_err = warmup_response.get("error") if isinstance(warmup_response, dict) else None
        print(f"[WARMUP] Complete in {warmup_time:.2f}s")
        if warmup_err:
            print(f"  [WARMUP ERROR] {warmup_err}")
            if args.fail_fast:
                return
        if warmup_stats:
            print(f"  Regions: {warmup_stats['num_regions']}, Avg: {warmup_stats['avg_width']:.0f}x{warmup_stats['avg_height']:.0f}px, Mem: {fmt_bytes(warmup_stats['avg_memory_bytes'])}")
        all_results.extend(warmup_results)
        
        # Process remaining images in batches
        remaining = images[1:]
        num_batches = (len(remaining) + args.batch_size - 1) // args.batch_size
        
        for i in range(0, len(remaining), args.batch_size):
            batch = remaining[i:i + args.batch_size]
            batch_num = i // args.batch_size + 1
            batch_names = [name for name, _ in batch]
            
            print(f"\n[BATCH {batch_num}/{num_batches}] Processing {len(batch)} images...")
            print(f"  Images: {batch_names}")
            
            batch_start = time.perf_counter()
            results, stats, err = _process_batch_maybe_split(batch, batch_label=f"BATCH {batch_num}/{num_batches}")
            batch_time = time.perf_counter() - batch_start
            
            # Store batch info with stats
            batch_info = {
                "num": batch_num,
                "images": len(batch),
                "time": batch_time,
                "names": batch_names,
                "stats": stats,
                "error": err,
            }
            batch_times.append(batch_info)
            
            stats_str = ""
            if stats:
                stats_str = f", {stats['num_regions']} regions, avg {stats['avg_width']:.0f}x{stats['avg_height']:.0f}px, {fmt_bytes(stats['avg_memory_bytes'])}"
            
            if err:
                print(f"[BATCH {batch_num}/{num_batches}] FAILED in {batch_time:.2f}s (error={err})")
                if args.fail_fast:
                    return
            else:
                print(f"[BATCH {batch_num}/{num_batches}] Complete in {batch_time:.2f}s ({len(results)} items, {len(results)/batch_time:.2f} items/sec{stats_str})")
            all_results.extend(results)
    else:
        # Single request mode (original behavior)
        start = time.perf_counter()
        response = service.process_images.remote(images, region_type=args.region_type)
        batch_time = time.perf_counter() - start
        all_results = response["results"]
        stats = response.get("stats")
        batch_times.append({
            "num": 1,
            "images": len(images),
            "time": batch_time,
            "names": [n for n, _ in images],
            "stats": stats,
            "error": response.get("error") if isinstance(response, dict) else None,
        })

    total_time = time.perf_counter() - overall_start

    # Save results
    for name, md_content in all_results:
        md_path = output_path / f"{name}.md"
        with open(md_path, "w") as f:
            f.write(md_content)
        print(f"Saved: {md_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"Total time:       {total_time:.2f}s")
    print(f"Items processed:  {len(all_results)}")
    if args.region_type:
        print(f"Region filter:    {args.region_type}")
    print(f"Throughput:       {len(all_results) / total_time:.2f} items/sec")
    print(f"Output directory: {output_path.absolute()}")
    
    # Batch timing breakdown
    if args.batch_size > 0:
        print(f"\n{'─'*60}")
        print("BATCH TIMING BREAKDOWN")
        print("─" * 60)
        if warmup_time:
            print(f"  Warmup:  {warmup_time:.2f}s (1 image)")
        for batch_info in batch_times:
            batch_num = batch_info["num"]
            batch_size = batch_info["images"]
            batch_time = batch_info["time"]
            batch_names = batch_info["names"]
            stats = batch_info.get("stats")
            
            throughput = batch_size / batch_time if batch_time > 0 else 0
            
            stats_str = ""
            if stats:
                stats_str = f" | {stats['num_regions']} regions, avg {stats['avg_width']:.0f}x{stats['avg_height']:.0f}px, {fmt_bytes(stats['avg_memory_bytes'])}"
            
            print(f"  Batch {batch_num}: {batch_time:.2f}s ({batch_size} images, {throughput:.2f} items/sec){stats_str}")
            # Flag slow batches
            if throughput < 0.5:
                print(f"    ⚠️  SLOW BATCH - Images: {batch_names}")
    
    print("=" * 60)
    print("\nTIP: Run again within 15 min for warm start (skip ~170s server startup)")


if __name__ == "__main__":
    main()
