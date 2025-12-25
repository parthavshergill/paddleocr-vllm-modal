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
    print(f"Concurrency:      192 (fixed at container startup)")
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
    
    # Batch processing mode
    if args.batch_size > 0:
        # Warmup request with first image
        print(f"\n[WARMUP] Sending 1 image to warm up server...")
        warmup_start = time.perf_counter()
        warmup_response = service.process_images.remote(
            images[:1],
            region_type=args.region_type,
        )
        warmup_time = time.perf_counter() - warmup_start
        warmup_results = warmup_response["results"]
        warmup_stats = warmup_response.get("stats")
        print(f"[WARMUP] Complete in {warmup_time:.2f}s")
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
            response = service.process_images.remote(
                batch,
                region_type=args.region_type,
            )
            batch_time = time.perf_counter() - batch_start
            
            results = response["results"]
            stats = response.get("stats")
            
            # Store batch info with stats
            batch_info = {
                "num": batch_num,
                "images": len(batch),
                "time": batch_time,
                "names": batch_names,
                "stats": stats,
            }
            batch_times.append(batch_info)
            
            stats_str = ""
            if stats:
                stats_str = f", {stats['num_regions']} regions, avg {stats['avg_width']:.0f}x{stats['avg_height']:.0f}px, {fmt_bytes(stats['avg_memory_bytes'])}"
            
            print(f"[BATCH {batch_num}/{num_batches}] Complete in {batch_time:.2f}s ({len(results)} items, {len(results)/batch_time:.2f} items/sec{stats_str})")
            all_results.extend(results)
    else:
        # Single request mode (original behavior)
        start = time.perf_counter()
        response = service.process_images.remote(
            images,
            region_type=args.region_type,
        )
        batch_time = time.perf_counter() - start
        all_results = response["results"]
        stats = response.get("stats")
        batch_times.append({
            "num": 1,
            "images": len(images),
            "time": batch_time,
            "names": [n for n, _ in images],
            "stats": stats,
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
