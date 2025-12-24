#!/usr/bin/env python3
"""
Count layout regions per image using PP-DocLayoutV2 layout detection only.

This script runs only the layout detection model (no VLM inference) to count
the number of regions (tables, text blocks, figures, etc.) detected per image.
This helps understand parallelism requirements for batch processing.

Usage:
    modal run count_regions.py --input-dir images-70/
    modal run count_regions.py --input-dir images/ --limit 10
"""

import modal
import time
import tempfile
from pathlib import Path
from collections import Counter

app = modal.App("paddleocr-region-counter")

# Lighter image - only need paddlepaddle + paddleocr (no vllm)
layout_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
    )
    # Install PaddlePaddle GPU from official index
    .pip_install(
        "paddlepaddle-gpu==3.2.1",
        index_url="https://www.paddlepaddle.org.cn/packages/stable/cu126/",
    )
    # Install PaddleOCR with doc-parser for layout detection
    .pip_install(
        "paddleocr[doc-parser]>=3.0.0",
        "pillow",
    )
)

# Cache for model weights
model_cache = modal.Volume.from_name("paddleocr-vl-cache", create_if_missing=True)


@app.function(
    gpu="L4",
    image=layout_image,
    volumes={"/root/.cache/huggingface": model_cache},
    timeout=1800,
)
def count_regions(images_data: list[tuple[str, bytes]]) -> list[dict]:
    """
    Count layout regions per image using PP-DocLayoutV2.
    
    Returns list of dicts with:
        - name: image name
        - num_regions: count of regions
        - labels: list of region type labels
        - region_sizes: list of (width, height, bytes_rgb_f32) per region
    """
    from paddleocr import LayoutDetection
    import os
    
    print("=" * 60)
    print("Layout Detection Region Counter")
    print("=" * 60)
    
    print("\n[1/2] Loading PP-DocLayoutV2 model...")
    start_load = time.perf_counter()
    
    # Initialize layout detection model
    layout_model = LayoutDetection(
        model_name="PP-DocLayoutV2",
        device="gpu:0",
    )
    
    load_time = time.perf_counter() - start_load
    print(f"[OK] Model loaded in {load_time:.1f}s")
    
    # Process all images
    print(f"\n[2/2] Processing {len(images_data)} images...")
    results = []
    temp_files = []
    
    try:
        # Write images to temp files
        for name, img_bytes in images_data:
            temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            temp_file.write(img_bytes)
            temp_file.close()
            temp_files.append((name, temp_file.name))
        
        # Process each image
        batch_start = time.perf_counter()
        debug_printed = False
        
        for i, (name, temp_path) in enumerate(temp_files):
            # Run layout detection
            layout_output = layout_model.predict(temp_path)
            
            # Debug: Print structure of first non-empty result
            if not debug_printed and layout_output:
                print("\n" + "=" * 60)
                print("DEBUG: Layout output structure")
                print("=" * 60)
                print(f"  layout_output type: {type(layout_output)}")
                print(f"  layout_output length: {len(layout_output)}")
                if layout_output:
                    first = layout_output[0]
                    print(f"  layout_output[0] type: {type(first)}")
                    
                    # Check if it's a dict
                    if isinstance(first, dict):
                        print(f"  layout_output[0] keys: {list(first.keys())}")
                        if 'boxes' in first:
                            print(f"  layout_output[0]['boxes'] type: {type(first['boxes'])}")
                            if first['boxes']:
                                box = first['boxes'][0]
                                print(f"  first box type: {type(box)}")
                                if isinstance(box, dict):
                                    print(f"  first box keys: {list(box.keys())}")
                                    print(f"  first box values: {box}")
                                else:
                                    box_attrs = [a for a in dir(box) if not a.startswith('_')]
                                    print(f"  first box attributes: {box_attrs}")
                    else:
                        # Show all non-private attributes
                        attrs = [a for a in dir(first) if not a.startswith('_')]
                        print(f"  layout_output[0] attributes: {attrs}")
                        
                        if hasattr(first, 'boxes') and first.boxes:
                            box = first.boxes[0]
                            print(f"\n  first.boxes[0] type: {type(box)}")
                            box_attrs = [a for a in dir(box) if not a.startswith('_')]
                            print(f"  first.boxes[0] attributes: {box_attrs}")
                            # Try to print values of common label attributes
                            for attr in ['label', 'cls_name', 'category', 'class_id', 'label_id', 'cls', 'name']:
                                if hasattr(box, attr):
                                    print(f"  box.{attr} = {getattr(box, attr)}")
                        elif hasattr(first, 'boxes'):
                            print(f"\n  first.boxes is empty or None: {first.boxes}")
                        
                        # Also try __dict__
                        if hasattr(first, '__dict__'):
                            print(f"\n  first.__dict__: {first.__dict__}")
                print("=" * 60 + "\n")
                debug_printed = True
            
            # Output format: list of result dicts, each with 'boxes' key
            # Each box is a dict with 'label' and 'coordinate' keys
            labels = []
            region_sizes = []  # List of (width, height, gpu_bytes)
            
            if layout_output and isinstance(layout_output[0], dict) and 'boxes' in layout_output[0]:
                # Standard format: output[0]['boxes'] is list of box dicts
                boxes = layout_output[0]['boxes']
                num_regions = len(boxes)
                for box in boxes:
                    # Extract label
                    label = box.get('label') or box.get('cls_name') or box.get('category') or box.get('cls') or 'unknown'
                    labels.append(str(label))
                    
                    # Extract region size from coordinates [x1, y1, x2, y2]
                    coord = box.get('coordinate', [0, 0, 0, 0])
                    x1, y1, x2, y2 = [float(c) for c in coord]
                    width = int(abs(x2 - x1))
                    height = int(abs(y2 - y1))
                    # GPU memory estimate: RGB float32 tensor = width * height * 3 * 4 bytes
                    gpu_bytes = width * height * 3 * 4
                    region_sizes.append((width, height, gpu_bytes))
                    
            elif layout_output and hasattr(layout_output[0], 'boxes'):
                # Object format: output[0].boxes is list of box objects
                boxes = layout_output[0].boxes
                num_regions = len(boxes)
                for box in boxes:
                    # Extract label
                    label = None
                    for attr in ['label', 'cls_name', 'category', 'cls', 'name']:
                        if hasattr(box, attr):
                            label = getattr(box, attr)
                            if label is not None:
                                break
                    labels.append(str(label) if label is not None else 'unknown')
                    
                    # Extract region size from coordinates
                    coord = getattr(box, 'coordinate', None) or getattr(box, 'bbox', [0, 0, 0, 0])
                    x1, y1, x2, y2 = [float(c) for c in coord]
                    width = int(abs(x2 - x1))
                    height = int(abs(y2 - y1))
                    gpu_bytes = width * height * 3 * 4
                    region_sizes.append((width, height, gpu_bytes))
                    
            elif layout_output and isinstance(layout_output[0], dict):
                # Direct dict format: list of dicts with 'label' key
                num_regions = len(layout_output)
                for box in layout_output:
                    labels.append(box.get("label", box.get("cls_name", "unknown")))
                    coord = box.get('coordinate', box.get('bbox', [0, 0, 0, 0]))
                    x1, y1, x2, y2 = [float(c) for c in coord]
                    width = int(abs(x2 - x1))
                    height = int(abs(y2 - y1))
                    gpu_bytes = width * height * 3 * 4
                    region_sizes.append((width, height, gpu_bytes))
            else:
                # Fallback: try to get length and labels
                num_regions = len(layout_output) if layout_output else 0
                labels = ["unknown"] * num_regions
                region_sizes = [(0, 0, 0)] * num_regions
            
            results.append({
                'name': name,
                'num_regions': num_regions,
                'labels': labels,
                'region_sizes': region_sizes,
            })
            
            if (i + 1) % 10 == 0 or (i + 1) == len(temp_files):
                print(f"  [{i+1}/{len(temp_files)}] Processed {name}: {num_regions} regions")
        
        batch_time = time.perf_counter() - batch_start
        print(f"\n[OK] Processed {len(images_data)} images in {batch_time:.2f}s")
        print(f"     Throughput: {len(images_data) / batch_time:.2f} images/sec")
        
        return results
        
    finally:
        # Clean up temp files
        for _, temp_path in temp_files:
            try:
                os.unlink(temp_path)
            except Exception:
                pass


@app.local_entrypoint()
def main(
    input_dir: str = "images/",
    limit: int = 0,
):
    """
    Count layout regions per image.
    
    Args:
        input_dir: Directory containing input images
        limit: Max number of images to process (0 = no limit)
    """
    extensions = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")
    
    # Load images
    images = []
    for path in sorted(Path(input_dir).iterdir()):
        if path.suffix.lower() in extensions:
            with open(path, "rb") as f:
                images.append((path.stem, f.read()))
    
    if not images:
        print(f"No images found in {input_dir}")
        return
    
    # Apply limit if specified
    total_found = len(images)
    if limit > 0:
        images = images[:limit]
    
    print(f"Found {total_found} images in {input_dir}")
    if limit > 0:
        print(f"Processing first {len(images)} images (--limit {limit})")
    print("Sending to Modal for processing...\n")
    
    start = time.perf_counter()
    
    # Count regions
    results = count_regions.remote(images)
    
    total_time = time.perf_counter() - start
    
    # Analyze results
    print("\n" + "=" * 60)
    print("REGION COUNT RESULTS")
    print("=" * 60)
    
    # Print per-image results
    print(f"\n{'Image':<45} | {'Regions':<7} | {'GPU Mem':<10} | {'Types'}")
    print("-" * 45 + "-+-" + "-" * 7 + "-+-" + "-" * 10 + "-+-" + "-" * 30)
    
    all_labels = []
    total_regions = 0
    all_region_sizes = []  # Collect all (width, height, bytes) tuples
    
    for result in results:
        name = result['name']
        num_regions = result['num_regions']
        labels = result['labels']
        region_sizes = result['region_sizes']
        
        # Calculate total GPU memory for this image's regions
        total_gpu_bytes = sum(size[2] for size in region_sizes)
        
        # Truncate long names
        display_name = name[:42] + "..." if len(name) > 45 else name
        labels_str = ", ".join(labels[:3])
        if len(labels) > 3:
            labels_str += f" (+{len(labels) - 3})"
        
        # Format memory size
        if total_gpu_bytes >= 1024 * 1024:
            mem_str = f"{total_gpu_bytes / (1024*1024):.1f} MB"
        elif total_gpu_bytes >= 1024:
            mem_str = f"{total_gpu_bytes / 1024:.1f} KB"
        else:
            mem_str = f"{total_gpu_bytes} B"
        
        print(f"{display_name:<45} | {num_regions:>7} | {mem_str:>10} | {labels_str}")
        
        all_labels.extend(labels)
        total_regions += num_regions
        all_region_sizes.extend(region_sizes)
    
    # Statistics
    avg_regions = total_regions / len(results) if results else 0
    label_counts = Counter(all_labels)
    
    print("-" * 45 + "-+-" + "-" * 7 + "-+-" + "-" * 10 + "-+-" + "-" * 30)
    print(f"\nRegion Statistics:")
    print(f"  Total images processed: {len(results)}")
    print(f"  Total regions detected: {total_regions}")
    print(f"  Average regions per image: {avg_regions:.2f}")
    print(f"  Min regions: {min(r['num_regions'] for r in results) if results else 0}")
    print(f"  Max regions: {max(r['num_regions'] for r in results) if results else 0}")
    
    print(f"\nRegion type distribution:")
    for label, count in label_counts.most_common():
        percentage = (count / total_regions * 100) if total_regions > 0 else 0
        print(f"  {label:<20} {count:>6} ({percentage:>5.1f}%)")
    
    # Memory statistics
    if all_region_sizes:
        all_bytes = [s[2] for s in all_region_sizes]
        all_widths = [s[0] for s in all_region_sizes]
        all_heights = [s[1] for s in all_region_sizes]
        total_bytes = sum(all_bytes)
        avg_bytes = total_bytes / len(all_bytes)
        min_bytes = min(all_bytes)
        max_bytes = max(all_bytes)
        
        def fmt_bytes(b):
            if b >= 1024 * 1024:
                return f"{b / (1024*1024):.2f} MB"
            elif b >= 1024:
                return f"{b / 1024:.2f} KB"
            else:
                return f"{b} B"
        
        print(f"\nGPU Memory Statistics (RGB float32 tensors):")
        print(f"  Total memory for all regions: {fmt_bytes(total_bytes)}")
        print(f"  Average per region: {fmt_bytes(avg_bytes)}")
        print(f"  Min region memory: {fmt_bytes(min_bytes)}")
        print(f"  Max region memory: {fmt_bytes(max_bytes)}")
        print(f"\nRegion Dimension Statistics:")
        print(f"  Average size: {sum(all_widths)/len(all_widths):.0f} x {sum(all_heights)/len(all_heights):.0f} px")
        print(f"  Min size: {min(all_widths)} x {min(all_heights)} px")
        print(f"  Max size: {max(all_widths)} x {max(all_heights)} px")
        
        # Breakdown by label type
        print(f"\nMemory by Region Type:")
        label_memory = {}
        for i, (w, h, b) in enumerate(all_region_sizes):
            lbl = all_labels[i] if i < len(all_labels) else 'unknown'
            if lbl not in label_memory:
                label_memory[lbl] = {'total': 0, 'count': 0, 'sizes': []}
            label_memory[lbl]['total'] += b
            label_memory[lbl]['count'] += 1
            label_memory[lbl]['sizes'].append((w, h, b))
        
        for label, data in sorted(label_memory.items(), key=lambda x: -x[1]['total']):
            avg_mem = data['total'] / data['count']
            avg_w = sum(s[0] for s in data['sizes']) / len(data['sizes'])
            avg_h = sum(s[1] for s in data['sizes']) / len(data['sizes'])
            print(f"  {label:<20}: {data['count']:>4} regions, avg {fmt_bytes(avg_mem):>10}, avg size {avg_w:.0f}x{avg_h:.0f}px")
    
    print(f"\n{'='*60}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {len(results) / total_time:.2f} images/sec")
    print("=" * 60)

