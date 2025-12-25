#!/usr/bin/env python3
"""
PaddleOCR-VL Service - Warm vLLM server using Modal's @app.cls decorator.

This version keeps the vLLM server running between invocations, eliminating
the ~60-90s startup time for subsequent calls. Uses Modal's lifecycle hooks:
- @modal.enter: Start vLLM server + initialize PaddleOCR pipeline (once per container)
- @modal.exit: Clean shutdown of vLLM server
- @modal.method: Process images using the warm server

Usage:
    # Deploy once (container stays warm based on scaledown_window)
    modal deploy paddleocr_vl_service.py

    # Run with warm server
    modal run paddleocr_vl_service.py --input-dir images/

    # Run with region type filtering (extract only tables)
    modal run paddleocr_vl_service.py --input-dir images/ --region-type table

    # Or call programmatically after deploy
    from paddleocr_vl_service import PaddleOCRVLService
    service = PaddleOCRVLService()
    results = service.process_images.remote(images_data)
"""

import modal
import time
import subprocess
import tempfile
from pathlib import Path
from collections import Counter

app = modal.App("paddleocr-vl-warm")

# Build image following official PaddleOCR installation guide
paddleocr_image = (
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
    # Install PaddleOCR with doc-parser and PaddleX for VL support
    .pip_install(
        "paddleocr[doc-parser]>=3.0.0",
        "paddlex>=3.0.0",
        "pillow",
        "httpx",
    )
    # Install vLLM for inference acceleration
    .pip_install(
        "vllm>=0.9.0",
        extra_index_url="https://download.pytorch.org/whl/cu126",
    )
)

# Cache for model weights
model_cache = modal.Volume.from_name("paddleocr-vl-cache", create_if_missing=True)

# vLLM server configuration
VLLM_PORT = 8118
VLLM_URL = f"http://localhost:{VLLM_PORT}"


# =============================================================================
# Helper Functions
# =============================================================================

def fmt_bytes(b: int) -> str:
    """Format bytes as human-readable string."""
    if b >= 1024 * 1024:
        return f"{b / (1024*1024):.2f} MB"
    elif b >= 1024:
        return f"{b / 1024:.2f} KB"
    else:
        return f"{b} B"


def extract_regions_from_layout(layout_output) -> list[dict]:
    """
    Extract region info from layout detection output.
    Returns list of dicts with 'label', 'coordinate', 'width', 'height', 'gpu_bytes'.
    """
    regions = []
    
    if layout_output and isinstance(layout_output[0], dict) and 'boxes' in layout_output[0]:
        # Standard format: output[0]['boxes'] is list of box dicts
        boxes = layout_output[0]['boxes']
        for box in boxes:
            label = box.get('label') or box.get('cls_name') or box.get('category') or box.get('cls') or 'unknown'
            coord = box.get('coordinate', [0, 0, 0, 0])
            x1, y1, x2, y2 = [float(c) for c in coord]
            width = int(abs(x2 - x1))
            height = int(abs(y2 - y1))
            gpu_bytes = width * height * 3 * 4  # RGB float32 tensor
            regions.append({
                'label': str(label),
                'coordinate': [x1, y1, x2, y2],
                'width': width,
                'height': height,
                'gpu_bytes': gpu_bytes,
            })
    elif layout_output and hasattr(layout_output[0], 'boxes'):
        # Object format: output[0].boxes is list of box objects
        boxes = layout_output[0].boxes
        for box in boxes:
            label = None
            for attr in ['label', 'cls_name', 'category', 'cls', 'name']:
                if hasattr(box, attr):
                    label = getattr(box, attr)
                    if label is not None:
                        break
            coord = getattr(box, 'coordinate', None) or getattr(box, 'bbox', [0, 0, 0, 0])
            x1, y1, x2, y2 = [float(c) for c in coord]
            width = int(abs(x2 - x1))
            height = int(abs(y2 - y1))
            gpu_bytes = width * height * 3 * 4
            regions.append({
                'label': str(label) if label is not None else 'unknown',
                'coordinate': [x1, y1, x2, y2],
                'width': width,
                'height': height,
                'gpu_bytes': gpu_bytes,
            })
    
    return regions


def log_region_stats(regions: list[dict], filter_types: list[str] | None = None):
    """Log detailed statistics about regions."""
    if not regions:
        print("  No regions detected")
        return
    
    # Count by type
    label_counts = Counter(r['label'] for r in regions)
    
    print(f"\n  Region type distribution (total: {len(regions)}):")
    for label, count in label_counts.most_common():
        # Calculate avg memory for this type
        type_regions = [r for r in regions if r['label'] == label]
        avg_bytes = sum(r['gpu_bytes'] for r in type_regions) / len(type_regions)
        avg_w = sum(r['width'] for r in type_regions) / len(type_regions)
        avg_h = sum(r['height'] for r in type_regions) / len(type_regions)
        
        match_marker = " <-- MATCH" if filter_types and label in filter_types else ""
        print(f"    {label:<20}: {count:>4} regions, avg {fmt_bytes(int(avg_bytes)):>10}, "
              f"avg size {avg_w:.0f}x{avg_h:.0f}px{match_marker}")


# =============================================================================
# vLLM Server Management
# =============================================================================

def start_vllm_server(port: int = VLLM_PORT) -> subprocess.Popen:
    """Start vLLM server with PaddleOCR-VL model optimized for OCR workloads."""
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "PaddlePaddle/PaddleOCR-VL",
        "--served-model-name", "PaddleOCR-VL-0.9B",
        "--trust-remote-code",
        "--port", str(port),
        # === OCR-specific optimizations ===
        "--mm-processor-cache-gb", "0",
        # === Parallelism settings ===
        "--max-num-seqs", "192",
        "--max-num-batched-tokens", "32768",
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.85",
        # Note: --kv-cache-dtype fp8 requires nvcc for FlashInfer JIT compilation
        # which isn't available in the Modal image
    ]
    
    print(f"[STARTUP] Starting vLLM server: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


def wait_for_server(base_url: str, timeout: int = 300) -> bool:
    """Wait for vLLM server to be ready."""
    import httpx
    
    health_url = base_url.replace("/v1", "") + "/health"
    
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = httpx.get(health_url, timeout=5)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


# =============================================================================
# PaddleOCR-VL Service
# =============================================================================

@app.cls(
    gpu="A100",  # 40GB - more VRAM for higher concurrency
    image=paddleocr_image,
    volumes={"/root/.cache/huggingface": model_cache},
    timeout=3600,
    # === Warmth control ===
    scaledown_window=900,  # Keep container alive 15 min after last request
    # Uncomment to always keep one container warm (costs money when idle):
    # min_containers=1,
)
class PaddleOCRVLService:
    """
    Warm PaddleOCR-VL service with persistent vLLM server.
    
    The vLLM server and PaddleOCR pipeline are initialized once when the
    container starts, then reused for all subsequent requests until the
    container scales down (after scaledown_window seconds of inactivity).
    """
    
    @modal.enter()
    def startup(self):
        """
        Called once when container starts.
        Starts vLLM server and initializes PaddleOCR-VL pipeline + layout model.
        """
        from paddleocr import PaddleOCRVL, LayoutDetection
        
        overall_start = time.perf_counter()
        
        print("=" * 60)
        print("CONTAINER STARTUP - Initializing warm vLLM server")
        print("=" * 60)
        
        # Start vLLM server
        print("\n[1/4] Starting vLLM server...")
        vllm_start = time.perf_counter()
        self.server_proc = start_vllm_server(VLLM_PORT)
        
        # Wait for server to be ready
        print("[2/4] Waiting for vLLM server to be ready...")
        if not wait_for_server(f"{VLLM_URL}/v1", timeout=300):
            self.server_proc.terminate()
            stdout, _ = self.server_proc.communicate(timeout=5)
            print(f"Server output:\n{stdout}")
            raise RuntimeError("vLLM server failed to start")
        
        vllm_startup_time = time.perf_counter() - vllm_start
        print(f"[OK] vLLM server ready at {VLLM_URL} (startup: {vllm_startup_time:.1f}s)")
        
        # Initialize PaddleOCR-VL pipeline
        # Note: vl_rec_max_concurrency must be set at init time, can't be changed per-request
        print("\n[3/4] Initializing PaddleOCR-VL pipeline (max_concurrency=96)...")
        pipeline_start = time.perf_counter()
        self.pipeline = PaddleOCRVL(
            vl_rec_backend="vllm-server",
            vl_rec_server_url=f"{VLLM_URL}/v1",
            vl_rec_max_concurrency=192,  # Match vLLM --max-num-seqs
            layout_detection_model_name="PP-DocLayoutV2",
            device="gpu:0",
        )
        pipeline_time = time.perf_counter() - pipeline_start
        print(f"[OK] Pipeline initialized (init: {pipeline_time:.1f}s)")
        
        # Initialize LayoutDetection model for region filtering
        print("\n[4/4] Initializing LayoutDetection model...")
        layout_start = time.perf_counter()
        self.layout_model = LayoutDetection(
            model_name="PP-DocLayoutV2",
            device="gpu:0",
        )
        layout_time = time.perf_counter() - layout_start
        print(f"[OK] Layout model initialized (init: {layout_time:.1f}s)")
        
        # Store timing for reporting
        self.startup_time = time.perf_counter() - overall_start
        self.vllm_startup_time = vllm_startup_time
        self.pipeline_init_time = pipeline_time
        self.layout_init_time = layout_time
        
        print("\n" + "=" * 60)
        print(f"CONTAINER READY - Total startup: {self.startup_time:.1f}s")
        print("=" * 60)
    
    @modal.exit()
    def shutdown(self):
        """Called when container scales down. Clean up vLLM server."""
        print("\n[SHUTDOWN] Terminating vLLM server...")
        if hasattr(self, 'server_proc'):
            self.server_proc.terminate()
            try:
                self.server_proc.wait(timeout=10)
                print("[SHUTDOWN] vLLM server terminated cleanly")
            except subprocess.TimeoutExpired:
                self.server_proc.kill()
                print("[SHUTDOWN] vLLM server killed (timeout)")
    
    @modal.method()
    def process_images(
        self,
        images_data: list[tuple[str, bytes]],
        region_type: str = "",
    ) -> dict:
        """
        Process images using the warm vLLM server.
        
        Args:
            images_data: List of (filename, image_bytes) tuples
            region_type: Only process regions of this type (e.g., "table" or "table,header").
                         If empty, processes full images.
            
        Returns:
            Dict with:
                - results: List of (filename, markdown_content) tuples
                - stats: Dict with region statistics (num_regions, avg_width, avg_height, avg_memory_bytes)
            
        Note:
            Concurrency is fixed at 192 (matching vLLM's --max-num-seqs).
        """
        from PIL import Image
        import numpy as np
        import io
        import os
        
        batch_start = time.perf_counter()
        filter_types = [t.strip() for t in region_type.split(",")] if region_type else None
        
        print("=" * 60)
        if filter_types:
            print(f"PROCESSING {len(images_data)} images (REGION FILTER: {filter_types})")
        else:
            print(f"PROCESSING {len(images_data)} images (warm server)")
        print("=" * 60)
        
        # Write images to temp files
        temp_files = []
        name_to_path = {}
        name_to_bytes = {}
        
        for name, img_bytes in images_data:
            temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            temp_file.write(img_bytes)
            temp_file.close()
            temp_files.append(temp_file.name)
            name_to_path[temp_file.name] = name
            name_to_bytes[name] = img_bytes
        
        try:
            # ============================================================
            # REGION TYPE FILTERING PATH (two-stage)
            # ============================================================
            if filter_types:
                print("\n[1/2] Running layout detection to find regions...")
                layout_start = time.perf_counter()
                
                # Collect all regions across all images
                all_regions_info = []  # List of (image_name, region_info, crop_bytes)
                all_regions = []  # All detected regions (for stats logging)
                
                for temp_path in temp_files:
                    name = name_to_path[temp_path]
                    layout_output = self.layout_model.predict(temp_path)
                    regions = extract_regions_from_layout(layout_output)
                    all_regions.extend(regions)
                    
                    # Filter regions by type
                    matching_regions = [r for r in regions if r['label'] in filter_types]
                    
                    # Crop image to each matching region
                    img = Image.open(io.BytesIO(name_to_bytes[name]))
                    for idx, region in enumerate(matching_regions):
                        x1, y1, x2, y2 = region['coordinate']
                        crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
                        
                        # Convert to numpy array (BGR for OpenCV/PaddleOCR compatibility)
                        crop_rgb = np.array(crop.convert("RGB"))
                        crop_bgr = crop_rgb[:, :, ::-1].copy()  # RGB -> BGR
                        
                        all_regions_info.append((f"{name}_region{idx}", region, crop_bgr))
                
                layout_time = time.perf_counter() - layout_start
                print(f"[OK] Layout detection complete in {layout_time:.1f}s")
                
                # Log region statistics
                total_regions_before = len(all_regions)
                total_regions_after = len(all_regions_info)
                print(f"\n  Filter applied: {filter_types}")
                print(f"  Total regions before filter: {total_regions_before}")
                print(f"  Total regions after filter:  {total_regions_after}")
                log_region_stats(all_regions, filter_types)
                
                # Log filtered region memory stats
                if all_regions_info:
                    filtered_stats = Counter()
                    filtered_memory = {}
                    for _, region, _ in all_regions_info:
                        label = region['label']
                        filtered_stats[label] += 1
                        if label not in filtered_memory:
                            filtered_memory[label] = []
                        filtered_memory[label].append(region['gpu_bytes'])
                    
                    print(f"\n  Filtered regions to process ({total_regions_after} total):")
                    for label, count in filtered_stats.most_common():
                        avg_mem = sum(filtered_memory[label]) / len(filtered_memory[label])
                        print(f"    {label:<20}: {count:>4} regions, avg memory {fmt_bytes(int(avg_mem))}")
                
                if not all_regions_info:
                    print("\n  [WARNING] No regions matched the filter. Nothing to process.")
                    return {"results": [], "stats": None}
                
                # Pass numpy arrays directly to VL model (no temp files!)
                print(f"\n[2/2] Processing {len(all_regions_info)} cropped regions with VL model...")
                crop_arrays = [info[2] for info in all_regions_info]
                
                # VL inference on cropped regions (with use_layout_detection=False)
                # prompt_label="table" tells VL model to expect table content
                # Note: use_queues=False is faster for decoupled approach since
                # there's no layout→VL pipeline to overlap (we do layout separately)
                vl_start = time.perf_counter()
                outputs = self.pipeline.predict(
                    crop_arrays,
                    use_layout_detection=False,  # Already cropped, skip layout
                    use_queues=False,  # Faster for decoupled approach
                    prompt_label="table",  # Hint to VL model for better table extraction
                )
                vl_time = time.perf_counter() - vl_start
                
                print(f"[OK] VL inference complete in {vl_time:.2f}s")
                print(f"     Throughput: {len(all_regions_info) / vl_time:.2f} regions/sec")
                
                # Collect results
                results = []
                for i, ((crop_name, region, _), res) in enumerate(zip(all_regions_info, outputs)):
                    md_content = ""
                    with tempfile.TemporaryDirectory() as tmpdir:
                        res.save_to_markdown(save_path=tmpdir)
                        for md_file in Path(tmpdir).glob("*.md"):
                            md_content += md_file.read_text()
                    
                    results.append((crop_name, md_content))
                    if (i + 1) % 10 == 0 or (i + 1) == len(all_regions_info):
                        print(f"  [{i+1}/{len(all_regions_info)}] {crop_name}: OK")
                
                # Timing summary
                total_time = time.perf_counter() - batch_start
                print("\n" + "=" * 60)
                print("TIMING SUMMARY (Region-Filtered Mode)")
                print("=" * 60)
                print(f"  Layout detection:     {layout_time:.1f}s")
                print(f"  VL inference:         {vl_time:.1f}s")
                print("  ─────────────────────────────")
                print(f"  Total request time:   {total_time:.1f}s")
                print(f"  VL throughput:        {len(all_regions_info) / vl_time:.2f} regions/sec")
                print(f"  Regions processed:    {len(all_regions_info)} (filter: {filter_types})")
                print("=" * 60)
                
                # Compute stats for filtered regions
                filtered_regions = [r for _, r, _ in all_regions_info]
                stats = {
                    "num_regions": len(filtered_regions),
                    "avg_width": sum(r['width'] for r in filtered_regions) / len(filtered_regions) if filtered_regions else 0,
                    "avg_height": sum(r['height'] for r in filtered_regions) / len(filtered_regions) if filtered_regions else 0,
                    "avg_memory_bytes": sum(r['gpu_bytes'] for r in filtered_regions) / len(filtered_regions) if filtered_regions else 0,
                }
                
                return {"results": results, "stats": stats}
            
            # ============================================================
            # DEFAULT PATH (no region filtering) - Uses decoupled approach
            # ============================================================
            else:
                # Decoupled approach: layout detection first, then VL inference
                # Same pattern as filtered path but processes all regions
                print("\n[1/2] Running layout detection (all regions)...")
                layout_start = time.perf_counter()
                
                # Collect all regions across all images
                # Store numpy arrays directly (no temp files needed!)
                all_regions_info = []  # List of (image_name, region_idx, region_info, crop_array)
                all_regions = []  # All detected regions (for stats logging)
                
                for temp_path in temp_files:
                    name = name_to_path[temp_path]
                    layout_output = self.layout_model.predict(temp_path)
                    regions = extract_regions_from_layout(layout_output)
                    all_regions.extend(regions)
                    
                    # Crop image to each region (no filtering)
                    img = Image.open(io.BytesIO(name_to_bytes[name]))
                    for idx, region in enumerate(regions):
                        x1, y1, x2, y2 = region['coordinate']
                        crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
                        
                        # Convert to numpy array (BGR for OpenCV/PaddleOCR compatibility)
                        crop_rgb = np.array(crop.convert("RGB"))
                        crop_bgr = crop_rgb[:, :, ::-1].copy()  # RGB -> BGR
                        
                        all_regions_info.append((name, idx, region, crop_bgr))
                
                layout_time = time.perf_counter() - layout_start
                print(f"[OK] Layout detection complete in {layout_time:.1f}s")
                print(f"\n  Total regions detected: {len(all_regions)}")
                log_region_stats(all_regions, filter_types=None)
                
                if not all_regions_info:
                    print("\n  [WARNING] No regions detected. Nothing to process.")
                    return {"results": [], "stats": None}
                
                # Pass numpy arrays directly to VL model (no temp files!)
                print(f"\n[2/2] Processing {len(all_regions_info)} cropped regions with VL model...")
                crop_arrays = [info[3] for info in all_regions_info]
                
                # VL inference on cropped regions (with use_layout_detection=False)
                # Note: use_queues=False is faster for decoupled approach since
                # there's no layout→VL pipeline to overlap (we do layout separately)
                vl_start = time.perf_counter()
                outputs = self.pipeline.predict(
                    crop_arrays,
                    use_layout_detection=False,  # Already cropped, skip layout
                    use_queues=False,  # Avoid PaddleX queue deadlock
                )
                vl_time = time.perf_counter() - vl_start
                
                print(f"[OK] VL inference complete in {vl_time:.2f}s")
                print(f"     Throughput: {len(all_regions_info) / vl_time:.2f} regions/sec")
                
                # Group results by original image
                image_results = {}  # name -> list of (region_idx, region_info, md_content)
                for (name, idx, region, _), res in zip(all_regions_info, outputs):
                    md_content = ""
                    with tempfile.TemporaryDirectory() as tmpdir:
                        res.save_to_markdown(save_path=tmpdir)
                        for md_file in Path(tmpdir).glob("*.md"):
                            md_content += md_file.read_text()
                    
                    if name not in image_results:
                        image_results[name] = []
                    image_results[name].append((idx, region, md_content))
                
                # Reconstruct per-image results (combine regions in order)
                results = []
                for img_name in sorted(image_results.keys()):
                    region_outputs = sorted(image_results[img_name], key=lambda x: x[0])
                    combined_md = ""
                    for idx, region, md_content in region_outputs:
                        # Add region header
                        combined_md += f"\n<!-- Region {idx}: {region['label']} -->\n"
                        combined_md += md_content
                    results.append((img_name, combined_md.strip()))
                
                # Log progress
                for i, (img_name, _) in enumerate(results):
                    print(f"  [{i+1}/{len(results)}] {img_name}: OK")
                
                # Timing summary
                total_time = time.perf_counter() - batch_start
                print("\n" + "=" * 60)
                print("TIMING SUMMARY (Decoupled Full-Image Mode)")
                print("=" * 60)
                print(f"  Layout detection:     {layout_time:.1f}s")
                print(f"  VL inference:         {vl_time:.1f}s")
                print("  ─────────────────────────────")
                print(f"  Total request time:   {total_time:.1f}s")
                print(f"  VL throughput:        {len(all_regions_info) / vl_time:.2f} regions/sec")
                print(f"  Regions processed:    {len(all_regions_info)} (no filter)")
                print(f"  Images:               {len(images_data)}")
                print("=" * 60)
                
                # Compute region stats
                region_stats = {
                    "num_regions": len(all_regions),
                    "avg_width": sum(r['width'] for r in all_regions) / len(all_regions) if all_regions else 0,
                    "avg_height": sum(r['height'] for r in all_regions) / len(all_regions) if all_regions else 0,
                    "avg_memory_bytes": sum(r['gpu_bytes'] for r in all_regions) / len(all_regions) if all_regions else 0,
                }
                
                return {"results": results, "stats": region_stats}
            
        finally:
            # Clean up temp files
            for temp_path in temp_files:
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass


@app.local_entrypoint()
def main(
    input_dir: str = "images/",
    output_dir: str = "output-warm",
    limit: int = 0,
    filter: str = "",
    region_type: str = "",
):
    """
    Extract tables from images using warm PaddleOCR-VL service.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save Markdown output
        limit: Max number of images to process (0 = no limit)
        filter: Only process images with this substring in filename
        region_type: Only process regions of this type (e.g., "table" or "table,header")
        
    Note:
        Concurrency is fixed at 192 (set at container startup to match vLLM's --max-num-seqs).
    """
    extensions = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")
    
    # Load images
    images = []
    for path in sorted(Path(input_dir).iterdir()):
        if path.suffix.lower() in extensions:
            if filter and filter not in path.name:
                continue
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
    if filter:
        print(f"Filter: '{filter}' -> {len(images)} images matched")
    if limit > 0:
        print(f"Processing first {len(images)} images (--limit {limit})")
    if region_type:
        print(f"Region type filter: '{region_type}'")
    print("Using WARM vLLM server mode (concurrency=96)")
    print("Sending to Modal...\n")
    
    start = time.perf_counter()
    
    # Call the warm service
    service = PaddleOCRVLService()
    response = service.process_images.remote(
        images,
        region_type=region_type,
    )
    
    total_time = time.perf_counter() - start
    
    results = response["results"]
    stats = response.get("stats")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for name, md_content in results:
        md_path = output_path / f"{name}.md"
        with open(md_path, "w") as f:
            f.write(md_content)
        
        print(f"Saved: {md_path}")
    
    print(f"\n{'='*60}")
    print(f"Total time (including upload): {total_time:.2f}s")
    print(f"Images processed: {len(results)}")
    if region_type:
        print(f"Region filter: {region_type}")
    if stats:
        print(f"Avg region size: {stats['avg_width']:.0f}x{stats['avg_height']:.0f}px")
        print(f"Avg region memory: {stats['avg_memory_bytes']/1024/1024:.2f}MB")
    print(f"Throughput: {len(results) / total_time:.2f} pages/sec (including overhead)")
    print(f"Output directory: {output_path.absolute()}")
    print("\nTIP: Run 'modal deploy paddleocr_vl_service.py' to keep the server warm")
    print("     between invocations (first call cold, subsequent calls fast)")

