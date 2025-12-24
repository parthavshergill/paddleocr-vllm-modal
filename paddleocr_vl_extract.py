#!/usr/bin/env python3
"""
PaddleOCR-VL extraction using official Docker image with vLLM backend.

Based on official PaddleOCR documentation:
https://www.paddleocr.ai/latest/version3.x/pipeline_usage/PaddleOCR-VL.html

Two modes available:
- direct: Uses PaddleOCR-VL directly (faster startup, good for small batches)
- vllm: Uses vLLM server backend (optimized inference, better for larger batches)

Parallelism is achieved via:
- use_queues=True: Enables async threaded pipeline (layout detection + VLM in parallel)
- vl_rec_max_concurrency: Concurrent requests to vLLM server
- Batch input: Pass all images as a list to predict() in a single call

Usage:
    modal run paddleocr_vl_extract.py --input-dir images/
    modal run paddleocr_vl_extract.py --input-dir images/ --mode vllm
"""

import modal
import time
import subprocess
import tempfile
from pathlib import Path

app = modal.App("paddleocr-vl-official")

# Build image following official PaddleOCR installation guide
# Using PaddlePaddle GPU + PaddleOCR with doc-parser for layout detection
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


def start_vllm_server(port: int = 8118) -> subprocess.Popen:
    """Start vLLM server with PaddleOCR-VL model optimized for OCR workloads."""
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "PaddlePaddle/PaddleOCR-VL",
        "--served-model-name", "PaddleOCR-VL-0.9B",
        "--trust-remote-code",
        "--port", str(port),
        # === OCR-specific optimizations ===
        # Enable prefix caching - system prompts may share prefixes
        # (image tokens are unique, but text prompts have overlap)
        # "--no-enable-prefix-caching",  # Try with caching enabled
        # Chunked prefill - leave enabled (default) for better memory efficiency
        # "--no-enable-chunked-prefill",
        # Disable multimodal processor cache - each image is truly unique
        "--mm-processor-cache-gb", "0",
        # === Parallelism settings ===
        # Increase concurrent requests for batch processing
        "--max-num-seqs", "32",
        # Higher batch token limit for better throughput
        "--max-num-batched-tokens", "32768",
        # Reduced context length - OCR outputs are typically shorter than chat
        "--max-model-len", "8192",
        # Leave room for layout model (~212MB) + batch processing buffers
        "--gpu-memory-utilization", "0.85",
    ]
    
    print(f"Starting vLLM server: {' '.join(cmd)}")
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
    
    # vLLM health endpoint is at /health, not /v1/health
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


@app.function(
    gpu="L4",
    image=paddleocr_image,
    volumes={"/root/.cache/huggingface": model_cache},
    timeout=3600,
)
def extract_with_vllm(
    images_data: list[tuple[str, bytes]],
    max_concurrency: int = 16,
) -> list[tuple[str, str]]:
    """
    Process images using PaddleOCR-VL with vLLM backend.
    
    Uses built-in parallelism:
    - use_queues=True: Async threaded pipeline (layout + VLM run in parallel)
    - vl_rec_max_concurrency: Concurrent requests to vLLM server
    - Batch input: All images processed in single predict() call
    """
    from paddleocr import PaddleOCRVL
    import os
    
    overall_start = time.perf_counter()
    
    print("=" * 60)
    print("Starting PaddleOCR-VL with vLLM backend (BATCH MODE)")
    print("=" * 60)
    
    # Start vLLM server in background
    vllm_port = 8118
    vllm_url = f"http://localhost:{vllm_port}"
    
    print("\n[1/4] Starting vLLM server...")
    vllm_start = time.perf_counter()
    server_proc = start_vllm_server(vllm_port)
    
    try:
        print("[2/4] Waiting for vLLM server to be ready...")
        if not wait_for_server(f"{vllm_url}/v1", timeout=300):
            server_proc.terminate()
            stdout, _ = server_proc.communicate(timeout=5)
            print(f"Server output:\n{stdout}")
            raise RuntimeError("vLLM server failed to start")
        
        vllm_startup_time = time.perf_counter() - vllm_start
        print(f"[OK] vLLM server ready at {vllm_url} (startup: {vllm_startup_time:.1f}s)")
        
        # Initialize PaddleOCR-VL with vLLM backend and concurrency settings
        print(f"\n[3/4] Initializing PaddleOCR-VL pipeline (max_concurrency={max_concurrency})...")
        pipeline_init_start = time.perf_counter()
        pipeline = PaddleOCRVL(
            vl_rec_backend="vllm-server",
            vl_rec_server_url=f"{vllm_url}/v1",
            vl_rec_max_concurrency=max_concurrency,  # Concurrent VLM requests
            layout_detection_model_name="PP-DocLayoutV2",
            device="gpu:0",
        )
        pipeline_init_time = time.perf_counter() - pipeline_init_start
        print(f"[OK] Pipeline initialized (init: {pipeline_init_time:.1f}s)")
        
        # Write all images to temp files
        print(f"\n[4/4] Processing {len(images_data)} images in BATCH with use_queues=True...")
        temp_files = []
        name_to_path = {}
        
        for name, img_bytes in images_data:
            temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            temp_file.write(img_bytes)
            temp_file.close()
            temp_files.append(temp_file.name)
            name_to_path[temp_file.name] = name
        
        try:
            # Process ALL images in a single batched call with async queues
            batch_start = time.perf_counter()
            
            # Pass list of all image paths - pipeline handles parallelism internally
            outputs = pipeline.predict(
                temp_files,
                use_queues=True,  # Enable async threaded pipeline!
            )
            
            batch_time = time.perf_counter() - batch_start
            print(f"[OK] Batch inference complete in {batch_time:.2f}s")
            print(f"     Throughput: {len(images_data) / batch_time:.2f} images/sec")
            
            # Collect results
            results = []
            for i, (temp_path, res) in enumerate(zip(temp_files, outputs)):
                name = name_to_path[temp_path]
                
                md_content = ""
                with tempfile.TemporaryDirectory() as tmpdir:
                    res.save_to_markdown(save_path=tmpdir)
                    for md_file in Path(tmpdir).glob("*.md"):
                        md_content += md_file.read_text()
                
                results.append((name, md_content))
                print(f"  [{i+1}/{len(images_data)}] {name}: OK")
            
            # Print timing summary
            overall_time = time.perf_counter() - overall_start
            print("\n" + "=" * 60)
            print("TIMING SUMMARY")
            print("=" * 60)
            print(f"  vLLM server startup:  {vllm_startup_time:.1f}s")
            print(f"  Pipeline init:        {pipeline_init_time:.1f}s")
            print(f"  Batch inference:      {batch_time:.1f}s")
            print("  ─────────────────────────────")
            print(f"  Total:                {overall_time:.1f}s")
            print(f"  Inference throughput: {len(images_data) / batch_time:.2f} images/sec")
            print("=" * 60)
            
            return results
            
        finally:
            # Clean up temp files
            for temp_path in temp_files:
                os.unlink(temp_path)
        
    finally:
        print("\nShutting down vLLM server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_proc.kill()


@app.function(
    gpu="L4",
    image=paddleocr_image,
    volumes={"/root/.cache/huggingface": model_cache},
    timeout=3600,
)
def extract_direct(images_data: list[tuple[str, bytes]]) -> list[tuple[str, str]]:
    """
    Simpler alternative: Use PaddleOCR-VL directly without vLLM server.
    
    Uses native PaddlePaddle inference with use_queues=True for parallelism.
    This is faster to start but may have lower throughput for large batches.
    """
    from paddleocr import PaddleOCRVL
    import os
    
    print("=" * 60)
    print("PaddleOCR-VL Direct Mode (BATCH with use_queues=True)")
    print("=" * 60)
    
    print("\n[1/2] Loading PaddleOCR-VL model...")
    start_load = time.perf_counter()
    
    pipeline = PaddleOCRVL(
        device="gpu:0",
        layout_detection_model_name="PP-DocLayoutV2",
    )
    
    load_time = time.perf_counter() - start_load
    print(f"[OK] Model loaded in {load_time:.1f}s")
    
    # Write all images to temp files
    print(f"\n[2/2] Processing {len(images_data)} images in BATCH...")
    temp_files = []
    name_to_path = {}
    
    for name, img_bytes in images_data:
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temp_file.write(img_bytes)
        temp_file.close()
        temp_files.append(temp_file.name)
        name_to_path[temp_file.name] = name
    
    try:
        # Process ALL images in a single batched call with async queues
        batch_start = time.perf_counter()
        
        outputs = pipeline.predict(
            temp_files,
            use_queues=True,  # Enable async threaded pipeline!
        )
        
        batch_time = time.perf_counter() - batch_start
        print(f"[OK] Batch inference complete in {batch_time:.2f}s")
        print(f"     Throughput: {len(images_data) / batch_time:.2f} images/sec")
        
        # Collect results
        results = []
        for i, (temp_path, res) in enumerate(zip(temp_files, outputs)):
            name = name_to_path[temp_path]
            
            md_content = ""
            with tempfile.TemporaryDirectory() as tmpdir:
                res.save_to_markdown(save_path=tmpdir)
                for md_file in Path(tmpdir).glob("*.md"):
                    md_content += md_file.read_text()
            
            results.append((name, md_content))
            print(f"  [{i+1}/{len(images_data)}] {name}: OK")
        
        return results
        
    finally:
        # Clean up temp files
        for temp_path in temp_files:
            os.unlink(temp_path)


@app.local_entrypoint()
def main(
    input_dir: str = "images/",
    output_dir: str = "output",
    mode: str = "direct",
    max_concurrency: int = 16,
    limit: int = 0,
):
    """
    Extract tables from images using PaddleOCR-VL.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save Markdown output
        mode: "vllm" for vLLM server backend, "direct" for native PaddlePaddle
        max_concurrency: Max concurrent VLM requests (vllm mode only)
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
    print(f"Mode: {mode}")
    if mode == "vllm":
        print(f"Max concurrency: {max_concurrency}")
    print("Sending to Modal for processing...\n")
    
    start = time.perf_counter()
    
    # Choose extraction method
    if mode == "vllm":
        results = extract_with_vllm.remote(images, max_concurrency=max_concurrency)
    else:
        results = extract_direct.remote(images)
    
    total_time = time.perf_counter() - start
    
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
    print(f"Throughput: {len(results) / total_time:.2f} pages/sec (including overhead)")
    print(f"Output directory: {output_path.absolute()}")
