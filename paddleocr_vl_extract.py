#!/usr/bin/env python3
"""
PaddleOCR-VL extraction using official Docker image with vLLM backend.

Based on official PaddleOCR documentation:
https://www.paddleocr.ai/latest/version3.x/pipeline_usage/PaddleOCR-VL.html

Usage:
    modal run paddleocr_vl_extract.py --input-dir images/
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
    """Start vLLM server with PaddleOCR-VL model (official recommended settings)."""
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "PaddlePaddle/PaddleOCR-VL",
        "--served-model-name", "PaddleOCR-VL-0.9B",
        "--trust-remote-code",
        "--port", str(port),
        # Performance optimizations from official docs
        "--max-num-batched-tokens", "16384",
        "--max-model-len", "16384",
        "--gpu-memory-utilization", "0.9",
        # OCR-specific optimizations (disable caching for OCR workloads)
        "--disable-frontend-multiprocessing",
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
def extract_with_vllm(images_data: list[tuple[str, bytes]]) -> list[tuple[str, str]]:
    """
    Process images using PaddleOCR-VL with vLLM backend.
    
    Returns list of (name, markdown_content) tuples.
    """
    from paddleocr import PaddleOCRVL
    import os
    
    print("=" * 60)
    print("Starting PaddleOCR-VL with vLLM backend")
    print("=" * 60)
    
    # Start vLLM server in background
    vllm_port = 8118
    vllm_url = f"http://localhost:{vllm_port}"
    
    print("\n[1/4] Starting vLLM server...")
    server_proc = start_vllm_server(vllm_port)
    
    try:
        print("[2/4] Waiting for vLLM server to be ready...")
        if not wait_for_server(f"{vllm_url}/v1", timeout=300):
            # Print server output for debugging
            server_proc.terminate()
            stdout, _ = server_proc.communicate(timeout=5)
            print(f"Server output:\n{stdout}")
            raise RuntimeError("vLLM server failed to start")
        
        print(f"[OK] vLLM server ready at {vllm_url}")
        
        # Initialize PaddleOCR-VL with vLLM backend
        print("\n[3/4] Initializing PaddleOCR-VL pipeline...")
        pipeline = PaddleOCRVL(
            vl_rec_backend="vllm-server",
            vl_rec_server_url=f"{vllm_url}/v1",
            # Use PP-DocLayoutV2 for layout detection
            layout_detection_model_name="PP-DocLayoutV2",
            device="gpu:0",
        )
        print("[OK] Pipeline initialized")
        
        # Process images
        print(f"\n[4/4] Processing {len(images_data)} images...")
        results = []
        
        for i, (name, img_bytes) in enumerate(images_data, 1):
            img_start = time.perf_counter()
            
            # Write image to temp file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                f.write(img_bytes)
                temp_path = f.name
            
            try:
                # Run prediction
                output = pipeline.predict(temp_path)
                
                # Extract Markdown
                md_content = ""
                
                with tempfile.TemporaryDirectory() as tmpdir:
                    for res in output:
                        res.save_to_markdown(save_path=tmpdir)
                    
                    for md_file in Path(tmpdir).glob("*.md"):
                        md_content += md_file.read_text()
                
                results.append((name, md_content))
                
                img_time = time.perf_counter() - img_start
                print(f"  [{i}/{len(images_data)}] {name}: {img_time:.2f}s")
                
            finally:
                os.unlink(temp_path)
        
        return results
        
    finally:
        # Cleanup vLLM server
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
    
    This is faster to start but may have lower throughput for large batches.
    """
    from paddleocr import PaddleOCRVL
    import os
    
    print("=" * 60)
    print("PaddleOCR-VL Direct Mode (no vLLM server)")
    print("=" * 60)
    
    print("\n[1/2] Loading PaddleOCR-VL model...")
    start_load = time.perf_counter()
    
    pipeline = PaddleOCRVL(
        device="gpu:0",
        layout_detection_model_name="PP-DocLayoutV2",
    )
    
    load_time = time.perf_counter() - start_load
    print(f"[OK] Model loaded in {load_time:.1f}s")
    
    # Process images
    print(f"\n[2/2] Processing {len(images_data)} images...")
    results = []
    start_inference = time.perf_counter()
    
    for i, (name, img_bytes) in enumerate(images_data, 1):
        img_start = time.perf_counter()
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(img_bytes)
            temp_path = f.name
        
        try:
            output = pipeline.predict(temp_path)
            
            md_content = ""
            
            with tempfile.TemporaryDirectory() as tmpdir:
                for res in output:
                    res.save_to_markdown(save_path=tmpdir)
                
                for md_file in Path(tmpdir).glob("*.md"):
                    md_content += md_file.read_text()
            
            results.append((name, md_content))
            
            img_time = time.perf_counter() - img_start
            print(f"  [{i}/{len(images_data)}] {name}: {img_time:.2f}s")
            
        finally:
            os.unlink(temp_path)
    
    inference_time = time.perf_counter() - start_inference
    print("\nInference complete:")
    print(f"  Total time: {inference_time:.2f}s")
    print(f"  Throughput: {len(images_data) / inference_time:.2f} pages/sec")
    
    return results


@app.local_entrypoint()
def main(
    input_dir: str = "images/",
    output_dir: str = "output",
    mode: str = "direct",
):
    """
    Extract tables from images using PaddleOCR-VL.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save Markdown output
        mode: "vllm" for vLLM server backend, "direct" for direct inference
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
    
    print(f"Found {len(images)} images in {input_dir}")
    print(f"Mode: {mode}")
    print("Sending to Modal for processing...\n")
    
    start = time.perf_counter()
    
    # Choose extraction method
    if mode == "vllm":
        results = extract_with_vllm.remote(images)
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

