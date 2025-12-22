"""
PaddleOCR-VL vLLM Server on Modal

Exposes an OpenAI-compatible API endpoint for high-throughput table extraction.
vLLM automatically batches concurrent requests for maximum GPU utilization.

Usage:
    # Deploy the server
    modal deploy table_extractor.py

    # The server URL will be printed - use it with the async client
"""

import modal
import subprocess
import time

app = modal.App("paddleocr-vllm-server")

# vLLM image
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm>=0.9.0")
)

# Volume for caching model weights
model_cache = modal.Volume.from_name("paddleocr-vl-cache", create_if_missing=True)


@app.function(
    gpu="L4",  # $0.80/hr, 24GB VRAM - best pages per dollar
    image=vllm_image,
    volumes={"/root/.cache/huggingface": model_cache},
    timeout=3600,  # 1 hour max
    allow_concurrent_inputs=1000,  # Accept many concurrent requests
    scaledown_window=300,  # Keep warm 5 min
)
@modal.web_server(port=8000, startup_timeout=600)
def serve():
    """Start vLLM server with optimized settings for throughput."""
    import os

    cmd = [
        "vllm", "serve", "PaddlePaddle/PaddleOCR-VL",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--trust-remote-code",
        "--served-model-name", "PaddleOCR-VL",
        # Throughput optimizations
        "--max-num-batched-tokens", "16384",
        "--max-num-seqs", "256",  # Conservative for L4's 24GB
        "--gpu-memory-utilization", "0.9",
        # PaddleOCR-VL specific (from official docs)
        "--no-enable-prefix-caching",
        "--mm-processor-cache-gb", "0",
    ]

    print(f"Starting vLLM server: {' '.join(cmd)}")
    subprocess.Popen(cmd, env={**os.environ})
