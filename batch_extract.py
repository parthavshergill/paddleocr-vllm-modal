#!/usr/bin/env python3
"""
Batch table extraction using vLLM offline inference.

This replicates the paper's benchmark methodology - direct batch processing
without HTTP overhead. Expected: ~0.8-1.2 pages/sec on L4/A10.

Usage:
    modal run batch_extract.py --input-dir images/
"""

import modal
import time
from pathlib import Path

app = modal.App("paddleocr-batch-extract")

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm>=0.9.0", "pillow")
)

model_cache = modal.Volume.from_name("paddleocr-vl-cache", create_if_missing=True)


@app.function(
    gpu="L4",
    image=vllm_image,
    volumes={"/root/.cache/huggingface": model_cache},
    timeout=3600,
)
def extract_tables_batch(images_bytes: list[bytes]) -> list[str]:
    """Process images in a single batch using vLLM offline inference."""
    from vllm import LLM, SamplingParams
    from PIL import Image
    import io

    print(f"Loading model...")
    start_load = time.perf_counter()

    llm = LLM(
        model="PaddlePaddle/PaddleOCR-VL",
        trust_remote_code=True,
        max_num_batched_tokens=16384,
        max_num_seqs=256,
        gpu_memory_utilization=0.95,
    )

    load_time = time.perf_counter() - start_load
    print(f"Model loaded in {load_time:.1f}s")

    # Build chat messages with base64 images
    conversations = []
    for img_bytes in images_bytes:
        import base64
        img_b64 = base64.b64encode(img_bytes).decode()
        conversations.append([{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": "Table Recognition:"},
            ],
        }])

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=4096,
    )

    print(f"Processing {len(conversations)} images in batch...")
    start_inference = time.perf_counter()

    outputs = llm.chat(conversations, sampling_params)

    inference_time = time.perf_counter() - start_inference
    throughput = len(conversations) / inference_time

    print(f"Inference complete:")
    print(f"  Total time: {inference_time:.2f}s")
    print(f"  Throughput: {throughput:.2f} pages/sec")
    print(f"  Avg latency: {inference_time / len(conversations):.2f}s per image")

    return [output.outputs[0].text for output in outputs]


@app.local_entrypoint()
def main(input_dir: str = "images/"):
    """Load images locally and send to Modal for batch processing."""
    extensions = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")

    images = []
    names = []
    for path in sorted(Path(input_dir).iterdir()):
        if path.suffix.lower() in extensions:
            with open(path, "rb") as f:
                images.append(f.read())
                names.append(path.stem)

    if not images:
        print(f"No images found in {input_dir}")
        return

    print(f"Sending {len(images)} images to Modal for batch processing...")

    start = time.perf_counter()
    results = extract_tables_batch.remote(images)
    total_time = time.perf_counter() - start

    # Save results
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    for name, content in zip(names, results):
        output_path = output_dir / f"{name}.md"
        with open(output_path, "w") as f:
            f.write(content)
        print(f"Saved: {output_path}")

    print(f"\n{'='*50}")
    print(f"Total (including upload): {total_time:.2f}s")
    print(f"Images processed: {len(results)}")
