"""
Minimal RunPod handler for debugging
No model loading - just test if workers can start
"""
import runpod
import os
import sys

print("=" * 60, flush=True)
print("MINIMAL HANDLER - NO MODEL", flush=True)
print("=" * 60, flush=True)
print(f"Python version: {sys.version}", flush=True)
print(f"Working dir: {os.getcwd()}", flush=True)

# Check /models directory
if os.path.exists("/models"):
    print("Contents of /models:", flush=True)
    for item in os.listdir("/models"):
        full_path = os.path.join("/models", item)
        if os.path.isfile(full_path):
            size_mb = os.path.getsize(full_path) / (1024**2)
            print(f"  - {item} ({size_mb:.1f} MB)", flush=True)
        else:
            print(f"  - {item}/ (directory)", flush=True)
else:
    print("/models directory not found!", flush=True)

# Try importing llama_cpp to see if CUDA setup works
print("\nTesting llama_cpp import...", flush=True)
try:
    from llama_cpp import Llama
    print("✓ llama_cpp imported successfully!", flush=True)
except Exception as e:
    print(f"✗ llama_cpp import failed: {e}", flush=True)


def handler(job):
    """Simple handler that doesn't load model"""
    print(f"Received job: {job}", flush=True)

    job_input = job.get("input", {})
    texts = job_input.get("input", ["no input"])

    # Return dummy embeddings (2560 dimensions like the real model)
    embeddings = []
    for i, text in enumerate(texts if isinstance(texts, list) else [texts]):
        embeddings.append({
            "embedding": [0.1] * 2560,  # Dummy embedding
            "index": i,
            "object": "embedding"
        })

    return {
        "message": "Minimal handler working! Model not loaded yet.",
        "received_texts": texts,
        "data": embeddings,
        "model": "test-no-model",
        "object": "list"
    }


print("Starting RunPod serverless (minimal handler)...", flush=True)
runpod.serverless.start({"handler": handler})
