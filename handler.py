"""
RunPod Serverless Handler for Qwen3-Embedding-4B-Q8_0
Exact same model as local - 100% embedding compatibility
Uses lazy loading to avoid startup crashes
"""
import runpod
import os
import sys

print("=" * 60, flush=True)
print("HANDLER MODULE LOADED", flush=True)
print("=" * 60, flush=True)

# Global model variable - will be loaded on first request
_llm = None
_model_loaded = False

def get_model():
    """Lazy load the model on first request"""
    global _llm, _model_loaded

    if _model_loaded:
        return _llm

    MODEL_PATH = os.getenv("MODEL_PATH", "/models/Qwen3-Embedding-4B-q8_0.gguf")
    N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "-1"))

    print(f"MODEL_PATH: {MODEL_PATH}", flush=True)
    print(f"N_GPU_LAYERS: {N_GPU_LAYERS}", flush=True)

    # Check if model file exists
    if os.path.exists(MODEL_PATH):
        size_gb = os.path.getsize(MODEL_PATH) / (1024**3)
        print(f"✓ Model found: {MODEL_PATH} ({size_gb:.2f} GB)", flush=True)
    else:
        print(f"✗ Model NOT found: {MODEL_PATH}", flush=True)
        # List /models directory
        if os.path.exists("/models"):
            print("Contents of /models:", flush=True)
            for item in os.listdir("/models"):
                full_path = os.path.join("/models", item)
                if os.path.isfile(full_path):
                    size = os.path.getsize(full_path) / (1024**3)
                    print(f"  - {item} ({size:.2f} GB)", flush=True)
                else:
                    print(f"  - {item}/ (dir)", flush=True)
        else:
            print("/models does not exist!", flush=True)
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    # Import and load llama
    print("Importing llama_cpp...", flush=True)
    from llama_cpp import Llama
    print("✓ llama_cpp imported", flush=True)

    print("Loading model into GPU...", flush=True)
    _llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=N_GPU_LAYERS,
        embedding=True,
        n_ctx=8192,
        verbose=True
    )
    print("✓ Model loaded!", flush=True)

    # Test embedding
    print("Testing embedding...", flush=True)
    test = _llm.embed("test")
    print(f"✓ Embedding works, dim={len(test)}", flush=True)

    _model_loaded = True
    return _llm


def handler(job):
    """RunPod handler - OpenAI-compatible embeddings endpoint"""
    print(f"Received job: {job.get('id', 'unknown')}", flush=True)

    try:
        # Lazy load model on first request
        llm = get_model()

        job_input = job["input"]
        texts = job_input.get("input", [])
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return {"error": "No input texts provided"}

        print(f"Processing {len(texts)} texts...", flush=True)

        embeddings = []
        total_tokens = 0

        for i, text in enumerate(texts):
            result = llm.embed(text)
            embeddings.append({
                "embedding": result,
                "index": i,
                "object": "embedding"
            })
            total_tokens += len(text.split()) * 1.3

        print(f"✓ Generated {len(embeddings)} embeddings", flush=True)

        return {
            "data": embeddings,
            "model": "Qwen3-Embedding-4B-Q8_0",
            "object": "list",
            "usage": {
                "prompt_tokens": int(total_tokens),
                "total_tokens": int(total_tokens)
            }
        }
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"ERROR: {error_msg}", flush=True)
        return {"error": str(e), "traceback": error_msg}


print("Starting RunPod serverless...", flush=True)
runpod.serverless.start({"handler": handler})
