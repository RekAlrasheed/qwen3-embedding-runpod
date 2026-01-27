"""
RunPod Serverless Handler for Qwen3-Embedding-4B-Q8_0
Exact same model as local - 100% embedding compatibility
"""
import runpod
import os
import sys

print("=" * 60)
print("STARTING HANDLER")
print("=" * 60)

# Debug: Print environment and check files
MODEL_PATH = os.getenv("MODEL_PATH", "/models/Qwen3-Embedding-4B-q8_0.gguf")
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "-1"))

print(f"MODEL_PATH env: {MODEL_PATH}")
print(f"N_GPU_LAYERS env: {N_GPU_LAYERS}")

# Check if model file exists
print(f"\nChecking for model file...")
if os.path.exists(MODEL_PATH):
    size_gb = os.path.getsize(MODEL_PATH) / (1024**3)
    print(f"✓ Model found at {MODEL_PATH} ({size_gb:.2f} GB)")
else:
    print(f"✗ Model NOT found at {MODEL_PATH}")
    # List what's in /models
    if os.path.exists("/models"):
        print(f"\nContents of /models:")
        for item in os.listdir("/models"):
            full_path = os.path.join("/models", item)
            if os.path.isfile(full_path):
                size = os.path.getsize(full_path) / (1024**3)
                print(f"  - {item} ({size:.2f} GB)")
            else:
                print(f"  - {item}/ (directory)")
    else:
        print("/models directory does not exist")
    sys.exit(1)

# Now try to load the model
print(f"\nLoading llama-cpp-python...")
try:
    from llama_cpp import Llama
    print("✓ llama_cpp imported successfully")
except Exception as e:
    print(f"✗ Failed to import llama_cpp: {e}")
    sys.exit(1)

print(f"\nInitializing model with n_gpu_layers={N_GPU_LAYERS}...")
try:
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=N_GPU_LAYERS,
        embedding=True,
        n_ctx=8192,
        verbose=True  # Enable verbose to see CUDA info
    )
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test embedding
print("\nTesting embedding generation...")
try:
    test_result = llm.embed("test")
    print(f"✓ Test embedding generated, dimension: {len(test_result)}")
except Exception as e:
    print(f"✗ Test embedding failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("HANDLER READY")
print("=" * 60)


def handler(job):
    """
    RunPod handler - OpenAI-compatible embeddings endpoint
    """
    try:
        job_input = job["input"]

        # Handle both single string and list of strings
        texts = job_input.get("input", [])
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return {"error": "No input texts provided"}

        # Generate embeddings
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
        return {"error": str(e), "traceback": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
