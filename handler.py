"""
RunPod Serverless Handler for Qwen3-Embedding-4B-Q8_0
Exact same model as local - 100% embedding compatibility
"""
import runpod
from llama_cpp import Llama
import os

# Initialize model once (stays in memory between requests)
MODEL_PATH = os.getenv("MODEL_PATH", "/runpod-volume/Qwen3-Embedding-4B-Q8_0.gguf")
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "-1"))  # -1 = all layers on GPU

print(f"Loading model from {MODEL_PATH}...")
llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=N_GPU_LAYERS,
    embedding=True,
    n_ctx=8192,
    verbose=False
)
print("Model loaded successfully!")


def handler(job):
    """
    RunPod handler - OpenAI-compatible embeddings endpoint

    Input format:
    {
        "input": ["text1", "text2", ...] or "single text"
    }

    Output format (OpenAI-compatible):
    {
        "data": [
            {"embedding": [...], "index": 0},
            {"embedding": [...], "index": 1},
            ...
        ],
        "model": "Qwen3-Embedding-4B-Q8_0",
        "usage": {"prompt_tokens": N, "total_tokens": N}
    }
    """
    job_input = job["input"]

    # Handle both single string and list of strings
    texts = job_input.get("input", [])
    if isinstance(texts, str):
        texts = [texts]

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
        # Approximate token count
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


runpod.serverless.start({"handler": handler})
