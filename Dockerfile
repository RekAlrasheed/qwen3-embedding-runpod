# RunPod Serverless - Qwen3-Embedding-4B-Q8_0 (CUDA)
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install llama-cpp-python with CUDA support
RUN pip install --upgrade pip && \
    CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python==0.3.4 --no-cache-dir

# Install RunPod SDK
RUN pip install runpod==1.7.0

# Copy handler
COPY handler.py .

# Download model from HuggingFace (Q8_0 version - exact match)
RUN pip install huggingface_hub && \
    python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='Mungert/Qwen3-Embedding-4B-GGUF', \
    filename='Qwen3-Embedding-4B-q8_0.gguf', \
    local_dir='/models')"

ENV MODEL_PATH=/models/Qwen3-Embedding-4B-q8_0.gguf
ENV N_GPU_LAYERS=-1

CMD ["python", "-u", "handler.py"]
