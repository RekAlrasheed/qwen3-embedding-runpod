# RunPod Serverless - Qwen3-Embedding-4B-Q8_0 (CUDA)
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install llama-cpp-python with CUDA support
RUN pip install --upgrade pip && \
    CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python==0.3.4 --no-cache-dir

# Install RunPod SDK
RUN pip install runpod==1.7.0

# Download model directly using wget (more reliable than huggingface_hub)
RUN mkdir -p /models && \
    wget -q --show-progress -O /models/Qwen3-Embedding-4B-q8_0.gguf \
    "https://huggingface.co/Mungert/Qwen3-Embedding-4B-GGUF/resolve/main/Qwen3-Embedding-4B-q8_0.gguf" && \
    ls -la /models/ && \
    echo "Model size:" && du -h /models/Qwen3-Embedding-4B-q8_0.gguf

# Copy handler
COPY handler.py .

ENV MODEL_PATH=/models/Qwen3-Embedding-4B-q8_0.gguf
ENV N_GPU_LAYERS=-1

CMD ["python", "-u", "handler.py"]
