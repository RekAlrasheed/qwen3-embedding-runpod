# RunPod Serverless - Qwen3-Embedding-4B-Q8_0

Fast GPU-accelerated embeddings using the **exact same Q8_0 quantized model** as local.
100% embedding compatibility with existing Weaviate index.

## Setup Instructions

### Step 1: Create RunPod Account
1. Go to [runpod.io](https://runpod.io)
2. Sign up and add ~$10 credits

### Step 2: Build & Push Docker Image

```bash
# Login to Docker Hub (or use RunPod's registry)
docker login

# Build the image
cd runpod-embedding
docker build -t YOUR_DOCKERHUB/qwen3-embedding-q8:latest .

# Push to registry
docker push YOUR_DOCKERHUB/qwen3-embedding-q8:latest
```

### Step 3: Create Serverless Endpoint on RunPod

1. Go to RunPod Console → Serverless → New Endpoint
2. Configure:
   - **Container Image**: `YOUR_DOCKERHUB/qwen3-embedding-q8:latest`
   - **GPU**: RTX 4090 or A100 (recommended)
   - **Max Workers**: 1-3 (based on usage)
   - **Idle Timeout**: 60 seconds
   - **Flash Boot**: Enabled
3. Deploy and get your endpoint URL

### Step 4: Get API Credentials
- Copy your **Endpoint ID** and **API Key** from RunPod dashboard

### Step 5: Test the Endpoint

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "input": ["Hello world", "Test embedding"]
    }
  }'
```

## Integration with TWK Portal

Update your `.env.local`:

```env
# Switch to RunPod Embeddings
EMBEDDING_SERVICE_URL=https://api.runpod.ai/v2/YOUR_ENDPOINT_ID
RUNPOD_API_KEY=YOUR_API_KEY
```

Or modify the embedding service to call RunPod API.

## Pricing

- RTX 4090: ~$0.44/hr active time
- A100 40GB: ~$1.04/hr active time
- Scales to zero when idle (no cost)

Estimated cost: ~$0.01-0.02 per full pipeline run (111 keywords)

## Model Details

- **Model**: Qwen3-Embedding-4B
- **Quantization**: Q8_0 (8-bit)
- **Dimensions**: 2560
- **Size**: 4.28 GB
- **Source**: [Mungert/Qwen3-Embedding-4B-GGUF](https://huggingface.co/Mungert/Qwen3-Embedding-4B-GGUF)
