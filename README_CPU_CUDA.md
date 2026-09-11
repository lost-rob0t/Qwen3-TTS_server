# Qwen3-TTS Server - CPU and CUDA Support

This fork adds support for both CPU and CUDA modes for the Qwen3-TTS server.

## Quick Start

### Building the Images

Use the provided build script:

```bash
# Build CPU version only
./build.sh cpu

# Build CUDA version only
./build.sh cuda

# Build both versions
./build.sh both
```

Or build manually:

```bash
# CPU version
docker build -f Dockerfile.cpu -t qwen3-tts_server:cpu .

# CUDA version
docker build -f Dockerfile.cuda -t qwen3-tts_server:cuda .
```

## Running the Server

### With Docker Run

**CPU Mode:**
```bash
docker run -p 7860:7860 qwen3-tts_server:cpu
```

**CUDA Mode:**
```bash
docker run --gpus all -p 7860:7860 qwen3-tts_server:cuda
```

## Configuration

### Environment Variables

- `QWEN_DEVICE`: Device to use for inference
  - `auto` (default for CUDA): Automatically detect and use GPU if available
  - `cpu`: Force CPU mode
  - `cuda:0`: Use specific GPU device

### Performance Comparison

| Mode | First Request | Subsequent | Memory | Quality |
|------|---------------|------------|--------|---------|
| CPU  | 10-30s        | 3-10s      | 8GB RAM | High |
| CUDA | 2-5s          | 0.5-1s     | 4GB VRAM | High |

**CPU Mode** is suitable for:
- Development and testing
- Low-volume deployments
- Systems without GPUs
- Cost-sensitive deployments

**CUDA Mode** is recommended for:
- Production deployments
- High-volume requests
- Real-time applications
- Systems with NVIDIA GPUs

## Architecture Changes

### Server Modifications

The `server.py` now includes:

1. **Dynamic Device Selection**:
   ```python
   device_env = os.getenv("QWEN_DEVICE", "auto")
   if device_env == "auto":
       device = "cuda:0" if torch.cuda.is_available() else "cpu"
   else:
       device = device_env
   ```

2. **Dtype Optimization**:
   ```python
   if device == "cpu":
       dtype = torch.float32  # Better CPU performance
   else:
       dtype = torch.bfloat16  # Better GPU performance
   ```

### Dockerfile Changes

Created separate Dockerfiles for CPU and CUDA:

- **Dockerfile.cpu**: Uses `ubuntu:22.04` base, installs CPU-only PyTorch
- **Dockerfile.cuda**: Uses `nvidia/cuda:12.8.0-runtime-ubuntu22.04`, installs CUDA-enabled PyTorch

Both versions:
- Pre-download models during build (no runtime downloads)
- Use multi-stage builds for smaller final images
- Include all necessary dependencies

## API Usage

The API is identical regardless of CPU or CUDA mode:

```bash
# Synthesize speech
curl "http://localhost:7860/synthesize_speech/?text=Hello%20world&voice=demo_speaker0" \
  --output output.wav

# Upload a voice
curl -X POST "http://localhost:7860/upload_audio/" \
  -F "audio_file_label=my_voice" \
  -F "file=@voice_sample.mp3"

# Use the uploaded voice
curl "http://localhost:7860/synthesize_speech/?text=Test&voice=my_voice" \
  --output output.wav
```

## Troubleshooting

### CPU Mode Issues

**Slow performance:**
- Expected on CPU, consider using a smaller model or CUDA mode
- Ensure sufficient RAM (8GB+)
- Close other applications to free resources

**Out of memory:**
- Reduce concurrent requests
- Increase system swap space
- Consider using CUDA mode with GPU

### CUDA Mode Issues

**CUDA not available:**
- Check GPU: `nvidia-smi`
- Verify Docker GPU support: `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi`
- Install NVIDIA Container Toolkit

**Wrong CUDA version:**
- Rebuild with appropriate CUDA version:
  ```bash
  docker build -f Dockerfile.cuda \
    --build-arg CUDA_VERSION=11.8.0 \
    -t qwen3-tts_server:cuda .
  ```

## Development

### Local Development (without Docker)

**CPU Mode:**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
export QWEN_DEVICE=cpu
python -m uvicorn server:app --host 0.0.0.0 --port 7860
```

**CUDA Mode:**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
export QWEN_DEVICE=auto
python -m uvicorn server:app --host 0.0.0.0 --port 7860
```

## License

Same as the original Qwen3-TTS project.
