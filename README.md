# Qwen3-TTS Server

A FastAPI-based text-to-speech server powered by Qwen3-TTS, featuring voice cloning, voice conversion, and multi-language support.

## Features

- **Voice cloning** - Clone any voice from a short reference audio clip
- **Voice conversion** - Transform the voice of existing audio files
- **Multi-language support** - 10 languages: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian
- **Automatic transcription** - Uses Whisper for reference audio transcription
- **Speed control** - Adjust speech speed via time-stretching
- **RunPod ready** - Optimized for deployment on RunPod with network volume support
- **AMD ROCm** - Runs on supported AMD GPUs without CUDA-only FlashAttention
- **Nix flake** - One command for builds, service installation, diagnostics, and voices
- **Persistent voices** - Add voices from audio or video while the server is running

## NixOS + AMD

Docker must be enabled and the host must expose `/dev/kfd` and `/dev/dri`. Then:

```bash
git clone https://github.com/lost-rob0t/Qwen3-TTS_server
cd Qwen3-TTS_server

# Check host access, build on first start, and keep it running.
nix run .# -- doctor rocm
nix run .# -- install-user rocm

# Optional: start at boot even before this user logs in.
sudo loginctl enable-linger "$USER"

# Verify the live backend.
nix run .# -- status
```

The first start builds the ROCm image and downloads the Qwen and Whisper models. Later starts reuse the image and the persistent model cache. Client applications can connect to:

```text
http://127.0.0.1:7860
```

The API binds only to loopback by default because it has no authentication.

### Voice manager

The same flake command extracts and normalizes audio from WAV, FLAC, MP3, M4A, other FFmpeg-supported audio, and video files. Voices are available without restarting the service.

```bash
# A local audio file.
nix run .# -- voice add narrator ./reference.mp3 \
  --text "Write the exact words spoken here when you know them."

# A 15-second clip beginning 1 minute 12 seconds into a video.
nix run .# -- voice add interview ./interview.mp4 --start 00:01:12

nix run .# -- voice list
nix run .# -- voice info narrator
nix run .# -- voice set-text narrator "Corrected exact transcript."
nix run .# -- voice rename interview guest
nix run .# -- voice set-default narrator
nix run .# -- voice export narrator ./narrator-reference.wav
nix run .# -- voice remove guest
```

Supplying `--text` avoids a Whisper transcription pass and gives cloning a more accurate reference. Re-import with `--replace` to update an existing voice. Data lives under `${XDG_DATA_HOME:-$HOME/.local/share}/qwen3-tts` unless `QWEN3_TTS_DATA_DIR` is set.

### Declarative NixOS service

```nix
{
  inputs.qwen3-tts.url = "github:lost-rob0t/Qwen3-TTS_server";

  outputs = inputs@{ nixpkgs, qwen3-tts, ... }: {
    nixosConfigurations.your-host = nixpkgs.lib.nixosSystem {
      modules = [
        qwen3-tts.nixosModules.default
        {
          services.qwen3-tts = {
            enable = true;
            backend = "rocm";
          };
        }
      ];
    };
  };
}
```

A Home Manager module is also exported as `qwen3-tts.homeManagerModules.default`. The user must be able to access Docker; for ROCm, the host kernel/driver must provide the AMD devices.

### AMD compatibility

The image uses AMD's tested `rocm/pytorch:latest` base and passes both AMD device nodes into the container. PyTorch intentionally reports ROCm GPUs through `torch.cuda`, so `cuda:0` in the health response is normal; the separate `backend` field reports `rocm`.

This supports GPUs present in AMD's current ROCm compatibility matrix. Old Polaris cards such as the RX 580 are not supported by current ROCm/PyTorch builds. Unsupported RDNA cards sometimes need a host-specific override, which the launcher passes through when explicitly set:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 nix run .# -- doctor rocm
HSA_OVERRIDE_GFX_VERSION=10.3.0 nix run .# -- install-user rocm
```

An override is not emulation and cannot make a fundamentally unsupported instruction set work. Confirm the correct value for the actual GPU before using it. See AMD's [ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html) and [PyTorch container instructions](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html).

To pin or replace the default ROCm base image, set `QWEN_ROCM_IMAGE` before the first build. For example, `QWEN_ROCM_IMAGE=rocm/pytorch:<tested-tag> nix run .# -- build rocm`.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/base_tts/` | GET | TTS with default English voice |
| `/synthesize_speech/` | GET | TTS with specified voice |
| `/upload_audio/` | POST | Upload reference voice |
| `/change_voice/` | POST | Voice conversion |
| `/voices/` | GET | List persistent reference voices |
| `/health` | GET | Show readiness, backend, device, and dtype |

### Endpoint Details

#### `GET /synthesize_speech/`

Generate speech from text using a specified voice.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | Yes | Text to synthesize |
| `voice` | string | Yes | Voice label (filename prefix in `resources/`) |
| `speed` | float | No | Speech speed multiplier (default: 1.0) |

#### `GET /base_tts/`

Generate speech using the default English voice.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | Yes | Text to synthesize |
| `speed` | float | No | Speech speed multiplier (default: 1.0) |

#### `POST /upload_audio/`

Upload an audio file to use as a reference voice.

| Field | Type | Description |
|-------|------|-------------|
| `audio_file_label` | string | Label/name for the voice |
| `file` | file | Audio file (wav, mp3, flac, ogg; max 5MB) |

#### `POST /change_voice/`

Convert the voice of an existing audio file.

| Field | Type | Description |
|-------|------|-------------|
| `reference_speaker` | string | Voice label to convert to |
| `file` | file | Audio file to convert |

## Quick Start

### Using Docker (Recommended)

```bash
# Build the image
docker build -t qwen3-tts_server .

# Run the container
docker run --gpus all -p 7860:7860 qwen3-tts_server
```

For AMD:

```bash
docker build -f Dockerfile.rocm -t qwen3-tts_server:rocm .
docker run --device=/dev/kfd --device=/dev/dri --ipc=host \
  --security-opt seccomp=unconfined -p 127.0.0.1:7860:7860 \
  qwen3-tts_server:rocm
```

### Local Development

```bash
# Create conda environment
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts

# Install PyTorch with CUDA
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install dependencies
pip install -r requirements.txt

# Optional: Install FlashAttention 2 for better performance
pip install flash-attn --no-build-isolation

# Run the server
./start.sh
# or
python -m uvicorn server:app --host 0.0.0.0 --port 7860
```

## Directory Structure

```
.
├── server.py           # Main FastAPI server
├── Dockerfile          # Multi-stage Docker build
├── requirements.txt    # Python dependencies
├── start.sh            # Startup script
├── resources/          # Voice reference files
│   └── demo_speaker0.mp3
└── outputs/            # Generated audio (temporary)
```

## Usage Examples

### Synthesize Speech

```bash
curl "http://localhost:7860/synthesize_speech/?text=Hello%20world&voice=demo_speaker0" \
  --output output.wav
```

### Upload a Voice

```bash
curl -X POST "http://localhost:7860/upload_audio/" \
  -F "audio_file_label=my_voice" \
  -F "file=@/path/to/voice_sample.mp3"
```

### Use Uploaded Voice

```bash
curl "http://localhost:7860/synthesize_speech/?text=Hello%20world&voice=my_voice" \
  --output output.wav
```

### Change Voice of Audio

```bash
curl -X POST "http://localhost:7860/change_voice/" \
  -F "reference_speaker=demo_speaker0" \
  -F "file=@/path/to/input.wav" \
  --output converted.wav
```

## Models

| Model | Purpose | Size |
|-------|---------|------|
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Voice cloning TTS | 1.7B parameters |
| `openai/whisper-base` | Audio transcription | 74M parameters |

CUDA/CPU images pre-download models during their build. The ROCm launcher downloads them on first start and keeps them in the persistent cache.

## Requirements

- **GPU**: Supported AMD ROCm or NVIDIA CUDA GPU with 8GB+ VRAM recommended
- **ROCm/CUDA**: A host driver compatible with the selected container backend
- **Python**: 3.10+

## RunPod Deployment

The Docker image is optimized for RunPod:
- Server files are in `/app/server/` (not `/workspace/`)
- `/workspace/` is left free for network volumes
- Runs Uvicorn as PID 1 for clean Docker and systemd lifecycle handling

## License

See the respective licenses for:
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- [OpenAI Whisper](https://github.com/openai/whisper)
