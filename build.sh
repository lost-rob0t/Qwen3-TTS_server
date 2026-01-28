#!/bin/bash
# Build script for Qwen3-TTS server
# Supports both CPU and CUDA modes

set -e

MODE="${1:-cuda}"
TAG_SUFFIX=""

case "$MODE" in
    cpu)
        echo "Building CPU-only version..."
        docker build -f Dockerfile.cpu -t qwen3-tts_server:cpu .
        echo "✓ Built: qwen3-tts_server:cpu"
        ;;
    cuda|gpu)
        echo "Building CUDA version..."
        docker build -f Dockerfile.cuda -t qwen3-tts_server:cuda .
        docker tag qwen3-tts_server:cuda qwen3-tts_server:latest
        echo "✓ Built: qwen3-tts_server:cuda"
        echo "✓ Tagged: qwen3-tts_server:latest"
        ;;
    both)
        echo "Building both CPU and CUDA versions..."
        docker build -f Dockerfile.cpu -t qwen3-tts_server:cpu .
        echo "✓ Built: qwen3-tts_server:cpu"
        docker build -f Dockerfile.cuda -t qwen3-tts_server:cuda .
        docker tag qwen3-tts_server:cuda qwen3-tts_server:latest
        echo "✓ Built: qwen3-tts_server:cuda"
        echo "✓ Tagged: qwen3-tts_server:latest"
        ;;
    *)
        echo "Usage: $0 {cpu|cuda|both}"
        echo ""
        echo "Examples:"
        echo "  $0 cpu     # Build CPU-only version"
        echo "  $0 cuda    # Build CUDA version (default)"
        echo "  $0 both    # Build both versions"
        exit 1
        ;;
esac

echo ""
echo "Available images:"
docker images | grep qwen3-tts_server || echo "No images found"
