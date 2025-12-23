#!/bin/bash
# Start OpenVINO Whisper API Server
# Optimized for Intel Core i5-1240P (P-cores only, with hyperthreading)

set -e

# Configuration - INT8 Turbo model for best speed/quality balance
MODEL_DIR="${MODEL_DIR:-model_int8_turbo}"
DEVICE="${DEVICE:-CPU}"
PORT="${PORT:-8000}"
# Use 8 threads = 4 P-cores × 2 hyperthreads
THREADS="${THREADS:-8}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Starting OpenVINO Whisper API (Optimized for i5-1240P)${NC}"
echo -e "${BLUE}Model: $MODEL_DIR${NC}"
echo -e "${BLUE}Configuration: Device=$DEVICE, Threads=$THREADS, Streams=1, Hint=LATENCY${NC}"
echo -e "${YELLOW}CPU Affinity: Pinned to P-cores (CPUs 0-7)${NC}"

# Check model
if [ ! -d "$MODEL_DIR" ]; then
    echo "Model directory not found. Downloading INT8 Turbo model..."
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='bweng/whisper-large-v3-turbo-int8-ov', local_dir='$MODEL_DIR')"
fi

# Run server with CPU pinning to P-cores only (0-7)
# - taskset -c 0-7: Pin to P-cores, avoid E-cores (8-15)
# - Threads=8: Use all P-core hyperthreads for max parallelism
# - Streams=1: Single stream for lowest latency per request
# - Hint=LATENCY: Optimize for response time
taskset -c 0-7 python serve_openai_api.py \
    --model-dir "$MODEL_DIR" \
    --device "$DEVICE" \
    --threads "$THREADS" \
    --streams 1 \
    --hint LATENCY \
    --port "$PORT"
