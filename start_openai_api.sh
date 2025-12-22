#!/bin/bash
# Start OpenAI-compatible Whisper API server

set -e

# Configuration
MODEL_DIR="${MODEL_DIR:-model}"
DEVICE="${DEVICE:-CPU}"  # Can be CPU, GPU, or AUTO
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
THREADS="${THREADS:-8}"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting OpenAI-compatible Whisper API...${NC}"
echo "Model directory: $MODEL_DIR"
echo "Device: $DEVICE"
echo "Host: $HOST"
echo "Port: $PORT"
echo "Threads: $THREADS"
echo ""

# Check if model exists
if [ ! -d "$MODEL_DIR" ]; then
    echo -e "${YELLOW}Warning: Model directory not found at $MODEL_DIR${NC}"
    echo "Run: python setup_model.py"
    exit 1
fi

# Kill any existing server on the port
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${YELLOW}Port $PORT is in use, stopping existing server...${NC}"
    pkill -f "serve_openai_api.py" || true
    sleep 2
fi

# Display device-specific information
if [ "$DEVICE" = "GPU" ]; then
    echo -e "${BLUE}Using Intel Iris Xe integrated graphics${NC}"
    echo -e "${BLUE}Ensure Intel GPU drivers are installed for optimal performance${NC}"
    echo ""
elif [ "$DEVICE" = "AUTO" ]; then
    echo -e "${BLUE}Auto-detecting best available device${NC}"
    echo ""
fi

# Start the server
echo -e "${GREEN}Starting server...${NC}"
python serve_openai_api.py \
    --model-dir "$MODEL_DIR" \
    --device "$DEVICE" \
    --host "$HOST" \
    --port "$PORT" \
    --threads "$THREADS"
