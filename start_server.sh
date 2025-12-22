#!/bin/bash
# Start OpenVINO Whisper server on CPU or GPU using conda environment

# Configuration
DEVICE="${DEVICE:-CPU}"  # Default to CPU, can be set to GPU or AUTO
PORT="${PORT:-7860}"

# Kill any existing server processes
echo "Stopping any existing servers..."
pkill -f "serve_whisper.py" || true
sleep 2

# Start the server
echo "Starting OpenVINO Whisper server on $DEVICE..."
if [ "$DEVICE" = "GPU" ]; then
    echo "Using Intel Iris Xe integrated graphics"
    conda run -n ov-whisper python serve_whisper.py --device GPU --port $PORT > server.log 2>&1 &
elif [ "$DEVICE" = "AUTO" ]; then
    echo "Auto-detecting best device"
    conda run -n ov-whisper python serve_whisper.py --device AUTO --port $PORT > server.log 2>&1 &
else
    echo "Using CPU"
    conda run -n ov-whisper python serve_whisper.py --device CPU --port $PORT > server.log 2>&1 &
fi

sleep 8

# Check if server is running
if lsof -i :$PORT > /dev/null 2>&1; then
    echo "✓ Server started successfully on $DEVICE!"
    echo ""
    echo "  Local URL:  http://localhost:$PORT"
    echo "  Device:     $DEVICE"
    echo "  View logs:  tail -f server.log"
    echo ""
    echo "  For remote access, run:"
    echo "    ngrok http $PORT"
    echo ""
    echo "  To use GPU, run:"
    echo "    DEVICE=GPU ./start_server.sh"
else
    echo "✗ Server failed to start. Check server.log for errors."
    tail -20 server.log
    exit 1
fi
