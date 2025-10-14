#!/bin/bash
# Start OpenVINO Whisper server on CPU using conda environment

# Kill any existing server processes
echo "Stopping any existing servers..."
pkill -f "serve_whisper.py"
sleep 2

# Start the server
echo "Starting OpenVINO Whisper server..."
conda run -n ov-whisper python serve_whisper.py --device CPU --port 7860 > server.log 2>&1 &

sleep 8

# Check if server is running
if lsof -i :7860 > /dev/null 2>&1; then
    echo "✓ Server started successfully!"
    echo ""
    echo "  Local URL:  http://localhost:7860"
    echo "  View logs:  tail -f server.log"
    echo ""
    echo "  For remote access, run:"
    echo "    ngrok http 7860"
else
    echo "✗ Server failed to start. Check server.log for errors."
    tail -20 server.log
    exit 1
fi
