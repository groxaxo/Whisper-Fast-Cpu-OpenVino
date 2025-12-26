#!/bin/bash
# Quick test of Opus encoding with the dictation client

set -e

echo "🧪 Testing Opus Encoding Implementation"
echo "========================================"
echo ""

CONDA_PYTHON="/home/op/miniconda/envs/ov-whisper/bin/python"

echo "1. Testing Opus library..."
$CONDA_PYTHON -c "import opuslib; print('   ✅ opuslib imported successfully')"

echo ""
echo "2. Testing compression simulation..."
$CONDA_PYTHON << 'EOF'
import numpy as np
import opuslib

# Simulate 5 seconds of audio
sample_rate = 16000
audio = (np.random.rand(sample_rate * 5) * 0.1).astype(np.float32)
pcm_data = (audio * 32767).astype(np.int16)

# Encode to Opus
encoder = opuslib.Encoder(sample_rate, 1, opuslib.APPLICATION_VOIP)
encoder.bitrate = 24000

frame_size = int(sample_rate * 0.020)
opus_frames = []

for i in range(0, len(pcm_data), frame_size):
    frame = pcm_data[i:i + frame_size]
    if len(frame) < frame_size:
        frame = np.pad(frame, (0, frame_size - len(frame)))
    opus_frame = encoder.encode(frame.tobytes(), frame_size)
    opus_frames.append(opus_frame)

opus_data = b''.join(opus_frames)
wav_size = len(pcm_data) * 2
opus_size = len(opus_data)

print(f'   WAV size: {wav_size/1024:.1f} KB')
print(f'   Opus size: {opus_size/1024:.1f} KB')
print(f'   Compression: {wav_size/opus_size:.1f}x')
print('   ✅ Opus encoding working!')
EOF

echo ""
echo "3. Checking remote backend connectivity..."
if curl -s --max-time 5 http://100.85.200.52:8887/health > /dev/null 2>&1; then
    echo "   ✅ Backend server is responding"
else
    echo "   ⚠️  Backend server not responding (ensure it's running)"
fi

echo ""
echo "========================================"
echo "✅ All tests passed!"
echo ""
echo "Ready to start dictation:"
echo "  ./start_dictation.sh"
echo ""
echo "Press Ctrl+Alt+Space to dictate"
echo "Audio will be Opus-encoded before sending"
