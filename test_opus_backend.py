#!/usr/bin/env python3
"""
Test script to verify Opus encoding works with the Whisper backend.
This creates a test Opus file and sends it to the server.
"""

import numpy as np
import opuslib
import io
import requests
import sys

def create_opus_test_file(duration=3.0, sample_rate=16000):
    """Create a test Opus-encoded audio file."""
    print(f"📝 Creating {duration}s test audio...")
    
    # Generate test audio (low amplitude noise)
    audio = (np.random.rand(int(sample_rate * duration)) * 0.01).astype(np.float32)
    pcm_data = (audio * 32767).astype(np.int16)
    
    # Encode to Opus
    print("🔧 Encoding to Opus...")
    encoder = opuslib.Encoder(sample_rate, 1, opuslib.APPLICATION_VOIP)
    encoder.bitrate = 24000
    
    frame_size = int(sample_rate * 0.020)
    opus_frames = []
    
    for i in range(0, len(pcm_data), frame_size):
        frame = pcm_data[i:i + frame_size]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)), mode='constant')
        opus_frame = encoder.encode(frame.tobytes(), frame_size)
        opus_frames.append(opus_frame)
    
    opus_data = b''.join(opus_frames)
    print(f"✅ Created Opus file: {len(opus_data)} bytes ({len(opus_data)/1024:.1f} KB)")
    
    return opus_data

def test_backend(server_url, opus_data):
    """Test sending Opus file to the backend."""
    print(f"\n📤 Testing server: {server_url}")
    
    try:
        opus_buffer = io.BytesIO(opus_data)
        response = requests.post(
            f"{server_url}/v1/audio/transcriptions",
            files={'file': ('test_audio.opus', opus_buffer, 'audio/ogg')},
            data={'model': 'whisper-1'},
            timeout=30
        )
        
        print(f"📊 Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ SUCCESS!")
            print(f"📝 Transcription: {result.get('text', '(empty)')}")
            return True
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("⏱️  Timeout - server may not be running or is unreachable")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"🔌 Connection error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Opus Backend Compatibility Test")
    print("=" * 60)
    
    # Get server URL from command line or use default
    if len(sys.argv) > 1:
        server_url = sys.argv[1].rstrip('/')
    else:
        server_url = "http://100.85.200.52:8887"
    
    # Create test Opus file
    opus_data = create_opus_test_file(duration=3.0)
    
    # Test the backend
    success = test_backend(server_url, opus_data)
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Backend successfully accepts and processes Opus files!")
    else:
        print("⚠️  Backend test failed - check server status")
        print("\n💡 To test with a different server:")
        print(f"   python {sys.argv[0]} http://your-server:8000")
    print("=" * 60)
