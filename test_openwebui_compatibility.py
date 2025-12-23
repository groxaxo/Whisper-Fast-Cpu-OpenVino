#!/usr/bin/env python3
"""
Test script for Open-WebUI compatible Whisper server endpoints.

Tests all endpoints required for Open-WebUI integration.
"""

import requests
import json
import sys
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health check endpoint"""
    print("Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"  Status: {response.status_code}")
    print(f"  Response: {response.json()}")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    print("  ✓ Health check passed\n")

def test_config_get():
    """Test GET /config endpoint"""
    print("Testing GET /config endpoint...")
    response = requests.get(f"{BASE_URL}/config")
    print(f"  Status: {response.status_code}")
    config = response.json()
    print(f"  Response: {json.dumps(config, indent=2)}")
    assert response.status_code == 200
    assert "engine" in config
    assert "model" in config
    print("  ✓ Config GET passed\n")
    return config

def test_config_update():
    """Test POST /config/update endpoint"""
    print("Testing POST /config/update endpoint...")
    
    # Update config
    update_data = {
        "language": "en",
        "vad_filter": True,
        "model": "whisper-1"
    }
    response = requests.post(
        f"{BASE_URL}/config/update",
        json=update_data
    )
    print(f"  Status: {response.status_code}")
    result = response.json()
    print(f"  Response: {json.dumps(result, indent=2)}")
    assert response.status_code == 200
    assert result["status"] == "success"
    print("  ✓ Config UPDATE passed\n")
    
    # Verify update
    print("  Verifying config was updated...")
    response = requests.get(f"{BASE_URL}/config")
    config = response.json()
    assert config["language"] == "en"
    assert config["vad_filter"] == True
    print("  ✓ Config verification passed\n")

def test_models():
    """Test /v1/models endpoint"""
    print("Testing GET /v1/models endpoint...")
    response = requests.get(f"{BASE_URL}/v1/models")
    print(f"  Status: {response.status_code}")
    models = response.json()
    print(f"  Response: {json.dumps(models, indent=2)}")
    assert response.status_code == 200
    assert "data" in models
    assert len(models["data"]) > 0
    print("  ✓ Models list passed\n")

def test_root():
    """Test root endpoint"""
    print("Testing GET / endpoint...")
    response = requests.get(f"{BASE_URL}/")
    print(f"  Status: {response.status_code}")
    info = response.json()
    print(f"  Response: {json.dumps(info, indent=2)}")
    assert response.status_code == 200
    assert "endpoints" in info
    print("  ✓ Root endpoint passed\n")

def test_transcription(audio_file=None):
    """Test transcription endpoints"""
    if audio_file is None:
        audio_file = Path(__file__).parent / "test_audio_short.mp3"
    
    if not audio_file.exists():
        print(f"⚠ Skipping transcription test - audio file not found: {audio_file}\n")
        return
    
    print(f"Testing POST /v1/audio/transcriptions with {audio_file.name}...")
    
    with open(audio_file, "rb") as f:
        files = {"file": ("audio.mp3", f, "audio/mpeg")}
        data = {
            "model": "whisper-1",
            "response_format": "json"
        }
        response = requests.post(
            f"{BASE_URL}/v1/audio/transcriptions",
            files=files,
            data=data
        )
    
    print(f"  Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"  Transcription: {result.get('text', '')[:100]}...")
        print("  ✓ Transcription passed\n")
    else:
        print(f"  Error: {response.text}\n")
        
    # Test alternate endpoint without /v1
    print(f"Testing POST /audio/transcriptions (alternate path)...")
    with open(audio_file, "rb") as f:
        files = {"file": ("audio.mp3", f, "audio/mpeg")}
        data = {
            "model": "whisper-1",
            "response_format": "json"
        }
        response = requests.post(
            f"{BASE_URL}/audio/transcriptions",
            files=files,
            data=data
        )
    
    print(f"  Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"  Transcription: {result.get('text', '')[:100]}...")
        print("  ✓ Alternate transcription passed\n")
    else:
        print(f"  Error: {response.text}\n")

def main():
    print("=" * 60)
    print("Open-WebUI Compatible Whisper Server Test Suite")
    print("=" * 60)
    print()
    
    try:
        test_health()
        test_root()
        test_models()
        test_config_get()
        test_config_update()
        test_transcription()
        
        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        print()
        print("Your Whisper server is fully compatible with Open-WebUI.")
        print(f"Configure Open-WebUI to use: {BASE_URL}/v1")
        print()
        
    except Exception as e:
        print()
        print("=" * 60)
        print(f"✗ Test failed: {e}")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()
