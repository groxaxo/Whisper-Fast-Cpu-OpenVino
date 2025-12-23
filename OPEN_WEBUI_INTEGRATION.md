# Open-WebUI Integration Guide

Complete guide for configuring this external Whisper server to be fully compatible with Open-WebUI.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration Steps](#configuration-steps)
- [Environment Variables](#environment-variables)
- [API Endpoints](#api-endpoints)
- [Testing Integration](#testing-integration)
- [Troubleshooting](#troubleshooting)

---

## Overview

This Whisper server provides OpenAI-compatible Speech-to-Text (STT) endpoints that integrate seamlessly with Open-WebUI. It uses OpenVINO for optimized CPU inference with support for:

- **Transcription** - Convert speech to text in the original language
- **Translation** - Translate foreign speech to English
- **Streaming support** - Handle chunked audio data
- **Multiple audio formats** - MP3, WAV, M4A, FLAC, OGG, WebM
- **Language detection** - Auto-detect or specify language
- **Timestamp support** - Optional word/segment timestamps

---

## Prerequisites

1. **Whisper Server Running**
   ```bash
   # Start the server (default port 8000)
   python serve_openai_api.py --model-dir model_int8_turbo --threads 12
   ```

2. **Open-WebUI Installed**
   ```bash
   # Install via pipx (recommended)
   pipx install open-webui
   
   # Or via pip
   pip install open-webui
   ```

3. **Network Connectivity**
   - Ensure the Whisper server is accessible from Open-WebUI
   - Default: `http://localhost:8000`
   - For remote access: `http://YOUR_IP:8000`

---

## Quick Start

### Option 1: Automatic Configuration (Recommended)

Use the included configuration script:

```bash
# Update Open-WebUI to use this Whisper server
python update_webui_config.py
```

### Option 2: Manual Configuration via Web UI

1. Open Open-WebUI in your browser
2. Navigate to: **Settings → Audio**
3. Configure STT settings:
   - **STT Engine**: `OpenAI`
   - **API Base URL**: `http://localhost:8000/v1`
   - **API Key**: Leave empty (not required)
   - **Model**: `whisper-1`
4. Click **Save**

### Option 3: Environment Variables

Set these before starting Open-WebUI:

```bash
export STT_OPENAI_API_BASE_URL="http://localhost:8000/v1"
export STT_OPENAI_API_KEY=""  # Empty for local server
export WHISPER_MODEL="whisper-1"
export STT_ENGINE="openai"
```

---

## Configuration Steps

### 1. Environment Variables and Persistent Configs

The Whisper server's configuration is managed via command-line arguments and environment variables:

#### Command-Line Arguments (Recommended)

```bash
python serve_openai_api.py \
  --model-dir model_int8_turbo \  # Model directory
  --device CPU \                   # CPU, GPU, or AUTO
  --host 0.0.0.0 \                # Bind to all interfaces
  --port 8000 \                    # API port
  --threads 12 \                   # Number of CPU threads (use physical cores)
  --streams AUTO \                 # Parallel inference streams
  --hint LATENCY                   # LATENCY or THROUGHPUT
```

#### Environment Variables

These are set automatically but can be overridden:

```bash
# Threading configuration
export OMP_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export MKL_NUM_THREADS=12
export NUMEXPR_NUM_THREADS=12
export OV_CPU_THREADS_NUM=12
```

### 2. Open-WebUI Configuration

#### A. Database Configuration (Direct)

The `update_webui_config.py` script directly modifies the Open-WebUI SQLite database:

```python
# Location varies by installation method:
# pipx: ~/.local/share/pipx/venvs/open-webui/lib/pythonX.X/site-packages/open_webui/data/webui.db
# pip: ~/.local/share/open-webui/webui.db
# docker: /app/backend/data/webui.db
```

**Configuration Structure:**
```json
{
  "audio": {
    "stt": {
      "openai": {
        "api_base_url": "http://localhost:8000/v1",
        "api_key": "",
        "model": "whisper-1"
      },
      "engine": "openai"
    }
  }
}
```

#### B. Web UI Configuration

Navigate to **Settings → Audio** in Open-WebUI:

**STT (Speech-to-Text) Settings:**
- **Engine**: Select `OpenAI`
- **API Base URL**: `http://localhost:8000/v1` (or your server URL)
- **API Key**: Leave empty for local server
- **Model**: `whisper-1`

**Advanced Options:**
- **Language**: Auto-detect or specify (e.g., `en`, `es`, `fr`, `de`)
- **Temperature**: `0.0` (recommended for consistency)
- **Response Format**: `json` (default)

### 3. Backend Handler Adaptations

The Whisper server implements all required OpenAI-compatible endpoints:

#### Transcription Handler
```
POST /v1/audio/transcriptions
```

**Supported Parameters:**
- `file` (required): Audio file
- `model`: Model name (default: `whisper-1`)
- `language`: Language code (e.g., `en`, `es`) or `auto`
- `prompt`: Optional context/prompt
- `response_format`: `json`, `text`, or `verbose_json`
- `temperature`: Sampling temperature (0.0 - 1.0)
- `timestamp_granularities`: `segment` for timestamps

**Example Request (cURL):**
```bash
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -F "file=@audio.mp3" \
  -F "model=whisper-1" \
  -F "language=en" \
  -F "response_format=json"
```

**Response:**
```json
{
  "text": "This is the transcribed text."
}
```

#### Translation Handler
```
POST /v1/audio/translations
```

Translates audio to English text.

**Example Request:**
```bash
curl -X POST "http://localhost:8000/v1/audio/translations" \
  -F "file=@spanish_audio.mp3" \
  -F "model=whisper-1"
```

### 4. Available Endpoints

| Endpoint | Method | Purpose | Open-WebUI Usage |
|----------|--------|---------|------------------|
| `/v1/audio/transcriptions` | POST | Transcribe audio | Primary STT endpoint |
| `/v1/audio/translations` | POST | Translate to English | Translation feature |
| `/v1/models` | GET | List models | Model discovery |
| `/health` | GET | Health check | Server monitoring |
| `/` | GET | API info | Documentation |
| `/docs` | GET | OpenAPI docs | Interactive API testing |

### 5. Configurable Audio Features

The server supports all major Whisper features:

#### Language Support
Specify language codes for better accuracy:
```bash
# In Open-WebUI audio settings
Language: en  # English
Language: es  # Spanish
Language: fr  # French
Language: de  # German
Language: auto  # Auto-detect
```

#### Timestamp Support
Enable segment timestamps:
```bash
# Via API request
timestamp_granularities=segment
```

Returns:
```json
{
  "text": "Full transcription...",
  "segments": [
    {"start": 0.0, "end": 2.5, "text": "First segment"},
    {"start": 2.5, "end": 5.0, "text": "Second segment"}
  ]
}
```

#### Response Formats
- `json` - Simple text response (default)
- `text` - Plain text only
- `verbose_json` - Full details with metadata

---

## API Endpoints

### Complete Endpoint Reference

#### 1. Transcription
```
POST /v1/audio/transcriptions
Content-Type: multipart/form-data
```

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | File | Yes | - | Audio file to transcribe |
| `model` | String | No | `whisper-1` | Model identifier |
| `language` | String | No | `auto` | Language code (ISO 639-1) |
| `prompt` | String | No | - | Context for better accuracy |
| `response_format` | String | No | `json` | `json`, `text`, or `verbose_json` |
| `temperature` | Float | No | `0.0` | Sampling temperature |
| `timestamp_granularities` | String | No | - | `segment` for timestamps |

#### 2. Translation
```
POST /v1/audio/translations
Content-Type: multipart/form-data
```

Translates any supported language to English.

#### 3. Model List
```
GET /v1/models
```

Returns available models:
```json
{
  "object": "list",
  "data": [
    {
      "id": "whisper-1",
      "object": "model",
      "created": 1703376000,
      "owned_by": "openvino"
    }
  ]
}
```

#### 4. Health Check
```
GET /health
```

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

## Testing Integration

### 1. Test Server Availability

```bash
# Check if server is running
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","model_loaded":true}
```

### 2. Test Transcription

```bash
# Create a test audio file or use existing
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -F "file=@test_audio.mp3" \
  -F "model=whisper-1" \
  -F "language=en"
```

### 3. Test from Open-WebUI

1. Open Open-WebUI in your browser
2. Navigate to a chat
3. Click the **microphone icon** 🎤
4. Record some audio
5. Check if transcription appears

### 4. Monitor Server Logs

```bash
# Watch server logs in real-time
tail -f api_server.log

# Or if running in foreground, check stdout
```

Expected log output:
```
2025-12-23 19:47:04 [INFO] Processing file: audio.mp3 (245678 bytes)
2025-12-23 19:47:04 [INFO] Audio duration: 5.23s, sample_rate: 16000
2025-12-23 19:47:05 [INFO] Transcription completed: 42 chars, took 0.85s (6.15x realtime)
```

---

## Troubleshooting

### Common Issues

#### 1. Connection Refused
**Error:** `Connection refused` or `Server Connection Error`

**Solutions:**
- Verify server is running: `curl http://localhost:8000/health`
- Check firewall rules: `sudo ufw allow 8000`
- Ensure correct URL in Open-WebUI: `http://localhost:8000/v1`
- For remote access, use server IP: `http://YOUR_IP:8000/v1`

#### 2. Model Not Found
**Error:** `Model configuration file not found`

**Solution:**
```bash
# Download the model first
python setup_model.py
```

#### 3. Empty Transcriptions
**Error:** Transcriptions return empty text

**Solutions:**
- Check audio file format (supported: MP3, WAV, M4A, FLAC, OGG, WebM)
- Verify audio has actual speech content
- Check server logs for errors
- Try different sample rate or format

#### 4. Slow Performance
**Issue:** Transcription takes too long

**Solutions:**
```bash
# Use INT8 quantized model
python serve_openai_api.py --model-dir model_int8_turbo

# Optimize thread count (use physical cores)
python serve_openai_api.py --threads 12  # For i5-1240P

# Use LATENCY hint for real-time
python serve_openai_api.py --hint LATENCY

# Enable streaming
python serve_openai_api.py --streams AUTO
```

#### 5. Database Configuration Not Persisting
**Issue:** Settings reset after restart

**Solution:**
```bash
# Ensure correct database path in update_webui_config.py
# For pipx installation:
db_path = "~/.local/share/pipx/venvs/open-webui/lib/python3.11/site-packages/open_webui/data/webui.db"

# For pip installation:
db_path = "~/.local/share/open-webui/webui.db"

# For Docker:
db_path = "/app/backend/data/webui.db"
```

### Debugging Steps

1. **Enable verbose logging:**
   ```bash
   # Server logs all requests automatically
   tail -f api_server.log
   ```

2. **Test with cURL:**
   ```bash
   # Direct API test
   curl -v -X POST "http://localhost:8000/v1/audio/transcriptions" \
     -F "file=@test.mp3" \
     -F "model=whisper-1"
   ```

3. **Check Open-WebUI logs:**
   ```bash
   # For pipx installation
   journalctl -u open-webui -f
   
   # Or check browser console for errors
   ```

4. **Verify configuration:**
   ```bash
   # Check database directly
   python - << 'EOF'
   import sqlite3, json
   conn = sqlite3.connect("~/.local/share/pipx/venvs/open-webui/lib/python3.11/site-packages/open_webui/data/webui.db")
   cursor = conn.cursor()
   cursor.execute("SELECT data FROM config LIMIT 1")
   config = json.loads(cursor.fetchone()[0])
   print(json.dumps(config.get("audio", {}), indent=2))
   conn.close()
   EOF
   ```

### Performance Optimization

For optimal real-time transcription on **Intel i5-1240P** (12 physical cores):

```bash
python serve_openai_api.py \
  --model-dir model_int8_turbo \
  --device CPU \
  --threads 12 \
  --streams AUTO \
  --hint LATENCY \
  --host 0.0.0.0 \
  --port 8000
```

**Expected Performance:**
- Real-time factor: **6-10x** (transcribes 6-10 seconds of audio per second)
- Latency: **< 1 second** for typical queries
- Memory: **~500-800 MB**

---

## Advanced Configuration

### Remote Server Setup

To access from other machines:

```bash
# Start server on all interfaces
python serve_openai_api.py --host 0.0.0.0 --port 8000

# Configure Open-WebUI
export STT_OPENAI_API_BASE_URL="http://YOUR_SERVER_IP:8000/v1"
```

### Auto-Start on Boot

Create systemd service:

```bash
# Edit whisper-server.service
sudo nano /etc/systemd/system/whisper-server.service
```

```ini
[Unit]
Description=OpenVINO Whisper API Server
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/home/YOUR_USERNAME/Whisper-Fast-Cpu-OpenVino
ExecStart=/usr/bin/python3 serve_openai_api.py --model-dir model_int8_turbo --threads 12
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start
sudo systemctl enable whisper-server
sudo systemctl start whisper-server
```

### Load Balancing

For high-load scenarios, run multiple instances:

```bash
# Instance 1
python serve_openai_api.py --port 8000 --threads 6

# Instance 2
python serve_openai_api.py --port 8001 --threads 6
```

Then use nginx or HAProxy for load balancing.

---

## API Compatibility Matrix

| Feature | OpenAI Whisper API | This Server | Open-WebUI |
|---------|-------------------|-------------|------------|
| Transcription | ✅ | ✅ | ✅ |
| Translation | ✅ | ✅ | ✅ |
| Language detection | ✅ | ✅ | ✅ |
| Timestamps | ✅ | ✅ | ⚠️ (limited) |
| Multiple formats | ✅ | ✅ | ✅ |
| Streaming | ⚠️ | ✅ | ✅ |
| Prompt support | ✅ | ✅ | ⚠️ (limited) |

✅ = Fully supported  
⚠️ = Partially supported  
❌ = Not supported

---

## Support and Resources

- **Server Documentation**: `README.md`
- **API Documentation**: `http://localhost:8000/docs` (when server is running)
- **Open-WebUI Docs**: https://docs.openwebui.com/
- **Whisper Model**: https://github.com/openai/whisper
- **OpenVINO**: https://github.com/openvinotoolkit/openvino

---

## License

This integration guide is part of the Whisper-Fast-Cpu-OpenVino project.
See `LICENSE` file for details.
