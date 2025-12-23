# Quick Open-WebUI Configuration Guide

## ✅ Server Status

Your Whisper server is **running and fully compatible** with Open-WebUI!

**Server URL:** `http://localhost:8000`

---

## 🚀 Quick Setup in Open-WebUI

### Method 1: Web Interface (Easiest)

1. Open Open-WebUI in your browser
2. Go to **Settings → Audio**
3. Configure STT:
   - **STT Engine**: `OpenAI`
   - **API Base URL**: `http://localhost:8000/v1`
   - **API Key**: Leave empty (not needed)
   - **Model**: `whisper-1`
4. Click **Save**

### Method 2: Automatic Script

```bash
python update_webui_config.py
```

### Method 3: Environment Variables

```bash
export STT_OPENAI_API_BASE_URL="http://localhost:8000/v1"
export STT_ENGINE="openai"
export WHISPER_MODEL="whisper-1"
```

---

## 📡 Available Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/audio/transcriptions` | POST | Transcribe audio (OpenAI compatible) |
| `/audio/transcriptions` | POST | Transcribe audio (alternate path) |
| `/v1/audio/translations` | POST | Translate audio to English |
| `/audio/translations` | POST | Translate (alternate path) |
| `/config` | GET | Get current server configuration |
| `/config/update` | POST | Update server configuration |
| `/v1/models` | GET | List available models |
| `/health` | GET | Health check |
| `/` | GET | API information |

---

## 🧪 Testing

### Quick Health Check
```bash
curl http://localhost:8000/health
```

### Run Full Test Suite
```bash
python test_openwebui_compatibility.py
```

### Test Configuration
```bash
# Get current config
curl http://localhost:8000/config | jq

# Update config
curl -X POST http://localhost:8000/config/update \
  -H "Content-Type: application/json" \
  -d '{"language": "en", "vad_filter": true}' | jq
```

### Test Transcription
```bash
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -F "file=@test_audio_short.mp3" \
  -F "model=whisper-1" | jq
```

---

## ⚙️ Configuration Options

### Supported Languages
When updating config or sending requests:
- `null` or `"auto"` - Auto-detect language
- `"en"` - English
- `"es"` - Spanish
- `"fr"` - French
- `"de"` - German
- `"zh"` - Chinese
- And many more...

### Dynamic Configuration
Update server settings without restart:
```bash
curl -X POST http://localhost:8000/config/update \
  -H "Content-Type: application/json" \
  -d '{
    "language": "auto",
    "vad_filter": false,
    "model": "whisper-1"
  }'
```

---

## 🔄 Server Management

### Check Server Status
```bash
sudo systemctl status whisper-server
```

### Restart Server
```bash
sudo systemctl restart whisper-server
```

### View Logs
```bash
sudo journalctl -u whisper-server -f
```

---

## 📝 Example Usage from Open-WebUI

Once configured, Open-WebUI will:

1. **Click microphone** 🎤 in chat
2. **Record audio**
3. **Send to** `http://localhost:8000/v1/audio/transcriptions`
4. **Receive transcription** in real-time
5. **Insert text** into chat

The server automatically:
- ✅ Detects language (if not specified)
- ✅ Handles multiple audio formats
- ✅ Provides fast inference (6-10x realtime)
- ✅ Returns accurate transcriptions

---

## 🎯 Current Server Config

To view current configuration:
```bash
curl -s http://localhost:8000/ | jq .current_config
```

Example output:
```json
{
  "engine": "whisper",
  "model": "whisper-1",
  "language": "auto-detect"
}
```

---

## 📚 Full Documentation

For complete details, see:
- **OPEN_WEBUI_INTEGRATION.md** - Complete integration guide
- **README.md** - Server documentation

---

## ✨ Features

- ✅ **OpenAI API Compatible** - Drop-in replacement
- ✅ **Open-WebUI Compatible** - All required endpoints
- ✅ **Auto Language Detection** - No manual selection needed
- ✅ **Dynamic Configuration** - Update settings on-the-fly
- ✅ **Multiple Paths** - Both `/v1` and alternate routes
- ✅ **Fast Inference** - Optimized for low latency
- ✅ **No API Key Required** - Works out of the box

---

**Your server is ready to use with Open-WebUI!** 🚀
