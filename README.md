```
╦ ╦┬ ┬┬┌─┐┌─┐┌─┐┬─┐  ╔═╗┌─┐┌─┐┌┬┐  ╔═╗╔═╗╦ ╦  ╔═╗┌─┐┌─┐┌┐┌╦  ╦╦╔╗╗╔═╗
║║║├─┤│└─┐├─┘├┤ ├┬┘  ╠╣ ├─┤└─┐ │───║  ╠═╝║ ║  ║ ║├─┘├┤ │││╚╗╔╝║║║║║ ║
╚╩╝┴ ┴┴└─┘┴  └─┘┴└─  ╚  ┴ ┴└─┘ ┴   ╚═╝╩  ╚═╝  ╚═╝┴  └─┘┘└┘ ╚╝ ╩╝╚╝╚═╝
```

# Whisper-Fast-CPU-OpenVINO

⚡ **Fast, local speech-to-text** using OpenVINO on CPU with OpenAI-compatible API and Open-WebUI integration.

## 🌟 Features

- 🚀 **Fast CPU Inference** - Optimized for Intel CPUs using OpenVINO (6-10x realtime)
- 🎯 **OpenAI API Compatible** - Drop-in replacement for OpenAI Whisper API
- 🌐 **Open-WebUI Integration** - Full STT support with dynamic configuration
- 🎤 **Global Dictation Client** - System-wide voice input with `Ctrl+Alt+Space`
- 🧠 **Multiple Models** - INT8/INT4 quantized models for speed/quality balance
- 🔧 **Auto-Detection** - Automatic language detection or manual selection
- ⚙️ **Dynamic Config** - Change settings on-the-fly without restart

## 📊 Performance

Tested on **Intel Core i5-1240P** (12 physical cores):
- **Model**: INT8 Turbo (~1GB)
- **Speed**: 6-10x realtime (transcribe 30s audio in ~3-5s)
- **Latency**: <1 second for typical queries
- **Memory**: ~500-800 MB

## 🚀 Quick Start

### Automated Setup (Recommended)

**One command to install everything:**
```bash
git clone https://github.com/groxaxo/Whisper-Fast-Cpu-OpenVino.git
cd Whisper-Fast-Cpu-OpenVino
./setup.sh
```

The script will:
- ✅ Install system dependencies
- ✅ Create conda environment
- ✅ Install Python packages
- ✅ Download INT8 Turbo model
- ✅ Test the installation
- ✅ Optionally setup auto-start service

### Manual Setup

```bash
git clone https://github.com/groxaxo/Whisper-Fast-Cpu-OpenVino.git
cd Whisper-Fast-Cpu-OpenVino

# Create conda environment
conda env create -f environment.yml
conda activate ov-whisper

# Download INT8 Turbo model (recommended)
python setup_model.py --model int8-turbo
```

### Start Server
```bash
./start_server.sh
```

Server runs on `http://localhost:8000` 🎉

### Test It Works
```bash
# Check health
curl http://localhost:8000/health

# Run full test suite
python test_openwebui_compatibility.py
```

## 🎤 Global Dictation

### Start Dictation Client
```bash
python dictation_client.py
```

### Usage
1. Click into any text field (Terminal, Browser, Editor)
2. Press **`Ctrl` + `Alt` + `Space`**
3. Speak your text (overlay shows status)
4. Press **`Ctrl` + `Alt` + `Space`** again to stop
5. Text appears automatically! ✨

## 🌐 Open-WebUI Integration

This server is **fully compatible** with [Open-WebUI](https://github.com/open-webui/open-webui)!

### Quick Setup

**Option 1: Web Interface (Easiest)**
1. Open Open-WebUI → **Settings → Audio**
2. Set **STT Engine**: `OpenAI`
3. Set **API Base URL**: `http://localhost:8000/v1`
4. Set **Model**: `whisper-1`
5. Click **Save**

**Option 2: Automatic Script**
```bash
python update_webui_config.py
```

**Option 3: Environment Variables**
```bash
export STT_OPENAI_API_BASE_URL="http://localhost:8000/v1"
export STT_ENGINE="openai"
export WHISPER_MODEL="whisper-1"
```

### Features
- ✅ Auto language detection
- ✅ Manual language selection
- ✅ Dynamic configuration via `/config/update`
- ✅ Real-time transcription
- ✅ Multiple audio formats (MP3, WAV, M4A, FLAC, OGG, WebM)

## 📡 API Endpoints

### OpenAI Compatible

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/audio/transcriptions` | POST | Transcribe audio (OpenAI compatible) |
| `/v1/audio/translations` | POST | Translate audio to English |
| `/v1/models` | GET | List available models |

### Open-WebUI Compatible

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/audio/transcriptions` | POST | Transcribe audio (alternate path) |
| `/audio/translations` | POST | Translate audio (alternate path) |
| `/config` | GET | Get server configuration |
| `/config/update` | POST | Update server configuration |
| `/health` | GET | Health check |

### Example Usage

**Transcribe Audio:**
```bash
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -F "file=@audio.mp3" \
  -F "model=whisper-1" \
  -F "language=en"
```

**Auto-Detect Language:**
```bash
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -F "file=@audio.mp3" \
  -F "model=whisper-1"
```

**Update Configuration:**
```bash
curl -X POST "http://localhost:8000/config/update" \
  -H "Content-Type: application/json" \
  -d '{"language": "es", "vad_filter": true}'
```

**Get Current Config:**
```bash
curl http://localhost:8000/config
```

## 🛠️ Installation (Manual)

### Prerequisites

**Ubuntu/Linux:**
```bash
sudo apt update
sudo apt install -y python3-tk portaudio19-dev
```

**Intel GPU Support (Optional):**
```bash
sudo apt install intel-opencl-icd intel-level-zero-gpu
```

### Python Environment
```bash
# Create environment
conda create -n ov-whisper python=3.11 -y
conda activate ov-whisper

# Install dependencies
pip install openvino-genai \
  faster-whisper \
  fastapi \
  uvicorn \
  soundfile \
  librosa \
  pynput \
  sounddevice \
  scipy \
  requests
```

### Download Models

**Interactive:**
```bash
python setup_model.py
```

**Automated:**
```bash
# INT8 Turbo (recommended, ~1GB)
python setup_model.py --model int8-turbo

# INT4 (faster, ~600MB)
python setup_model.py --model int4
```

## ⚙️ Server Configuration

### Command-Line Options

```bash
python serve_openai_api.py \
  --model-dir model_int8_turbo \  # Model directory
  --device CPU \                   # CPU, GPU, or AUTO
  --host 0.0.0.0 \                # Bind to all interfaces
  --port 8000 \                    # API port
  --threads 12 \                   # Number of CPU threads
  --streams AUTO \                 # Parallel inference streams
  --hint LATENCY                   # LATENCY or THROUGHPUT
```

### Optimization for Intel 12th Gen+

The `start_server.sh` script is pre-configured for optimal performance:

| Setting | Value | Reason |
|---------|-------|--------|
| **Model** | INT8 Turbo | Best speed/quality balance |
| **Threads** | 12 | All P-core threads (i5-1240P) |
| **CPU Affinity** | 0-11 | P-cores only (avoid E-cores) |
| **Streams** | AUTO | Automatic optimization |
| **Hint** | LATENCY | Low-latency response |

### Auto-Start on Boot

**Create systemd service:**
```bash
sudo nano /etc/systemd/system/whisper-server.service
```

**Service file (example included: `whisper-server.service`):**
```ini
[Unit]
Description=OpenVINO Whisper API Server
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/path/to/Whisper-Fast-Cpu-OpenVino
ExecStart=/path/to/conda/envs/ov-whisper/bin/python serve_openai_api.py --model-dir model_int8_turbo --threads 12
Restart=always

[Install]
WantedBy=multi-user.target
```

**Enable and start:**
```bash
sudo systemctl enable whisper-server
sudo systemctl start whisper-server
```

## 📁 Project Structure

```
Whisper-Fast-Cpu-OpenVino/
├── serve_openai_api.py              # Main API server
├── dictation_client.py               # Global dictation client
├── setup_model.py                    # Model downloader
├── start_server.sh                   # Optimized startup script
├── environment.yml                   # Conda environment
├── test_openwebui_compatibility.py  # Test suite
├── update_webui_config.py           # Auto-configure Open-WebUI
├── whisper-server.service           # systemd service template
├── OPEN_WEBUI_INTEGRATION.md        # Detailed integration guide
└── README.md                         # This file
```

## 🧪 Testing

### Health Check
```bash
curl http://localhost:8000/health
# Expected: {"status":"ok","model_loaded":true}
```

### Full Test Suite
```bash
python test_openwebui_compatibility.py
```

Tests all endpoints:
- ✅ Health check
- ✅ Configuration GET/UPDATE
- ✅ Transcription (both paths)
- ✅ Translation
- ✅ Model listing

### Manual Test
```bash
# Use included test audio
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -F "file=@test_audio_short.mp3" \
  -F "model=whisper-1" | jq
```

## 🎯 Use Cases

### 1. Open-WebUI Voice Input
- Configure Open-WebUI to use this server
- Click microphone in chat
- Speak and get instant transcription

### 2. System-Wide Dictation
- Run `dictation_client.py`
- Use `Ctrl+Alt+Space` in any application
- Universal voice typing

### 3. OpenAI API Replacement
- Point any OpenAI Whisper client to `http://localhost:8000/v1`
- No API key needed
- 100% local and private

### 4. Custom Applications
- Build your own voice apps
- Use FastAPI endpoints
- Integrate with existing systems

## 🔧 Troubleshooting

### Server Won't Start
```bash
# Check if port is in use
sudo lsof -i :8000

# View logs
tail -f api_server.log

# Or if using systemd
sudo journalctl -u whisper-server -f
```

### Model Not Found
```bash
# Download model again
python setup_model.py --model int8-turbo

# Verify model exists
ls -la model_int8_turbo/
```

### Slow Performance
```bash
# Check CPU usage
htop

# Verify using correct model
curl http://localhost:8000/config | jq

# Restart with optimal settings
./start_server.sh
```

### Open-WebUI Connection Error
```bash
# Test server is accessible
curl http://localhost:8000/health

# Check Open-WebUI configuration
curl http://localhost:8000/config

# Update Open-WebUI settings
python update_webui_config.py
```

### Dictation Hotkey Not Working
- **Wayland**: Switch to Xorg (Wayland blocks global hotkeys)
- **Permissions**: Check if `pynput` has accessibility permissions
- **Conflict**: Another app might be using `Ctrl+Alt+Space`

## 📊 Available Models

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| **INT8 Turbo** (default) | ~1GB | 6-10x realtime | High | Recommended for most users |
| INT4 | ~600MB | Fastest | Good | Low-power devices |

## 🌍 Supported Languages

Auto-detection works for 99+ languages. Manual selection supports:

- English (`en`)
- Spanish (`es`)
- French (`fr`)
- German (`de`)
- Chinese (`zh`)
- Japanese (`ja`)
- Korean (`ko`)
- Russian (`ru`)
- And many more...

## 📖 Documentation

**Main Documentation:**
- **[README.md](README.md)** - Main documentation, quick start, and features
- **[OPEN_WEBUI_INTEGRATION.md](OPEN_WEBUI_INTEGRATION.md)** - Complete Open-WebUI integration guide
- **API Docs**: `http://localhost:8000/docs` (interactive documentation when server is running)

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenVINO](https://github.com/openvinotoolkit/openvino) - Fast CPU/GPU inference
- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition model
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - Optimized Whisper implementation
- [Open-WebUI](https://github.com/open-webui/open-webui) - Beautiful web interface

## 🔗 Links

- **GitHub**: https://github.com/groxaxo/Whisper-Fast-Cpu-OpenVino
- **Issues**: https://github.com/groxaxo/Whisper-Fast-Cpu-OpenVino/issues
- **OpenVINO Docs**: https://docs.openvino.ai/
- **Open-WebUI**: https://docs.openwebui.com/

---

**Made with ❤️ for fast, private, local speech recognition**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-green.svg)](https://www.python.org/)
[![OpenVINO](https://img.shields.io/badge/OpenVINO-2024-blue.svg)](https://www.openvino.ai/)
