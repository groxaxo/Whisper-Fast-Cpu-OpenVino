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

## 🚀 Quick Start

### 1. Setup
```bash
# Clone and setup (installs dependencies & downloads model)
git clone https://github.com/groxaxo/Whisper-Fast-Cpu-OpenVino.git
cd Whisper-Fast-Cpu-OpenVino
./setup.sh
```

### 2. Start Services
**Start the API Server:**
```bash
./start_server.sh
```
*Server runs on http://localhost:8000*

**Start Dictation Client (Optional):**
```bash
./start_dictation.sh
```
*Press `Ctrl+Alt+Space` to dictate anywhere!*

## 🎤 How to Use Dictation

1. Run `./start_dictation.sh`
2. Click into any text field (browser, terminal, editor)
3. Press **`Ctrl+Alt+Space`** -> 🔴 Red "Listening..." overlay appears
4. Speak your text clearly
5. Press **`Ctrl+Alt+Space`** again -> ⏳ Processing -> ⚡ Text appears!

## 🌐 Open-WebUI Integration

This server is fully compatible with [Open-WebUI](https://github.com/open-webui/open-webui).

**Configuration:**
- **STT Engine**: `OpenAI`
- **API Base URL**: `http://localhost:8000/v1`
- **Model**: `whisper-1`

See [OPEN_WEBUI_INTEGRATION.md](OPEN_WEBUI_INTEGRATION.md) for a detailed guide.

## 📊 Performance (Intel i5-1240P)

- **Model**: INT8 Turbo (~1GB)
- **Speed**: 6-10x realtime
- **Latency**: < 1 second
- **Memory**: ~600-800 MB

## 🛠️ Manual Installation

If you prefer not to use `setup.sh`:

```bash
# 1. Create Conda environment
conda create -n ov-whisper python=3.11 -y
conda activate ov-whisper

# 2. Install dependencies
pip install openvino-genai faster-whisper fastapi uvicorn soundfile librosa pynput sounddevice scipy requests

# 3. System dependencies (Ubuntu)
sudo apt install python3-tk portaudio19-dev

# 4. Download Model
python setup_model.py --model int8-turbo
```

## 🔧 Troubleshooting

- **Server won't start?** Check if port 8000 is free: `lsof -i :8000`
- **Hotkey not working?** Ensure your user is in the `input` group: `sudo usermod -aG input $USER` (requires logout/login).
- **Audio issues?** Check microphone permissions and selection.

## 📝 License

Apache License 2.0
