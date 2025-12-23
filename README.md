# Whisper-Fast-Cpu-OpenVINO + Global Dictation

High-performance local speech recognition using **OpenVINO GenAI** and **Hugging Face** models.
Now featuring a **Global Dictation Client** with overlay UI for seamless voice input on Linux/Ubuntu.

## Features
-   🚀 **Fast CPU Inference**: Optimized for Intel CPUs (Core i5/i7/i9) using OpenVINO.
-   🎯 **Global Hotkey**: Press `Ctrl+Alt+Space` anywhere to dictate.
-   🖥️ **Visual Overlay**: Real-time feedback overlay (Listening, Transcribing).
-   🧠 **OpenAI Compatible**: Exposes a standard `/v1/audio/transcriptions` API.
-   💻 **Optimization**: Tuned for hybrid architectures (e.g., Intel 12th Gen P-Cores).

## Performance

Tested on Intel Core i5-1240P:
-   **INT8 Turbo Model**: ~7x realtime (30s audio transcribed in ~4.3s)
-   **CPU Affinity**: Pinned to P-cores (0-7) for optimal performance
-   **Threads**: 8 (all P-core hyperthreads)

## Prerequisites (Ubuntu/Linux)

### 1. System Dependencies
```bash
sudo apt update
sudo apt install -y python3-tk portaudio19-dev
```

### 2. Conda Environment
```bash
conda create -n ov-whisper python=3.11 -y
conda activate ov-whisper
pip install -r requirements.txt
# Additional requirements for Overlay UI
pip install pynput sounddevice scipy requests
```

> **Note for Intel iGPU**: If you want to use the GPU (Iris Xe), install drivers:
> `sudo apt install intel-opencl-icd intel-level-zero-gpu`

## Optimization (Intel 12th Gen+)

This repository is pre-configured for **Latency** optimization on Intel Alder Lake (12th Gen) and newer hybrid CPUs:

| Setting | Value | Reason |
|---------|-------|--------|
| **Model** | INT8 Turbo | Best speed/quality balance |
| **Threads** | 8 | All P-core hyperthreads |
| **CPU Affinity** | 0-7 | P-cores only (avoids E-cores) |
| **Streams** | 1 | Low latency per request |
| **Hint** | LATENCY | Optimized for response time |

## Quick Start

### 1. Setup Model
Download the recommended INT8 Turbo model (~1GB):
```bash
python setup_model.py --model int8-turbo
```

Or run interactively:
```bash
python setup_model.py
```

### 2. Start Server
Run the optimized server script:
```bash
./start_server.sh
```
*Server runs on `http://localhost:8000`*

### 3. Run Dictation Client
Launch the overlay client:
```bash
python dictation_client.py
```

## How to Use Dictation
1.  Ensure both server and client are running.
2.  Click into any text field (Terminal, Browser, Editor).
3.  Press **`Ctrl` + `Alt` + `Space`**.
4.  Current Status will appear in center of screen:
    -   🔴 **Listening...**
5.  Speak your text.
6.  Press **`Ctrl` + `Alt` + `Space`** again to stop.
    -   ⏳ **Capturing...**
    -   ⚡ **Transcribing...**
7.  Text will be typed automatically.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/audio/transcriptions` | POST | Transcribe audio (OpenAI compatible) |
| `/v1/audio/translations` | POST | Translate audio to English |
| `/v1/models` | GET | List available models |
| `/health` | GET | Health check |

## Files Structure
-   `serve_openai_api.py`: The main FastAPI server.
-   `dictation_client.py`: The global dictation UI client.
-   `start_server.sh`: Optimized startup script with CPU pinning.
-   `setup_model.py`: Model downloader with INT8/INT4 options.

## Available Models

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| **INT8 Turbo** (default) | ~1GB | 5-7x realtime | High |
| INT4 | ~600MB | Fastest | Good |

## Troubleshooting
-   **Low audio levels**: Check your microphone input in system settings.
-   **Connection Refused**: Ensure `./start_server.sh` is running.
-   **Wayland**: Global hotkeys (`pynput`) might not work on Wayland. Switch to Xorg login if issues persist.
-   **Slow on E-cores**: The server is configured to avoid E-cores. If manually running, use `taskset -c 0-7`.
