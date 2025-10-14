# Quick Start Guide

## 🚀 Installation & Setup

```bash
# Clone repository
git clone https://github.com/groxaxo/Whisper-Fast-Cpu-OpenVino.git
cd Whisper-Fast-Cpu-OpenVino

# Create conda environment
conda env create -f environment.yml
conda activate ov-whisper

# Download model (auto or manual)
python setup_model.py --auto  # Auto-downloads if HF CLI available
# OR
python setup_model.py         # Interactive selection

# Start server (choose one)
./start_server.sh         # Gradio web interface
./start_openai_api.sh     # OpenAI-compatible API (for Open WebUI)
```

**Model Options:**
- **INT8-Turbo** (1.5GB) - Balanced speed/accuracy ⭐ Recommended for modern CPUs
- **INT8-Lite** (1.5GB) - Optimized for older/weaker CPUs (<4 cores)
- **INT4** (800MB) - Maximum speed, smallest size

Run `python compare_models.py` for detailed comparison.

## 🌐 Access

**Gradio Interface:**
- **Local:** http://localhost:7860
- **Remote:** `ngrok http 7860`

**OpenAI API:** ⭐ **NEW!**
- **Base URL:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **Compatible with:** Open WebUI, Continue, and other OpenAI API clients

## 🎤 Streaming Mode (NEW!)
**Accumulates text as you speak - nothing gets deleted!**

1. Use **🎤 Streaming Audio (Microphone)** input
2. Click microphone button
3. Start speaking
4. Watch transcript grow in real-time
5. Click "Clear Transcript" when done

**Perfect for:**
- Live conversations
- Dictation
- Real-time meetings
- Continuous translation

## 🔌 OpenAI API Mode ⭐ **NEW!**
**Compatible with Open WebUI and other OpenAI API clients!**

### Quick Test:
```bash
# Start the API server
./start_openai_api.sh

# Test with curl
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@sample.wav" \
  -F "model=whisper-1" \
  -F "language=en"
```

### Open WebUI Setup:
1. Open Open WebUI settings
2. Go to **Audio → STT Settings**
3. Set **API Base URL:** `http://localhost:8000`
4. Set **API Key:** any value (not validated)
5. Set **Model:** `whisper-1`
6. Save and test!

### Python Example:
```python
import requests

with open('audio.mp3', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/v1/audio/transcriptions',
        files={'file': f},
        data={'model': 'whisper-1', 'language': 'en'}
    )
    print(response.json()['text'])
```

## 📁 Upload Mode (Gradio Interface)
**Now with precision trimming controls!**

1. Use **📁 Upload Audio (File)** input
2. Click upload button and select your audio file
3. **Trim audio** using manual time controls (optional)
   - **Trim Start:** Enter start time in seconds (e.g., 5.5)
   - **Trim End:** Enter end time (or leave empty for end)
   - Precision: 0.01 seconds (hundredths)
   - Preview with play/pause controls
4. Click **🎯 Process Uploaded Audio** button
5. Get transcription of selected portion

**Perfect for:**
- Pre-recorded audio with precise trimming
- Podcasts (skip intro/outro with exact times)
- Interviews (extract sections by timestamp)
- Long audio files (process any length)

## 🎯 Key Features

### Both Modes Support:
- ✅ **Auto language detection**
- ✅ **Transcribe** (keep original language)
- ✅ **Translate** (convert to English)
- ✅ **Timestamps** (optional)

### What's Fixed:
| Issue | Status |
|-------|--------|
| Streaming overwrites text | ✅ Fixed - accumulates now |
| Upload not working | ✅ Fixed - has dedicated handler |
| Memory corruption crashes | ✅ Fixed - rate limiting + deep copy + GC |
| WebSocket protocol crashes | ✅ Fixed - proper memory management |
| Translation errors | ✅ Fixed - works reliably |

## 🔧 Controls

- **Language:** Select source or use "auto"
- **Task:** Choose "transcribe" or "translate"
- **Timestamps:** Toggle on/off
- **Clear Transcript:** Reset accumulated text

## 📊 Monitoring

```bash
# Check server status
lsof -i :7860

# View logs
tail -f server.log

# Stop server
pkill -f serve_whisper.py
```

## 💡 Tips

1. **Streaming:** Speak clearly, pause between thoughts (processes every 3 seconds)
2. **Upload:** Supports WAV, MP3, and other formats - use manual trim controls
3. **Trimming:** Enter exact start/end times with 0.01s precision - no restrictions!
4. **Time format:** Just enter seconds (e.g., 5.5 = 5.5 seconds, 30.25 = 30.25 seconds)
5. **Process button:** Must click to start transcription (won't auto-process)
6. **Translation:** Works in real-time for both modes
7. **Clear:** Use clear button between topics/sessions
8. **Stability:** Server now handles long sessions without crashing

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| Port in use | Run `pkill -f serve_whisper` first |
| No transcription | Check audio levels in browser |
| Text disappears | Fixed! Uses state accumulation |
| Upload broken | Fixed! Has dedicated handler |

## 📚 Documentation

- **Performance Benchmarks:** See `BENCHMARKS.md` ⭐ NEW!
- **Complete Guide:** See `README.md`
- **Environment:** See `environment.yml`

## 🚀 Performance

**Tested on Intel i7-12700KF (12th Gen) with 19 threads:**
- **INT8-Turbo:** 2.12x real-time ⭐ (fastest, recommended)
- **INT8-Lite:** 1.78x real-time (good for long files)
- **INT4:** 1.10x real-time (doesn't scale well)

**Thread optimization matters!** INT8-Turbo gained 28% speed with 19 vs 8 threads.

Run `python benchmark_simple.py --threads 19` to test on your system.
