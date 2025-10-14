# Quick Start Guide

## 🚀 Installation & Setup

```bash
# Clone repository
git clone https://github.com/groxaxo/Whisper-Fast-Cpu-OpenVino.git
cd Whisper-Fast-Cpu-OpenVino

# Create conda environment
conda env create -f environment.yml
conda activate ov-whisper

# Start server
./start_server.sh
```

## 🌐 Access
- **Local:** http://localhost:7860
- **Remote:** `ngrok http 7860`

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

## 📁 Upload Mode (ENHANCED!)
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

**Tested on Intel i7-12700KF:**
- **1.85x real-time** average speed
- **3.30x real-time** on longer files
- 1 minute of audio in ~32 seconds
