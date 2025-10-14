# Whisper-Fast-CPU-OpenVINO

**Fast CPU-based Whisper transcription server using OpenVINO optimizations**

[![OpenVINO](https://img.shields.io/badge/OpenVINO-2025.2.0-blue)](https://github.com/openvinotoolkit/openvino)
[![Whisper](https://img.shields.io/badge/Whisper-large--v3-green)](https://github.com/openai/whisper)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange)](LICENSE)

## Overview

Real-time speech transcription and translation server powered by OpenAI's Whisper model, optimized for CPU inference using Intel's OpenVINO toolkit. Features a user-friendly Gradio interface with streaming support and precision audio trimming.

### Key Features

- 🔌 **OpenAI API Compatible** - Drop-in replacement for OpenAI Whisper API (Open WebUI ready!)
- 🎤 **Real-time streaming** - Live microphone transcription with text accumulation
- 📁 **File upload** - Process pre-recorded audio with precision trimming (0.01s accuracy)
- 🌍 **100+ languages** - Auto-detection or manual selection
- 🔄 **Translation** - Translate any language to English in real-time
- ⚡ **CPU optimized** - Fast inference using OpenVINO INT8/INT4 quantization
- 🎯 **Precision trimming** - Extract exact audio segments by timestamp
- 💾 **Memory efficient** - Stable for long sessions with automatic garbage collection

## Credits

- **Original Whisper Model:** [OpenAI Whisper](https://github.com/openai/whisper) - State-of-the-art speech recognition
- **OpenVINO Optimization:** [Intel OpenVINO](https://github.com/openvinotoolkit/openvino) - INT8/INT4 quantized models for fast CPU inference
- **Model Weights:** 
  - [OpenVINO/whisper-large-v3-turbo-int8-ov](https://huggingface.co/OpenVINO/whisper-large-v3-turbo-int8-ov) - Recommended
  - [bweng/whisper-large-v3-turbo-int8-ov](https://huggingface.co/bweng/whisper-large-v3-turbo-int8-ov) - Optimized for weaker hardware
  - [OpenVINO/whisper-large-v3-int4-ov](https://huggingface.co/OpenVINO/whisper-large-v3-int4-ov) - Maximum speed

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/groxaxo/Whisper-Fast-Cpu-OpenVino.git
cd Whisper-Fast-Cpu-OpenVino

# Create conda environment
conda env create -f environment.yml
conda activate ov-whisper
```

### 2. Download Model

**Automatic download** (if HuggingFace CLI available):
```bash
python setup_model.py --auto  # Auto-downloads recommended model
```

**Or choose manually:**

```bash
# Interactive setup
python setup_model.py

# Direct download
python setup_model.py --model int8-turbo  # Recommended: balanced performance
python setup_model.py --model int8-lite   # For weaker/older CPUs
python setup_model.py --model int4        # Maximum speed, smallest size
```

**Model Comparison:**

| Model | Size | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| **INT8-Turbo** ⭐ | 1.5 GB | 1.5-2.0x RT | 100% | Modern CPUs (4+ cores) |
| **INT8-Lite** | 1.5 GB | 1.2-1.8x RT | 100% | Older/weaker CPUs (<4 cores) |
| **INT4** | 800 MB | 2.0-3.0x RT | ~96% | Maximum speed, embedded systems |

*RT = Real-Time (e.g., 2.0x means 1 minute audio processed in 30 seconds)*

Run `python compare_models.py` for detailed comparison.

### 3. Start Server

**Option A: Gradio Interface (Web UI)**
```bash
# Easy start
./start_server.sh

# Or manual start
python serve_whisper.py --device CPU --port 7860
```

**Option B: OpenAI-Compatible API** ⭐ **NEW!**
```bash
# Start OpenAI-compatible API (for Open WebUI, etc.)
./start_openai_api.sh

# Or manual start
python serve_openai_api.py --device CPU --port 8000
```

### 4. Access Interface

**Gradio Interface:**
- **Local:** http://localhost:7860
- **Remote:** Use `ngrok http 7860` for external access

**OpenAI API:**
- **Base URL:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **Endpoints:** `/v1/audio/transcriptions`, `/v1/audio/translations`
- **Compatible with:** Open WebUI, Continue, and other OpenAI API clients

## Usage

### Streaming Mode (Microphone)

1. Click the 🎤 **Streaming Audio** microphone button
2. Start speaking
3. Watch transcript accumulate in real-time (updates every 3 seconds)
4. Click "Clear Transcript" to reset

**Perfect for:** Live conversations, meetings, dictation, real-time translation

### Upload Mode (File)

1. Click 📁 **Upload Audio** and select your file
2. **(Optional)** Enter trim times:
   - **Trim Start:** Start time in seconds (e.g., `5.5`)
   - **Trim End:** End time in seconds (leave empty for end of file)
3. Click **🎯 Process Uploaded Audio**
4. View transcription and status

**Perfect for:** Podcasts, interviews, pre-recorded content, extracting specific sections

### Trimming Examples

| Goal | Trim Start | Trim End | Result |
|------|-----------|----------|--------|
| Full file | 0 | (empty) | Entire audio |
| First 30s | 0 | 30 | 0:00 to 0:30 |
| Skip 10s intro | 10 | (empty) | 0:10 to end |
| Middle section | 45.5 | 120.75 | 75.25 seconds |
| Precise clip | 123.45 | 125.67 | Exactly 2.22s |

### OpenAI API Mode ⭐ **NEW!**

Use the OpenAI-compatible API with any client that supports the OpenAI Whisper API format.

**Python Example:**
```python
import requests

# Transcribe audio
with open('audio.mp3', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/v1/audio/transcriptions',
        files={'file': f},
        data={'model': 'whisper-1', 'language': 'en'}
    )
    print(response.json()['text'])
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "model=whisper-1" \
  -F "language=en"
```

**Open WebUI Integration:**
1. Open your Open WebUI settings
2. Go to Audio → STT Settings
3. Set API Base URL: `http://localhost:8000`
4. Set API Key: any value (not validated)
5. Model: `whisper-1`
6. Save and test!

**Perfect for:** Open WebUI, Continue, Cursor, any OpenAI-compatible client

## Supported Languages

---
language: 
- en
- zh
- de
- es
- ru
- ko
- fr
- ja
- pt
- tr
- pl
- ca
- nl
- ar
- sv
- it
- id
- hi
- fi
- vi
- he
- uk
- el
- ms
- cs
- ro
- da
- hu
- ta
- no
- th
- ur
- hr
- bg
- lt
- la
- mi
- ml
- cy
- sk
- te
- fa
- lv
- bn
- sr
- az
- sl
- kn
- et
- mk
- br
- eu
- is
- hy
- ne
- mn
- bs
- kk
- sq
- sw
- gl
- mr
- pa
- si
- km
- sn
- yo
- so
- af
- oc
- ka
- be
- tg
- sd
- gu
- am
- yi
- lo
- uz
- fo
- ht
- ps
- tk
- nn
- mt
- sa
- lb
- my
- bo
- tl
- mg
- as
- tt
- haw
- ln
- ha
- ba
- jw
- su
tags:
- audio
- automatic-speech-recognition
- hf-asr-leaderboard
pipeline_tag: automatic-speech-recognition
license: apache-2.0
license_link: https://choosealicense.com/licenses/apache-2.0/
---

## Model Information

### Base Model
- **Model:** [OpenAI Whisper Large V3](https://huggingface.co/openai/whisper-large-v3)
- **Optimization:** INT8 quantization via [Intel OpenVINO](https://github.com/openvinotoolkit/openvino)
- **Compression:** [NNCF](https://github.com/openvinotoolkit/nncf) weight compression
- **Format:** [OpenVINO IR](https://docs.openvino.ai/2025/documentation/openvino-ir-format.html)


### Quantization Details

- **Mode:** INT8_ASYM
- **Group Size:** 128
- **Method:** `nncf.compress_weights`

See [OpenVINO optimization guide](https://docs.openvino.ai/2025/openvino-workflow/model-optimization-guide/weight-compression.html) for details.


## Requirements

- **Python:** 3.11+
- **OpenVINO:** 2025.2.0+
- **Optimum Intel:** 1.23.0+ (optional)
- **OS:** Linux, macOS, Windows


## Configuration

### Gradio Interface Options

```bash
python serve_whisper.py [OPTIONS]

Options:
  --device DEVICE          Target device (default: CPU)
  --port PORT             Server port (default: 7860)
  --host HOST             Server host (default: 0.0.0.0)
  --threads THREADS       CPU threads (default: 8)
  --model-dir PATH        Model directory (default: ./model)
  --segment-seconds SEC   Audio segment length (default: 30.0)
  --streams STREAMS       Parallel inference streams (default: AUTO)
```

### OpenAI API Options

```bash
python serve_openai_api.py [OPTIONS]

Options:
  --device DEVICE          Target device (default: CPU)
  --port PORT             Server port (default: 8000)
  --host HOST             Server host (default: 0.0.0.0)
  --threads THREADS       CPU threads (default: 8)
  --model-dir PATH        Model directory (default: ./model)
  --streams STREAMS       Parallel inference streams (default: AUTO)
```

### Model Selection

You can run any of the three available models by specifying the `--model-dir`:

```bash
# INT8-Turbo (default, fastest on modern CPUs)
python serve_openai_api.py --model-dir model --threads 16

# INT8-Lite (optimized for older/weaker CPUs)
python serve_openai_api.py --model-dir model_int8_lite --threads 8

# INT4 (smallest size, good for embedded systems)
python serve_openai_api.py --model-dir model_int4 --threads 8
```

### Performance Tuning

- **Threads:** Adjust `--threads` based on your CPU (8-20 recommended)
- **Streams:** Use `AUTO` or set manually for parallel processing
- **Streaming interval:** 3 seconds (optimized for stability in Gradio)
- **Memory:** Automatic garbage collection prevents buildup

## Architecture

### Gradio Interface (serve_whisper.py)

**Streaming Mode:**
```
Microphone → WebSocket → 3s chunks → OpenVINO Pipeline → Accumulate → Display
```

**Upload Mode:**
```
File Upload → Trim (optional) → Process Button → OpenVINO Pipeline → Display
```

### OpenAI API (serve_openai_api.py)

**Request Flow:**
```
HTTP POST → File Upload → Audio Processing → OpenVINO Pipeline → JSON Response
```

**Endpoints:**
- `/v1/audio/transcriptions` - Transcribe audio (same language)
- `/v1/audio/translations` - Translate audio to English
- `/v1/models` - List available models
- `/health` - Health check

### Key Components

- **Frontend:** Gradio web interface OR FastAPI REST API
- **Backend:** OpenVINO GenAI WhisperPipeline
- **Audio Processing:** 16kHz resampling, mono conversion, normalization
- **Memory Management:** Deep copy, rate limiting, garbage collection
- **API Compatibility:** OpenAI Whisper API format (Open WebUI ready)

## Testing

### Quick API Test

```bash
# Start the OpenAI API server
./start_openai_api.sh

# Test with sample audio
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@sample.wav" \
  -F "model=whisper-1" \
  -F "language=en"

# Check health
curl http://localhost:8000/health
```

### Benchmark Your System

```bash
# Test with optimal threads for your CPU
python benchmark_simple.py --model-dir model --threads 16

# Compare all available models
python benchmark_all_models.py

# Visual comparison
python compare_models.py
```

### Model Testing

Test each model individually:

```bash
# Test INT8-Turbo
python serve_openai_api.py --model-dir model --threads 16 &
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@sample.wav" -F "model=whisper-1"

# Test INT8-Lite
python serve_openai_api.py --model-dir model_int8_lite --threads 8 &
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@sample.wav" -F "model=whisper-1"

# Test INT4
python serve_openai_api.py --model-dir model_int4 --threads 8 &
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@sample.wav" -F "model=whisper-1"
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Port already in use | Run `pkill -f serve_whisper` or `pkill -f serve_openai_api` |
| Server crashes | Check `server.log` or `api_server.log` for errors |
| No transcription | Verify audio input levels and format (16kHz recommended) |
| Slow processing | Reduce `--threads` or try a different model |
| Memory issues | Server auto-manages with GC, try INT4 model for lower memory |
| API connection refused | Ensure server is running: `lsof -i :8000` |
| Model not found | Run `python setup_model.py` to download models |

## Performance

### Benchmark Results (Intel i7-12700KF, 12th Gen)

Real-world benchmark with 84 seconds of test audio across three files:

| Model | Location | Avg Speed | RTF | Processing Time | Memory | Accuracy |
|-------|----------|-----------|-----|-----------------|--------|----------|
| **INT8-Turbo** ⭐ | `model/` | **2.12x RT** | 0.49x | 33.8s / 84s | ~2.0 GB | 100% |
| **INT8-Lite** | `model_int8_lite/` | 1.78x RT | 0.66x | 35.5s / 84s | ~1.8 GB | 100% |
| **INT4** | `model_int4/` | 1.10x RT | 1.51x | 59.7s / 84s | ~1.5 GB | ~96% |

**RTF** = Real-Time Factor (lower is better)  
**Speed** = Processing speed multiplier (higher is better)

**Key Findings:**
- **INT8-Turbo** is the fastest with 19 threads: processes 1 minute in ~28 seconds
- **INT8-Lite** performs well on long files (2.84x on 70s audio)
- **INT4** has smallest memory footprint but doesn't scale well with high thread counts
- Both INT8 models maintain 100% accuracy (same as original Whisper)
- Thread scaling: INT8-Turbo gains 28% speed with 19 vs 8 threads
- Stable memory usage during processing

**Model Selection Guide:**

| Your Hardware | Recommended Model | Directory | Thread Count | Expected Performance |
|---------------|-------------------|-----------|--------------|----------------------|
| High-end CPU (8+ cores, 3+ GHz) | **INT8-Turbo** ⭐ | `model/` | 16-20 threads | 2.0-2.5x real-time |
| Mid-range CPU (4-6 cores, 2.5+ GHz) | **INT8-Lite** | `model_int8_lite/` | 8-12 threads | 1.5-2.0x real-time |
| Older CPU (<4 cores) | **INT8-Lite** | `model_int8_lite/` | 4-8 threads | 1.2-1.5x real-time |
| Limited memory/storage | **INT4** | `model_int4/` | 8 threads | 1.0-1.5x real-time |

**Thread Scaling Analysis:**
- **INT8-Turbo:** 1.66x (8 threads) → **2.12x (19 threads)** ✅ Benefits from more threads
- **INT8-Lite:** 1.85x (8 threads) → 1.78x (19 threads) ⚠️ Slight regression with high threads
- **INT4:** Slower with high thread counts due to memory bandwidth limits

See [BENCHMARKS.md](BENCHMARKS.md) for detailed analysis and [BENCHMARK_COMPARISON.md](BENCHMARK_COMPARISON.md) for full comparison.

**Run benchmarks on your system:**
```bash
# Test with optimal threads for your CPU
python benchmark_simple.py --model-dir model --threads 16

# Compare all available models
python benchmark_all_models.py

# Visual comparison chart
python compare_models.py
```

## Limitations

See [OpenAI Whisper limitations](https://huggingface.co/openai/whisper-large-v3) for model-specific constraints.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

Apache 2.0 - See [LICENSE](LICENSE) for details

- **Original Whisper Model:** Apache 2.0 by OpenAI
- **OpenVINO Optimizations:** Apache 2.0 by Intel
- **This Server Implementation:** Apache 2.0

## Acknowledgments

- **OpenAI** - For the groundbreaking Whisper speech recognition model
- **Intel** - For OpenVINO toolkit and INT8 quantized model weights
- **OpenVINO Team** - For GenAI pipeline and optimization tools
- **Gradio** - For the excellent web interface framework

## Citation

If you use this project, please cite:

```bibtex
@misc{whisper-fast-cpu-openvino,
  title={Whisper-Fast-CPU-OpenVINO: Fast CPU-based Whisper Transcription Server},
  author={Your Name},
  year={2025},
  url={https://github.com/groxaxo/Whisper-Fast-CPU-OpenVINO}
}

@article{radford2022whisper,
  title={Robust Speech Recognition via Large-Scale Weak Supervision},
  author={Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  journal={arXiv preprint arXiv:2212.04356},
  year={2022}
}
```

## Support

- **Issues:** [GitHub Issues](https://github.com/groxaxo/Whisper-Fast-CPU-OpenVINO/issues)
- **Discussions:** [GitHub Discussions](https://github.com/groxaxo/Whisper-Fast-CPU-OpenVINO/discussions)

## Legal information

The original model is distributed under [apache-2.0](https://choosealicense.com/licenses/apache-2.0/) license. More details can be found in [original model card](https://huggingface.co/openai/whisper-large-v3).

## Disclaimer

Intel is committed to respecting human rights and avoiding causing or contributing to adverse impacts on human rights. See [Intel’s Global Human Rights Principles](https://www.intel.com/content/dam/www/central-libraries/us/en/documents/policy-human-rights.pdf). Intel’s products and software are intended only to be used in applications that do not cause or contribute to adverse impacts on human rights.

---

**Made with ❤️ using OpenAI Whisper and Intel OpenVINO**
