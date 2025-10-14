# Whisper-Fast-CPU-OpenVINO

**Fast CPU-based Whisper transcription server using OpenVINO optimizations**

[![OpenVINO](https://img.shields.io/badge/OpenVINO-2025.2.0-blue)](https://github.com/openvinotoolkit/openvino)
[![Whisper](https://img.shields.io/badge/Whisper-large--v3-green)](https://github.com/openai/whisper)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange)](LICENSE)

## Overview

Real-time speech transcription and translation server powered by OpenAI's Whisper model, optimized for CPU inference using Intel's OpenVINO toolkit. Features a user-friendly Gradio interface with streaming support and precision audio trimming.

### Key Features

- 🎤 **Real-time streaming** - Live microphone transcription with text accumulation
- 📁 **File upload** - Process pre-recorded audio with precision trimming (0.01s accuracy)
- 🌍 **100+ languages** - Auto-detection or manual selection
- 🔄 **Translation** - Translate any language to English in real-time
- ⚡ **CPU optimized** - Fast inference using OpenVINO INT8 quantization
- 🎯 **Precision trimming** - Extract exact audio segments by timestamp
- 💾 **Memory efficient** - Stable for long sessions with automatic garbage collection

## Credits

- **Original Whisper Model:** [OpenAI Whisper](https://github.com/openai/whisper) - State-of-the-art speech recognition
- **OpenVINO Optimization:** [Intel OpenVINO](https://github.com/openvinotoolkit/openvino) - INT8 quantized model for fast CPU inference
- **Model Weights:** [OpenVINO/whisper-large-v3-int8-ov](https://huggingface.co/OpenVINO/whisper-large-v3-int8-ov)

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/groxaxo/Whisper-Fast-CPU-OpenVINO.git
cd Whisper-Fast-CPU-OpenVINO

# Create conda environment
conda env create -f environment.yml
conda activate ov-whisper
```

### 2. Start Server

```bash
# Easy start
./start_server.sh

# Or manual start
python serve_whisper.py --device CPU --port 7860
```

### 3. Access Interface

- **Local:** http://localhost:7860
- **Remote:** Use `ngrok http 7860` for external access

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

### Command Line Options

```bash
python serve_whisper.py [OPTIONS]

Options:
  --device DEVICE          Target device (default: CPU)
  --port PORT             Server port (default: 7860)
  --host HOST             Server host (default: 0.0.0.0)
  --threads THREADS       CPU threads (default: 8)
  --model-dir PATH        Model directory (default: ./model)
  --segment-seconds SEC   Audio segment length (default: 30.0)
```

### Performance Tuning

- **Threads:** Adjust `--threads` based on your CPU (default: 8)
- **Streaming interval:** 3 seconds (optimized for stability)
- **Memory:** Automatic garbage collection prevents buildup

## Architecture

### Streaming Mode
```
Microphone → WebSocket → 3s chunks → OpenVINO Pipeline → Accumulate → Display
```

### Upload Mode  
```
File Upload → Trim (optional) → Process Button → OpenVINO Pipeline → Display
```

### Key Components

- **Frontend:** Gradio web interface
- **Backend:** OpenVINO GenAI WhisperPipeline
- **Audio Processing:** 16kHz resampling, numpy arrays
- **Memory Management:** Deep copy, rate limiting, garbage collection

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Port already in use | Run `pkill -f serve_whisper` first |
| Server crashes | Check `server.log` for errors |
| No transcription | Verify audio input levels |
| Slow processing | Reduce `--threads` or use shorter audio |
| Memory issues | Server auto-manages with GC |

## Performance

- **CPU Mode:** ~1-2 seconds per second of audio (8 threads)
- **Memory:** ~1-2GB RAM for model + processing
- **Streaming:** 3-second intervals for stability
- **Accuracy:** Same as original Whisper large-v3

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
