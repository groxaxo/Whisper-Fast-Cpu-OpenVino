# Model Download Instructions

The OpenVINO Whisper model files are not included in this repository due to their size.

## Automatic Download

The model will be automatically downloaded on first run from HuggingFace Hub:
- **Model:** [OpenVINO/whisper-large-v3-int8-ov](https://huggingface.co/OpenVINO/whisper-large-v3-int8-ov)

## Manual Download

If you prefer to download manually:

```bash
pip install huggingface_hub

python -c "
import huggingface_hub as hf_hub
hf_hub.snapshot_download('OpenVINO/whisper-large-v3-int8-ov', local_dir='model')
"
```

## Model Files

After download, this directory should contain:
- `openvino_encoder_model.xml`
- `openvino_encoder_model.bin`
- `openvino_decoder_model.xml`
- `openvino_decoder_model.bin`
- `openvino_decoder_with_past_model.xml`
- `openvino_decoder_with_past_model.bin`
- `tokenizer.json`
- `preprocessor_config.json`
- `generation_config.json`

## Credits

- **Original Model:** OpenAI Whisper Large V3
- **Optimization:** Intel OpenVINO (INT8 quantization)
- **Source:** https://huggingface.co/OpenVINO/whisper-large-v3-int8-ov
