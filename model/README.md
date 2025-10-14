---
license: apache-2.0
base_model:
- openai/whisper-large-v3-turbo
pipeline_tag: automatic-speech-recognition
tags:
- openvino
- whisper
- intel
---

Model creator: openai
Original model: https://huggingface.co/openai/whisper-large-v3-turbo

`optimum-cli export openvino --trust-remote-code --model openai/whisper-large-v3-turbo --weight-format int8 --disable-stateful whisper-large-v3-turbo`

## Compatibility
The provided OpenVINO™ IR model is compatible with:
* OpenVINO version 2024.5.0 and higher
* Optimum Intel 1.21.0 and higher
  
## Running Model Inference with [Optimum Intel](https://huggingface.co/docs/optimum/intel/index)
1. Install packages required for using [Optimum Intel](https://huggingface.co/docs/optimum/intel/index) integration with the OpenVINO backend:
```
pip install optimum[openvino]
```
2. Run model inference:
```
from transformers import AutoProcessor
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
model_id = "bweng/whisper-large-v3-turbo-int8-ov"
tokenizer = AutoProcessor.from_pretrained(model_id)
model = OVModelForSpeechSeq2Seq.from_pretrained(model_id)
dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation", trust_remote_code=True)
sample = dataset[0]
input_features = processor(
    sample["audio"]["array"],
    sampling_rate=sample["audio"]["sampling_rate"],
    return_tensors="pt",
).input_features
outputs = model.generate(input_features)
text = processor.batch_decode(outputs)[0]
print(text)
```

## Running Model Inference with [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai)
1. Install packages required for using OpenVINO GenAI.
```
pip install huggingface_hub
pip install -U --pre --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly openvino openvino-tokenizers openvino-genai
```
2. Download model from HuggingFace Hub
   
```
import huggingface_hub as hf_hub
model_id = "bweng/whisper-large-v3-turbo-int8"
model_path = "whisper-large-v3-turbo-int8"
hf_hub.snapshot_download(model_id, local_dir=model_path)
```
3. Run model inference:
```
import openvino_genai as ov_genai
import datasets
device = "NPU"
pipe = ov_genai.WhisperPipeline(model_path, device)
dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation", trust_remote_code=True)
sample = dataset[0]["audio]["array"]
print(pipe.generate(sample))
```
More GenAI usage examples can be found in OpenVINO GenAI library [docs](https://github.com/openvinotoolkit/openvino.genai/blob/master/src/README.md) and [samples](https://github.com/openvinotoolkit/openvino.genai?tab=readme-ov-file#openvino-genai-samples)
## Limitations
Check the original model card for [original model card](https://huggingface.co/openai/whisper-tiny) for limitations.