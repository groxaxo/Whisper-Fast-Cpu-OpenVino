#!/usr/bin/env python3
"""OpenAI-compatible Whisper API server using OpenVINO GenAI pipeline."""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Header, Depends
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn
import openvino_genai as ov_genai

TARGET_SAMPLE_RATE = 16000
LOGGER = logging.getLogger("serve_openai_api")

# Initialize FastAPI app
app = FastAPI(
    title="OpenVINO Whisper API",
    description="OpenAI-compatible Whisper API using OpenVINO",
    version="1.0.0",
)

# Security (optional bearer token)
security = HTTPBearer(auto_error=False)

# Global variables for model
pipeline: Optional[ov_genai.WhisperPipeline] = None
language_tokens: Dict[str, str] = {}

# Server configuration state
class ServerConfig(BaseModel):
    """Server configuration state"""
    engine: str = "whisper"
    model: str = "whisper-1"
    model_dir: str = "model"
    device: str = "CPU"
    vad_filter: bool = False
    language: Optional[str] = None
    threads: int = 8
    streams: Union[str, int] = "AUTO"
    hint: str = "LATENCY"

# Global configuration instance
server_config = ServerConfig()


class TranscriptionResponse(BaseModel):
    """OpenAI transcription response format"""

    text: str
    task: Optional[str] = None
    language: Optional[str] = None
    duration: Optional[float] = None
    segments: Optional[list] = None


def parse_streams(value: str) -> Union[str, int]:
    if value is None:
        return "AUTO"
    normalized = value.strip().upper()
    if normalized == "AUTO":
        return "AUTO"
    try:
        streams = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--streams must be an integer or 'AUTO'"
        ) from exc
    if streams <= 0:
        raise argparse.ArgumentTypeError("--streams must be > 0")
    return streams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible Whisper API server using OpenVINO GenAI."
    )
    parser.add_argument(
        "--model-dir",
        default="model",
        help="Path to the exported OpenVINO Whisper model directory (default: ./model).",
    )
    parser.add_argument(
        "--device",
        default="CPU",
        choices=["CPU", "GPU", "AUTO"],
        help="Target device for OpenVINO execution (default: CPU). Use GPU for Intel Iris Xe graphics, AUTO for automatic selection.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host/IP for the API server bind (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the API server (default: 8000).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of CPU threads to dedicate to inference.",
    )
    parser.add_argument(
        "--streams",
        type=parse_streams,
        default="AUTO",
        help="Number of parallel inference streams (integer or AUTO).",
    )
    parser.add_argument(
        "--hint",
        default="LATENCY",
        choices=["LATENCY", "THROUGHPUT"],
        help="Performance hint for OpenVINO (LATENCY or THROUGHPUT, default: LATENCY).",
    )
    return parser.parse_args()


def load_language_tokens(model_dir: str) -> Dict[str, str]:
    """Load language tokens from model config"""
    config_path = os.path.join(model_dir, "generation_config.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Model configuration file not found: {config_path}\n"
            f"Run 'python setup_model.py' to download the model."
        )

    with open(config_path, "r", encoding="utf-8") as config_file:
        data = json.load(config_file)

    language_tokens: Dict[str, str] = {}
    for token in data.get("lang_to_id", {}):
        plain = token.strip("<|>")
        language_tokens[plain] = token

    return language_tokens



# Backend abstraction
class FasterWhisperPipeline:
    """Wrapper for faster-whisper to match OpenVINO API"""
    def __init__(self, model_id: str, device: str, compute_type: str = "float16"):
        from faster_whisper import WhisperModel
        LOGGER.info(f"Loading Faster-Whisper model: {model_id} on {device} ({compute_type})")
        self.model = WhisperModel(model_id, device=device, compute_type=compute_type)
    
    def generate(self, audio_data: list, **kwargs):
        """mimic generate method"""
        # faster-whisper expects a file path or numpy array
        # audio_data is a list of floats, convert to numpy
        audio_np = np.array(audio_data, dtype=np.float32)
        
        # Map kwargs
        # OpenVINO uses 'task', 'language', 'return_timestamps'
        # faster-whisper uses 'task', 'language', 'word_timestamps'
        
        fw_kwargs = {
            "task": kwargs.get("task", "transcribe"),
            "language": kwargs.get("language"),
            "beam_size": 5,
        }
        
        if kwargs.get("max_new_tokens"):
             # faster-whisper doesn't have max_new_tokens in transcribe, it streams
             # checking if we need to set anything specific. usually not needed for fw
             pass

        # faster-whisper transcribe returns (segments, info)
        segments, info = self.model.transcribe(
            audio_np, 
            word_timestamps=kwargs.get("return_timestamps", False),
            **fw_kwargs
        )
        
        # Materialize generator
        segments = list(segments)
        
        return FasterWhisperResult(segments, info)

class FasterWhisperResult:
    """Wrapper for result to match OpenVINO output"""
    def __init__(self, segments, info):
        self.segments = segments
        self.info = info
        # Combine text
        self.texts = ["".join([s.text for s in segments])]
        
        # Wrapper for chunks if timestamps requested
        self.chunks = []
        for s in segments:
            # check if segment has words
            if hasattr(s, 'words') and s.words:
                for w in s.words:
                     self.chunks.append(Chunk(w.start, w.end, w.word))
            else:
                 self.chunks.append(Chunk(s.start, s.end, s.text))

class Chunk:
    def __init__(self, start, end, text):
        self.start_ts = start
        self.end_ts = end
        self.text = text


def build_pipeline(
    config: ServerConfig
) -> Union[ov_genai.WhisperPipeline, FasterWhisperPipeline]:
    """Build pipeline based on configuration"""
    
    if config.engine == "faster-whisper":
        # Map device names
        device = "cuda" if config.device.upper() in ["GPU", "CUDA"] else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        return FasterWhisperPipeline(
            model_id=config.model, # e.g. "large-v3-turbo" or path
            device=device,
            compute_type=compute_type
        )
    
    else: # OpenVINO
        extra_kwargs: Dict[str, Union[str, int]] = {}
        
        # CPU-specific optimizations
        if config.device == "CPU":
            extra_kwargs["INFERENCE_NUM_THREADS"] = config.threads
            extra_kwargs["PERFORMANCE_HINT"] = config.hint
            if config.streams != "AUTO":
                extra_kwargs["NUM_STREAMS"] = config.streams
        
        LOGGER.info(
            "Loading OpenVINO pipeline (device=%s, threads=%d)",
            config.device,
            config.threads
        )
        return ov_genai.WhisperPipeline(config.model_dir, config.device, **extra_kwargs)




def load_audio_file(file_data: bytes) -> tuple[np.ndarray, int]:
    """Load audio from uploaded file bytes"""
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_file:
        tmp_file.write(file_data)
        tmp_path = tmp_file.name

    try:
        # Load audio using soundfile (more reliable than torchaudio)
        import soundfile as sf

        audio_array, sample_rate = sf.read(tmp_path)

        # Convert to mono if needed
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)

        # Resample if needed
        if sample_rate != TARGET_SAMPLE_RATE:
            import librosa

            audio_array = librosa.resample(
                audio_array, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE
            )
            sample_rate = TARGET_SAMPLE_RATE

        # Normalize
        max_abs = float(np.max(np.abs(audio_array))) if audio_array.size else 0.0
        if max_abs > 1.0:
            audio_array = audio_array / max_abs

        return audio_array.astype(np.float32), sample_rate

    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    timestamp_granularities: Optional[str] = Form(None),
):
    """
    OpenAI-compatible transcription endpoint.

    Transcribes audio to text in the same language.
    Compatible with Open WebUI and other OpenAI API clients.
    """
    try:
        # Read file data
        file_data = await file.read()

        if not file_data:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Load and process audio
        LOGGER.info(f"Processing file: {file.filename} ({len(file_data)} bytes)")
        audio_array, sample_rate = load_audio_file(file_data)

        duration = len(audio_array) / float(sample_rate)
        LOGGER.info(f"Audio duration: {duration:.2f}s, sample_rate: {sample_rate}")

        # Prepare generation kwargs
        kwargs = {"return_timestamps": timestamp_granularities == "segment"}
        kwargs["task"] = "transcribe"
        kwargs["max_new_tokens"] = 2048  # Allow very long transcriptions

        # Ignore client language parameter to allow auto-detection
        # This fixes issues where clients (like Open WebUI) send a default language (e.g., 'en')
        # which forces Whisper to translate instead of transcribe.
        if language:
            LOGGER.info(f"Ignoring client language request: '{language}', using auto-detection")
        
        # Always use auto-detection (passed as None/None to kwargs logic below)
        effective_language = None
        
        # Note: If you want to respect server_config.language, uncomment the line below instead:
        # effective_language = server_config.language 

        if effective_language and effective_language != "auto":
            whisper_token = language_tokens.get(effective_language)
            if whisper_token:
                kwargs["language"] = whisper_token
                LOGGER.info(f"Using server-configured language: {effective_language}")
            else:
                LOGGER.warning(f"Unknown language code: {effective_language}, using auto-detect")
        else:
            LOGGER.info("Using automatic language detection")

        # Run inference
        start_time = time.time()
        audio_data = audio_array.tolist()
        result = pipeline.generate(audio_data, **kwargs)
        inference_time = time.time() - start_time

        # Extract text
        text = result.texts[0].strip() if result.texts else ""

        LOGGER.info(
            f"Transcription completed: {len(text)} chars, "
            f"took {inference_time:.2f}s ({duration / inference_time:.2f}x realtime)"
        )

        # Build response
        response_data = {
            "text": text,
            "task": "transcribe",
            "language": language or "auto",
            "duration": duration,
        }

        # Add segments if requested
        if timestamp_granularities == "segment" and hasattr(result, "chunks"):
            segments = []
            for chunk in result.chunks:
                segments.append(
                    {
                        "start": float(chunk.start_ts),
                        "end": float(chunk.end_ts),
                        "text": chunk.text.strip(),
                    }
                )
            response_data["segments"] = segments

        # Return different formats
        if response_format == "text":
            return text
        elif response_format == "verbose_json":
            return JSONResponse(content=response_data)
        else:  # json (default)
            return JSONResponse(content={"text": text})

    except Exception as e:
        LOGGER.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/audio/translations")
async def translate_audio(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
):
    """
    OpenAI-compatible translation endpoint.

    Translates audio to English text.
    Compatible with Open WebUI and other OpenAI API clients.
    """
    try:
        # Read file data
        file_data = await file.read()

        if not file_data:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Load and process audio
        LOGGER.info(f"Translating file: {file.filename} ({len(file_data)} bytes)")
        audio_array, sample_rate = load_audio_file(file_data)

        duration = len(audio_array) / float(sample_rate)

        # Prepare generation kwargs
        kwargs = {"return_timestamps": False}
        kwargs["task"] = "translate"
        kwargs["max_new_tokens"] = 2048  # Allow very long translations

        # Run inference
        start_time = time.time()
        audio_data = audio_array.tolist()
        result = pipeline.generate(audio_data, **kwargs)
        inference_time = time.time() - start_time

        # Extract text
        text = result.texts[0].strip() if result.texts else ""

        LOGGER.info(
            f"Translation completed: {len(text)} chars, took {inference_time:.2f}s"
        )

        # Build response
        response_data = {
            "text": text,
            "task": "translate",
            "language": "en",
            "duration": duration,
        }

        # Return different formats
        if response_format == "text":
            return text
        elif response_format == "verbose_json":
            return JSONResponse(content=response_data)
        else:  # json (default)
            return JSONResponse(content={"text": text})

    except Exception as e:
        LOGGER.exception("Translation failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible endpoint)"""
    return {
        "object": "list",
        "data": [
            {
                "id": "whisper-1",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "openvino",
            }
        ],
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "model_loaded": pipeline is not None}


@app.get("/config")
async def get_config(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """
    Get current server configuration.
    
    Compatible with Open-WebUI configuration queries.
    """
    return {
        "engine": server_config.engine,
        "model": server_config.model,
        "vad_filter": server_config.vad_filter,
        "language": server_config.language,
        "device": server_config.device,
        "threads": server_config.threads,
        "streams": server_config.streams,
        "hint": server_config.hint,
    }


@app.post("/config/update")
async def update_config(
    config_update: dict,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """
    Update server configuration dynamically.
    
    Compatible with Open-WebUI configuration updates.
    Accepts JSON body with configuration parameters.
    """
    global server_config
    
    try:
        # Update configuration fields
        if "engine" in config_update:
            server_config.engine = config_update["engine"]
        if "model" in config_update:
            server_config.model = config_update["model"]
        if "vad_filter" in config_update:
            server_config.vad_filter = config_update["vad_filter"]
        if "language" in config_update:
            server_config.language = config_update["language"]
        if "device" in config_update:
            server_config.device = config_update["device"]
        if "threads" in config_update:
            server_config.threads = config_update["threads"]
        if "streams" in config_update:
            server_config.streams = config_update["streams"]
        if "hint" in config_update:
            server_config.hint = config_update["hint"]
        
        LOGGER.info(f"Configuration updated: {config_update}")
        
        return {
            "status": "success",
            "message": "Configurations updated successfully.",
            "config": {
                "engine": server_config.engine,
                "model": server_config.model,
                "vad_filter": server_config.vad_filter,
                "language": server_config.language,
            }
        }
    except Exception as e:
        LOGGER.exception("Configuration update failed")
        raise HTTPException(status_code=500, detail=f"Configuration update failed: {str(e)}")


# Alternate routes without /v1 prefix (for Open-WebUI compatibility)
@app.post("/audio/transcriptions")
async def transcribe_audio_alt(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    timestamp_granularities: Optional[str] = Form(None),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """Alternate transcription endpoint without /v1 prefix"""
    # Delegate to main endpoint
    return await transcribe_audio(file, model, language, prompt, response_format, temperature, timestamp_granularities)


@app.post("/audio/translations")
async def translate_audio_alt(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """Alternate translation endpoint without /v1 prefix"""
    # Delegate to main endpoint
    return await translate_audio(file, model, prompt, response_format, temperature)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with beautiful HTML landing page"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Whisper Fast CPU OpenVINO - Speech to Text API</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            line-height: 1.6;
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        
        .banner {
            font-family: 'Courier New', monospace;
            font-size: 10px;
            line-height: 1.2;
            white-space: pre;
            margin-bottom: 20px;
            color: #fff;
            text-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
        }
        
        h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .content {
            padding: 40px;
        }
        
        .status {
            background: #f0fdf4;
            border-left: 4px solid #10b981;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 8px;
        }
        
        .status h2 {
            color: #10b981;
            margin-bottom: 10px;
        }
        
        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        
        .feature-card {
            background: #f9fafb;
            padding: 25px;
            border-radius: 12px;
            border: 1px solid #e5e7eb;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
        }
        
        .feature-card h3 {
            color: #667eea;
            margin-bottom: 10px;
            font-size: 1.3em;
        }
        
        .endpoints {
            background: #1f2937;
            color: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 30px;
        }
        
        .endpoints h2 {
            color: #60a5fa;
            margin-bottom: 20px;
        }
        
        .endpoint {
            background: #374151;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
        }
        
        .endpoint-method {
            display: inline-block;
            background: #10b981;
            color: white;
            padding: 4px 12px;
            border-radius: 4px;
            font-weight: bold;
            margin-right: 10px;
            font-size: 0.9em;
        }
        
        .endpoint-method.post {
            background: #3b82f6;
        }
        
        .endpoint-path {
            color: #60a5fa;
        }
        
        .endpoint-desc {
            color: #9ca3af;
            font-size: 0.9em;
            margin-top: 5px;
        }
        
        .config-info {
            background: #fffbeb;
            border-left: 4px solid #f59e0b;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }
        
        .config-info h3 {
            color: #f59e0b;
            margin-bottom: 15px;
        }
        
        .config-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #fde68a;
        }
        
        .config-item:last-child {
            border-bottom: none;
        }
        
        .config-label {
            font-weight: 600;
            color: #92400e;
        }
        
        .config-value {
            color: #d97706;
            font-family: 'Courier New', monospace;
        }
        
        .quick-start {
            background: #f0f9ff;
            border-left: 4px solid #3b82f6;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }
        
        .quick-start h3 {
            color: #3b82f6;
            margin-bottom: 15px;
        }
        
        .code-block {
            background: #1f2937;
            color: #e5e7eb;
            padding: 15px;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            overflow-x: auto;
            margin-top: 10px;
        }
        
        .footer {
            background: #f9fafb;
            padding: 30px;
            text-align: center;
            color: #6b7280;
        }
        
        .footer a {
            color: #667eea;
            text-decoration: none;
        }
        
        .footer a:hover {
            text-decoration: underline;
        }
        
        @media (max-width: 768px) {
            .banner {
                font-size: 6px;
            }
            
            h1 {
                font-size: 1.8em;
            }
            
            .content {
                padding: 20px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="banner">╦ ╦┬ ┬┬┌─┐┌─┐┌─┐┬─┐  ╔═╗┌─┐┌─┐┌┬┐  ╔═╗╔═╗╦ ╦  ╔═╗┌─┐┌─┐┌┐┌╦  ╦╦╔╗╔╔═╗
║║║├─┤│└─┐├─┘├┤ ├┬┘  ╠╣ ├─┤└─┐ │───║  ╠═╝║ ║  ║ ║├─┘├┤ │││╚╗╔╝║║║║║ ║
╚╩╝┴ ┴┴└─┘┴  └─┘┴└─  ╚  ┴ ┴└─┘ ┴   ╚═╝╩  ╚═╝  ╚═╝┴  └─┘┘└┘ ╚╝ ╩╝╚╝╚═╝</div>
            <h1>Speech to Text API</h1>
            <p class="subtitle">⚡ Fast, Local, Private | OpenAI Compatible</p>
        </div>
        
        <div class="content">
            <div class="status">
                <h2>✅ Server is Running</h2>
                <p><strong>Status:</strong> Online and ready to transcribe</p>
                <p><strong>Version:</strong> 1.0.0</p>
            </div>
            
            <div class="config-info">
                <h3>⚙️ Current Configuration</h3>
                <div class="config-item">
                    <span class="config-label">Engine:</span>
                    <span class="config-value">""" + server_config.engine + """</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Model:</span>
                    <span class="config-value">""" + server_config.model + """</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Language:</span>
                    <span class="config-value">""" + (server_config.language or "auto-detect") + """</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Device:</span>
                    <span class="config-value">""" + server_config.device + """</span>
                </div>
            </div>
            
            <h2 style="margin-bottom: 20px;">🌟 Key Features</h2>
            <div class="features">
                <div class="feature-card">
                    <h3>🚀 Fast CPU Inference</h3>
                    <p>Optimized for Intel CPUs using OpenVINO. Get 6-10x realtime performance with low latency.</p>
                </div>
                <div class="feature-card">
                    <h3>🎯 OpenAI Compatible</h3>
                    <p>Drop-in replacement for OpenAI Whisper API. Works with existing tools and integrations.</p>
                </div>
                <div class="feature-card">
                    <h3>🌐 Open-WebUI Ready</h3>
                    <p>Full support for Open-WebUI voice input. Easy setup with dynamic configuration.</p>
                </div>
                <div class="feature-card">
                    <h3>🔒 100% Local</h3>
                    <p>All processing happens on your machine. Your audio never leaves your computer.</p>
                </div>
                <div class="feature-card">
                    <h3>🧠 Smart Language Detection</h3>
                    <p>Automatically detects the language being spoken. Supports 99+ languages.</p>
                </div>
                <div class="feature-card">
                    <h3>⚡ Multiple Formats</h3>
                    <p>Supports MP3, WAV, M4A, FLAC, OGG, and WebM audio files.</p>
                </div>
            </div>
            
            <div class="endpoints">
                <h2>📡 API Endpoints</h2>
                
                <div class="endpoint">
                    <span class="endpoint-method post">POST</span>
                    <span class="endpoint-path">/v1/audio/transcriptions</span>
                    <div class="endpoint-desc">Transcribe audio to text (OpenAI compatible)</div>
                </div>
                
                <div class="endpoint">
                    <span class="endpoint-method post">POST</span>
                    <span class="endpoint-path">/v1/audio/translations</span>
                    <div class="endpoint-desc">Translate audio to English text</div>
                </div>
                
                <div class="endpoint">
                    <span class="endpoint-method">GET</span>
                    <span class="endpoint-path">/v1/models</span>
                    <div class="endpoint-desc">List available models</div>
                </div>
                
                <div class="endpoint">
                    <span class="endpoint-method">GET</span>
                    <span class="endpoint-path">/config</span>
                    <div class="endpoint-desc">Get current server configuration</div>
                </div>
                
                <div class="endpoint">
                    <span class="endpoint-method post">POST</span>
                    <span class="endpoint-path">/config/update</span>
                    <div class="endpoint-desc">Update server settings on-the-fly</div>
                </div>
                
                <div class="endpoint">
                    <span class="endpoint-method">GET</span>
                    <span class="endpoint-path">/health</span>
                    <div class="endpoint-desc">Check server health status</div>
                </div>
                
                <div class="endpoint">
                    <span class="endpoint-method">GET</span>
                    <span class="endpoint-path">/docs</span>
                    <div class="endpoint-desc">Interactive API documentation (Swagger UI)</div>
                </div>
            </div>
            
            <div class="quick-start">
                <h3>🚀 Quick Test</h3>
                <p>Test the transcription endpoint with a simple curl command:</p>
                <div class="code-block">curl -X POST "http://localhost:8000/v1/audio/transcriptions" \\
  -F "file=@your_audio.mp3" \\
  -F "model=whisper-1"</div>
            </div>
            
            <div class="quick-start">
                <h3>🌐 Use with Open-WebUI</h3>
                <p>Configure Open-WebUI to use this server:</p>
                <ol style="margin-left: 20px; margin-top: 10px;">
                    <li>Open <strong>Settings → Audio</strong> in Open-WebUI</li>
                    <li>Set <strong>STT Engine</strong> to <code>OpenAI</code></li>
                    <li>Set <strong>API Base URL</strong> to <code>http://localhost:8000/v1</code></li>
                    <li>Set <strong>Model</strong> to <code>whisper-1</code></li>
                    <li>Click <strong>Save</strong> and start using voice input!</li>
                </ol>
            </div>
        </div>
        
        <div class="footer">
            <p><strong>Whisper-Fast-CPU-OpenVINO</strong> | Made with ❤️ for fast, private, local speech recognition</p>
            <p style="margin-top: 10px;">
                <a href="/docs" target="_blank">API Documentation</a> | 
                <a href="/health">Health Check</a> | 
                <a href="https://github.com/groxaxo/Whisper-Fast-Cpu-OpenVino" target="_blank">GitHub</a>
            </p>
        </div>
    </div>
</body>
</html>
"""
    return HTMLResponse(content=html_content)


@app.get("/api/info")
async def api_info():
    """JSON API information endpoint"""
    return {
        "name": "OpenVINO Whisper API",
        "version": "1.0.0",
        "status": "online",
        "endpoints": {
            "transcribe": "/v1/audio/transcriptions or /audio/transcriptions",
            "translate": "/v1/audio/translations or /audio/translations",
            "models": "/v1/models",
            "config": "/config",
            "config_update": "/config/update",
            "health": "/health",
            "docs": "/docs",
        },
        "compatible_with": ["OpenAI Whisper API", "Open WebUI"],
        "current_config": {
            "engine": server_config.engine,
            "model": server_config.model,
            "language": server_config.language or "auto-detect",
            "device": server_config.device,
        }
    }


def print_banner():
    """Print ASCII banner on startup"""
    banner = """
╦ ╦┬ ┬┬┌─┐┌─┐┌─┐┬─┐  ╔═╗┌─┐┌─┐┌┬┐  ╔═╗╔═╗╦ ╦  ╔═╗┌─┐┌─┐┌┐┌╦  ╦╦╔╗╔╔═╗
║║║├─┤│└─┐├─┘├┤ ├┬┘  ╠╣ ├─┤└─┐ │───║  ╠═╝║ ║  ║ ║├─┘├┤ │││╚╗╔╝║║║║║ ║
╚╩╝┴ ┴┴└─┘┴  └─┘┴└─  ╚  ┴ ┴└─┘ ┴   ╚═╝╩  ╚═╝  ╚═╝┴  └─┘┘└┘ ╚╝ ╩╝╚╝╚═╝
⚡ Fast, Local Speech-to-Text on CPU | OpenAI API Compatible
"""
    print(banner)

import argparse
from typing import Dict
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Whisper-Fast-CPU-OpenVINO API Server")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./models/whisper-tiny-en",
        help="Path to the directory containing the OpenVINO model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="CPU",
        help="Device to run inference on (e.g., CPU, GPU).",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host address to bind to."
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to listen on."
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Number of threads to use for inference. 0 for automatic.",
    )
    parser.add_argument(
        "--streams",
        type=int,
        default=0,
        help="Number of streams to use for inference. 0 for automatic.",
    )
    parser.add_argument(
        "--hint",
        type=str,
        default="performance",
        choices=["performance", "latency"],
        help="Performance hint for OpenVINO (performance or latency).",
    )
    parser.add_argument(
        "--engine",
        default="openvino",
        choices=["openvino", "faster-whisper"],
        help="Backend engine to use (openvino or faster-whisper).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model ID for faster-whisper (e.g. large-v3-turbo). If not set, uses model-dir for OpenVINO.",
    )
    return parser.parse_args()


def load_language_tokens(model_dir: str) -> Dict[str, str]:
    """Load language tokens from model config"""
    config_path = os.path.join(model_dir, "generation_config.json")

    if not os.path.exists(config_path):
        LOGGER.warning(f"Model configuration file not found: {config_path}")
        return {}

    with open(config_path, "r") as f:
        config = json.load(f)

    # Extract language mapping
    if "lang_to_id" in config:
        return {k: v for k, v in config["lang_to_id"].items()}
    
    return {}


def main() -> None:
    """Main entry point"""
    global pipeline, language_tokens, server_config

    args = parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )
    
    # Print banner
    print_banner()

    # Initialize server configuration with command-line args
    server_config.engine = args.engine
    server_config.model_dir = args.model_dir
    server_config.device = args.device
    server_config.threads = args.threads
    server_config.streams = args.streams
    server_config.hint = args.hint
    
    if args.model:
        server_config.model = args.model
    # If using OpenVINO, model is implied by model_dir, but for consistency:
    if args.engine == "openvino":
        server_config.model = args.model_dir

    # Set thread environment variables
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(args.threads))
    os.environ.setdefault("OV_CPU_THREADS_NUM", str(args.threads))

    # Load model
    if args.engine == "openvino":
        LOGGER.info(f"Loading OpenVINO model from: {args.model_dir}")
        language_tokens = load_language_tokens(args.model_dir)
    else:
        LOGGER.info(f"Loading Faster-Whisper model: {server_config.model}")
        
    pipeline = build_pipeline(server_config)
    LOGGER.info("Model loaded successfully!")

    # Start server
    LOGGER.info(f"Starting OpenAI-compatible API server on {args.host}:{args.port}")
    LOGGER.info(f"API Documentation: http://{args.host}:{args.port}/docs")
    LOGGER.info(
        f"Compatible with Open WebUI - use http://{args.host}:{args.port} as base URL"
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
