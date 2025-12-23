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
from fastapi.responses import JSONResponse
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


def build_pipeline(
    model_dir: str, device: str, threads: int, streams: Union[str, int], hint: str
) -> ov_genai.WhisperPipeline:
    """Build OpenVINO Whisper pipeline"""
    extra_kwargs: Dict[str, Union[str, int]] = {}

    # CPU-specific optimizations
    if device == "CPU":
        extra_kwargs["INFERENCE_NUM_THREADS"] = threads
        extra_kwargs["PERFORMANCE_HINT"] = hint
        if streams:
            extra_kwargs["NUM_STREAMS"] = streams
    # GPU-specific optimizations (Intel Iris Xe, etc.)
    elif device == "GPU":
        extra_kwargs["PERFORMANCE_HINT"] = hint
        # GPU streams can improve throughput for parallel requests
        if streams:
            extra_kwargs["NUM_STREAMS"] = streams
    # AUTO mode - let OpenVINO decide
    else:  # AUTO
        extra_kwargs["PERFORMANCE_HINT"] = hint
        if streams:
            extra_kwargs["NUM_STREAMS"] = streams

    LOGGER.info(
        "Loading Whisper pipeline (device=%s, threads=%d, streams=%s)",
        device,
        threads if device == "CPU" else 0,
        streams,
    )
    return ov_genai.WhisperPipeline(model_dir, device, **extra_kwargs)


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

        # Set language if specified, otherwise use server_config or auto-detect
        effective_language = language or server_config.language
        if effective_language and effective_language != "auto":
            whisper_token = language_tokens.get(effective_language)
            if whisper_token:
                kwargs["language"] = whisper_token
                LOGGER.info(f"Using language: {effective_language}")
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


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "OpenVINO Whisper API",
        "version": "1.0.0",
        "endpoints": {
            "transcribe": "/v1/audio/transcriptions or /audio/transcriptions",
            "translate": "/v1/audio/translations or /audio/translations",
            "models": "/v1/models",
            "config": "/config",
            "config_update": "/config/update",
            "health": "/health",
        },
        "compatible_with": "OpenAI Whisper API, Open WebUI",
        "current_config": {
            "engine": server_config.engine,
            "model": server_config.model,
            "language": server_config.language or "auto-detect",
        }
    }


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

    # Initialize server configuration with command-line args
    server_config.model_dir = args.model_dir
    server_config.device = args.device
    server_config.threads = args.threads
    server_config.streams = args.streams
    server_config.hint = args.hint

    # Set thread environment variables
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(args.threads))
    os.environ.setdefault("OV_CPU_THREADS_NUM", str(args.threads))

    # Load model
    LOGGER.info(f"Loading model from: {args.model_dir}")
    language_tokens = load_language_tokens(args.model_dir)
    pipeline = build_pipeline(args.model_dir, args.device, args.threads, args.streams, args.hint)
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
