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
import torchaudio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import openvino_genai as ov_genai

TARGET_SAMPLE_RATE = 16000
LOGGER = logging.getLogger("serve_openai_api")

# Initialize FastAPI app
app = FastAPI(
    title="OpenVINO Whisper API",
    description="OpenAI-compatible Whisper API using OpenVINO",
    version="1.0.0"
)

# Global variables for model
pipeline: Optional[ov_genai.WhisperPipeline] = None
language_tokens: Dict[str, str] = {}


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
        help="Target device for OpenVINO execution (default: CPU)."
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host/IP for the API server bind (default: 0.0.0.0)."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the API server (default: 8000)."
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of CPU threads to dedicate to inference."
    )
    parser.add_argument(
        "--streams",
        type=parse_streams,
        default="AUTO",
        help="Number of parallel inference streams (integer or AUTO).",
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
    model_dir: str,
    device: str,
    threads: int,
    streams: Union[str, int]
) -> ov_genai.WhisperPipeline:
    """Build OpenVINO Whisper pipeline"""
    extra_kwargs: Dict[str, Union[str, int]] = {
        "INFERENCE_NUM_THREADS": threads,
        "PERFORMANCE_HINT": "THROUGHPUT",
    }
    if streams:
        extra_kwargs["NUM_STREAMS"] = streams

    LOGGER.info(
        "Loading Whisper pipeline (device=%s, threads=%d, streams=%s)",
        device, threads, streams
    )
    return ov_genai.WhisperPipeline(model_dir, device, **extra_kwargs)


def load_audio_file(file_data: bytes) -> tuple[np.ndarray, int]:
    """Load audio from uploaded file bytes"""
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_file:
        tmp_file.write(file_data)
        tmp_path = tmp_file.name
    
    try:
        # Load audio using torchaudio
        waveform, sample_rate = torchaudio.load(tmp_path)
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if sample_rate != TARGET_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, TARGET_SAMPLE_RATE
            )
        
        # Convert to numpy
        audio_array = waveform.squeeze().numpy()
        
        # Normalize
        max_abs = float(np.max(np.abs(audio_array))) if audio_array.size else 0.0
        if max_abs > 1.0:
            audio_array = audio_array / max_abs
        
        return audio_array.astype(np.float32), TARGET_SAMPLE_RATE
        
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
    timestamp_granularities: Optional[str] = Form(None)
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
        
        # Set language if specified
        if language and language != "auto":
            whisper_token = language_tokens.get(language)
            if whisper_token:
                kwargs["language"] = whisper_token
            else:
                LOGGER.warning(f"Unknown language code: {language}, using auto-detect")
        
        # Run inference
        start_time = time.time()
        audio_data = audio_array.tolist()
        result = pipeline.generate(audio_data, **kwargs)
        inference_time = time.time() - start_time
        
        # Extract text
        text = result.texts[0].strip() if result.texts else ""
        
        LOGGER.info(
            f"Transcription completed: {len(text)} chars, "
            f"took {inference_time:.2f}s ({duration/inference_time:.2f}x realtime)"
        )
        
        # Build response
        response_data = {
            "text": text,
            "task": "transcribe",
            "language": language or "auto",
            "duration": duration
        }
        
        # Add segments if requested
        if timestamp_granularities == "segment" and hasattr(result, 'chunks'):
            segments = []
            for chunk in result.chunks:
                segments.append({
                    "start": float(chunk.start_ts),
                    "end": float(chunk.end_ts),
                    "text": chunk.text.strip()
                })
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
    temperature: float = Form(0.0)
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
            f"Translation completed: {len(text)} chars, "
            f"took {inference_time:.2f}s"
        )
        
        # Build response
        response_data = {
            "text": text,
            "task": "translate",
            "language": "en",
            "duration": duration
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
                "owned_by": "openvino"
            }
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None
    }


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "OpenVINO Whisper API",
        "version": "1.0.0",
        "endpoints": {
            "transcribe": "/v1/audio/transcriptions",
            "translate": "/v1/audio/translations",
            "models": "/v1/models",
            "health": "/health"
        },
        "compatible_with": "OpenAI Whisper API, Open WebUI"
    }


def main() -> None:
    """Main entry point"""
    global pipeline, language_tokens
    
    args = parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )
    
    # Set thread environment variables
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(args.threads))
    os.environ.setdefault("OV_CPU_THREADS_NUM", str(args.threads))
    
    # Load model
    LOGGER.info(f"Loading model from: {args.model_dir}")
    language_tokens = load_language_tokens(args.model_dir)
    pipeline = build_pipeline(args.model_dir, args.device, args.threads, args.streams)
    LOGGER.info("Model loaded successfully!")
    
    # Start server
    LOGGER.info(f"Starting OpenAI-compatible API server on {args.host}:{args.port}")
    LOGGER.info(f"API Documentation: http://{args.host}:{args.port}/docs")
    LOGGER.info(f"Compatible with Open WebUI - use http://{args.host}:{args.port} as base URL")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
