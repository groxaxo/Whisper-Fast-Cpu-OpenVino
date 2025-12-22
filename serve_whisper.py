#!/usr/bin/env python3
"""Gradio-based Whisper server using OpenVINO GenAI pipeline."""

from __future__ import annotations

import argparse
import json
import os
import time
import copy
import gc
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple, Union

import logging
import sys
from threading import Lock

import numpy as np
import torch
import torchaudio
import gradio as gr
import openvino_genai as ov_genai

TARGET_SAMPLE_RATE = 16000
LOGGER = logging.getLogger("serve_whisper")
PIPELINE_LOCK = Lock()
LAST_CALL_TIME = {"streaming": 0.0, "upload": 0.0}
MIN_CALL_INTERVAL = 1.5  # Minimum seconds between calls


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
    parser = argparse.ArgumentParser(description="Serve Whisper over HTTP using OpenVINO GenAI.")
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
    parser.add_argument("--host", default="0.0.0.0", help="Host/IP for the Gradio server bind (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=7860, help="Port for the Gradio server (default: 7860).")
    parser.add_argument(
        "--threads",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of CPU threads to dedicate to inference (default: min(8, CPU count) to prevent memory issues).",
    )
    parser.add_argument(
        "--streams",
        type=parse_streams,
        default="AUTO",
        help="Number of parallel inference streams (integer or AUTO).",
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=30.0,
        help="Maximum duration in seconds per inference batch (set <=0 to disable batching).",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Enable Gradio public sharing tunnel (use with caution).",
    )
    return parser.parse_args()


@lru_cache(maxsize=1)
def load_language_tokens(model_dir: str) -> Dict[str, str]:
    config_path = os.path.join(model_dir, "generation_config.json")
    
    # Check if file exists, if not attempt auto-download or provide helpful error message
    if not os.path.exists(config_path):
        LOGGER.warning(f"generation_config.json not found at: {config_path}")
        LOGGER.info("Attempting auto-download...")
        
        # Try to auto-download
        import subprocess
        try:
            result = subprocess.run(
                [sys.executable, "setup_model.py", "--auto", "--target-dir", model_dir],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0 and os.path.exists(config_path):
                LOGGER.info("Model downloaded successfully!")
            else:
                raise FileNotFoundError("Auto-download failed")
                
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            LOGGER.error(f"Auto-download failed: {e}")
            LOGGER.error(f"Current working directory: {os.getcwd()}")
            LOGGER.error(f"Model directory contents: {os.listdir(model_dir) if os.path.exists(model_dir) else 'Directory does not exist'}")
            LOGGER.error("\nPlease ensure:")
            LOGGER.error("1. You are running the script from the correct directory")
            LOGGER.error("2. Download the model manually: python setup_model.py")
            LOGGER.error("3. The model directory path is correct (use --model-dir to specify)")
            raise FileNotFoundError(
                f"Model configuration file not found: {config_path}\n"
                f"Run 'python setup_model.py' to download the model, or check your --model-dir path."
            )
    
    with open(config_path, "r", encoding="utf-8") as config_file:
        data = json.load(config_file)
    language_tokens: Dict[str, str] = {}
    for token in data.get("lang_to_id", {}):
        plain = token.strip("<|>")
        language_tokens[plain] = token
    return language_tokens


def build_pipeline(model_dir: str, device: str, threads: int, streams: Union[str, int]) -> ov_genai.WhisperPipeline:
    extra_kwargs: Dict[str, Union[str, int]] = {}
    
    # CPU-specific optimizations
    if device == "CPU":
        extra_kwargs["INFERENCE_NUM_THREADS"] = threads
        extra_kwargs["PERFORMANCE_HINT"] = "THROUGHPUT"
        if streams:
            extra_kwargs["NUM_STREAMS"] = streams
    # GPU-specific optimizations (Intel Iris Xe, etc.)
    elif device == "GPU":
        extra_kwargs["PERFORMANCE_HINT"] = "LATENCY"
        # GPU doesn't use CPU threads, but streams can improve throughput
        if streams:
            extra_kwargs["NUM_STREAMS"] = streams
        # Enable GPU throttling for better stability on integrated graphics
        extra_kwargs["GPU_THROUGHPUT_STREAMS"] = 1 if streams == "AUTO" else streams
    # AUTO mode - let OpenVINO decide
    else:  # AUTO
        extra_kwargs["PERFORMANCE_HINT"] = "LATENCY"
        if streams:
            extra_kwargs["NUM_STREAMS"] = streams

    LOGGER.info(
        "Loading Whisper pipeline (device=%s, threads=%d, streams=%s, performance_hint=%s)",
        device,
        threads if device == "CPU" else 0,
        streams,
        extra_kwargs.get("PERFORMANCE_HINT"),
    )
    return ov_genai.WhisperPipeline(model_dir, device, **extra_kwargs)


def ensure_mono(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        return data
    # Average channels to get mono audio.
    return data.mean(axis=1)


def resample_if_needed(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    if sample_rate == TARGET_SAMPLE_RATE:
        return waveform
    return torchaudio.functional.resample(waveform, sample_rate, TARGET_SAMPLE_RATE)


def normalize_waveform(waveform: np.ndarray) -> np.ndarray:
    max_abs = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if max_abs > 1.0:
        waveform = waveform / max_abs
    return waveform.astype(np.float32)


def prepare_audio(audio: Tuple[int, Sequence[float]] | None) -> Optional[Tuple[np.ndarray, int]]:
    if audio is None:
        LOGGER.warning("Received empty audio payload.")
        return None

    sample_rate: Optional[int] = None
    data: Optional[Sequence[float] | np.ndarray] = None

    if isinstance(audio, (list, tuple)):
        if len(audio) == 2 and isinstance(audio[0], (int, float)):
            sample_rate = int(audio[0])
            data = audio[1]
        else:
            data = audio
    elif isinstance(audio, dict):
        sample_rate = audio.get("sample_rate") or audio.get("sampling_rate")
        data = audio.get("data") or audio.get("array")
    elif isinstance(audio, np.ndarray):
        data = audio
    else:
        LOGGER.error("Unsupported audio payload type: %s", type(audio).__name__)
        return None

    if data is None:
        LOGGER.warning("Audio payload missing data field.")
        return None

    waveform = ensure_mono(np.asarray(data, dtype=np.float32))
    if not waveform.size:
        LOGGER.warning("Audio payload is empty after mono conversion.")
        return None

    if sample_rate is None:
        sample_rate = TARGET_SAMPLE_RATE

    tensor = torch.from_numpy(waveform)
    tensor = resample_if_needed(tensor, sample_rate)
    processed = tensor.numpy()
    normalized = normalize_waveform(processed)
    return normalized, TARGET_SAMPLE_RATE


def iter_waveform_batches(
    waveform: np.ndarray,
    sample_rate: int,
    segment_seconds: float,
) -> List[Tuple[int, np.ndarray]]:
    if segment_seconds <= 0:
        return [(0, waveform)]

    chunk_samples = max(int(segment_seconds * sample_rate), sample_rate)
    total_samples = len(waveform)
    batches: List[Tuple[int, np.ndarray]] = []
    for start in range(0, total_samples, chunk_samples):
        end = min(start + chunk_samples, total_samples)
        chunk = waveform[start:end]
        if chunk.size:
            batches.append((start, chunk))
    return batches


def transcribe(
    audio: Tuple[int, Sequence[float]] | None,
    language_code: str,
    task: str,
    return_timestamps: bool,
    pipeline: ov_genai.WhisperPipeline,
    language_tokens: Dict[str, str],
    segment_seconds: float,
):
    prepared = prepare_audio(audio)
    if prepared is None:
        return "", []
    waveform, sample_rate = prepared

    duration = len(waveform) / float(sample_rate)
    
    # Skip very short audio chunks to prevent crashes
    if duration < 1.0:
        LOGGER.debug("Skipping audio chunk too short: %.2fs", duration)
        return "", []
    
    LOGGER.info(
        "Transcribing audio duration=%.2fs sr=%d language=%s task=%s",
        duration,
        sample_rate,
        language_code,
        task,
    )

    kwargs = {"return_timestamps": return_timestamps}
    if task in {"transcribe", "translate"}:
        kwargs["task"] = task
    if language_code and language_code != "auto":
        whisper_token = language_tokens.get(language_code)
        if whisper_token is None:
            LOGGER.error("Unsupported language code provided: %s", language_code)
            raise gr.Error(f"Unsupported language code: {language_code}")
        kwargs["language"] = whisper_token

    batches = iter_waveform_batches(waveform, sample_rate, segment_seconds)

    combined_text_parts: List[str] = []
    segments: List[Dict[str, float | str]] = []

    for start_sample, batch in batches:
        offset_seconds = start_sample / float(sample_rate)
        
        # Skip empty or too-short batches
        if len(batch) < sample_rate:  # Less than 1 second
            continue
            
        try:
            with PIPELINE_LOCK:
                # Convert to list only once, ensure it's a clean copy
                audio_data = batch.astype(np.float32).tolist()
                result = pipeline.generate(audio_data, **kwargs)
        except RuntimeError as exc:
            LOGGER.exception("Inference request failed")
            raise gr.Error("Inference backend is busy, please retry.") from exc
        except Exception as exc:
            LOGGER.exception("Unexpected error during inference")
            return "", []

        batch_text = result.texts[0].strip() if result.texts else ""
        if batch_text:
            combined_text_parts.append(batch_text)

        if return_timestamps:
            for chunk in result.chunks:
                segments.append(
                    {
                        "start": round(offset_seconds + float(chunk.start_ts), 3),
                        "end": round(offset_seconds + float(chunk.end_ts), 3),
                        "text": chunk.text.strip(),
                    }
                )

    full_text = " ".join(part for part in combined_text_parts).strip()

    if return_timestamps:
        LOGGER.info(
            "Transcription output length=%d characters, segments=%d", len(full_text), len(segments)
        )
        return full_text, segments

    LOGGER.info("Transcription output length=%d characters", len(full_text))
    return full_text, []


def create_interface(
    pipeline: ov_genai.WhisperPipeline,
    language_tokens: Dict[str, str],
    segment_seconds: float,
):
    language_options = ["auto"] + sorted(language_tokens.keys())

    with gr.Blocks(title="OpenVINO Whisper Server") as demo:
        gr.Markdown("# OpenVINO Whisper Realtime Transcription")
        gr.Markdown(
            "**🎤 Streaming Mode:** Use microphone for real-time transcription with accumulation.\n\n"
            "**📁 Upload Mode:** Upload audio file, trim to desired length using the waveform controls, then click 'Process' button."
        )

        with gr.Row():
            with gr.Column():
                # Streaming audio input (microphone only)
                streaming_audio = gr.Audio(
                    sources=["microphone"], 
                    type="numpy", 
                    streaming=True, 
                    label="🎤 Streaming Audio (Microphone)",
                    show_download_button=False,
                )
                # Upload audio input (file upload only) with trimming support
                upload_audio = gr.Audio(
                    sources=["upload"], 
                    type="numpy", 
                    streaming=False, 
                    label="📁 Upload Audio (File)",
                    show_download_button=True,
                    editable=True,
                    waveform_options={
                        "show_recording_waveform": True,
                        "show_controls": True,
                    },
                )
                
                # Manual trim controls for precise selection
                with gr.Row():
                    trim_start = gr.Number(
                        label="Trim Start (seconds)", 
                        value=0, 
                        minimum=0,
                        precision=2,
                        info="Start time in seconds (e.g., 5.5 = 5.5 seconds)"
                    )
                    trim_end = gr.Number(
                        label="Trim End (seconds)", 
                        value=None,
                        minimum=0,
                        precision=2,
                        info="End time in seconds (leave empty for end of file)"
                    )
                
                process_btn = gr.Button("🎯 Process Uploaded Audio", variant="primary", size="lg")
            
            with gr.Column():
                language_dd = gr.Dropdown(
                    choices=language_options,
                    value="auto",
                    label="Language",
                    info="Select a target language or leave on auto-detect.",
                )
                task_radio = gr.Radio(
                    choices=["transcribe", "translate"],
                    value="transcribe",
                    label="Task",
                    info="Transcribe keeps source language; translate converts to English.",
                )
                timestamps_checkbox = gr.Checkbox(value=True, label="Return timestamps")
                clear_btn = gr.Button("🗑️ Clear Transcript", variant="secondary")

        status_text = gr.Textbox(label="Status", value="Ready", lines=1, interactive=False)
        transcript_output = gr.Textbox(label="Transcript", lines=6)
        segments_output = gr.Dataframe(
            headers=["start", "end", "text"],
            datatype=["number", "number", "str"],
            label="Segments",
        )

        # State to accumulate streaming transcripts
        accumulated_text = gr.State(value="")
        accumulated_segments = gr.State(value=[])

        def streaming_handler(audio, prev_text, prev_segments, language_choice, task_choice, ts_flag):
            """Handle streaming audio - accumulates results"""
            try:
                if audio is None:
                    return prev_text, prev_segments, prev_text, prev_segments
                
                # Rate limiting - prevent too-frequent calls
                current_time = time.time()
                if current_time - LAST_CALL_TIME["streaming"] < MIN_CALL_INTERVAL:
                    LOGGER.debug("Skipping call - too soon (rate limiting)")
                    return prev_text, prev_segments, prev_text, prev_segments
                
                LAST_CALL_TIME["streaming"] = current_time
                
                # Transcribe the new chunk
                new_text, new_segments = transcribe(
                    audio,
                    language_choice,
                    task_choice,
                    ts_flag,
                    pipeline,
                    language_tokens,
                    segment_seconds,
                )
                
                # Accumulate: append new text to previous (with deep copy to prevent memory issues)
                if new_text:
                    combined_text = (prev_text + " " + new_text).strip() if prev_text else new_text
                    # Deep copy segments to prevent reference issues
                    prev_segs_copy = copy.deepcopy(prev_segments) if prev_segments else []
                    new_segs_copy = copy.deepcopy(new_segments) if new_segments else []
                    combined_segments = prev_segs_copy + new_segs_copy
                else:
                    combined_text = prev_text
                    combined_segments = copy.deepcopy(prev_segments) if prev_segments else []
                
                # Explicit garbage collection to prevent memory buildup
                gc.collect()
                
                # Return: updated state and outputs
                return combined_text, combined_segments, combined_text, combined_segments
                
            except Exception as e:
                LOGGER.exception("Error in streaming handler")
                error_msg = f"Error: {str(e)}"
                # Return safe copies on error
                safe_text = str(prev_text) if prev_text else ""
                safe_segments = copy.deepcopy(prev_segments) if prev_segments else []
                gc.collect()  # Clean up after error too
                return safe_text, safe_segments, error_msg, safe_segments

        def upload_handler(audio, trim_start_val, trim_end_val, language_choice, task_choice, ts_flag):
            """Handle uploaded audio files - processes complete file or trimmed portion"""
            try:
                if audio is None:
                    return "⚠️ No audio file selected", "", [], "", []
                
                # Rate limiting for uploads too
                current_time = time.time()
                if current_time - LAST_CALL_TIME["upload"] < MIN_CALL_INTERVAL:
                    LOGGER.warning("Upload called too soon, throttling")
                    time.sleep(MIN_CALL_INTERVAL - (current_time - LAST_CALL_TIME["upload"]))
                
                LAST_CALL_TIME["upload"] = current_time
                
                # Apply trimming if specified
                if isinstance(audio, tuple):
                    sample_rate, audio_data = audio
                else:
                    # If it's already just the array
                    audio_data = audio[1] if isinstance(audio, tuple) else audio
                    sample_rate = audio[0] if isinstance(audio, tuple) else TARGET_SAMPLE_RATE
                
                total_duration = len(audio_data) / float(sample_rate)
                
                # Apply trim
                start_sample = 0
                end_sample = len(audio_data)
                
                if trim_start_val is not None and trim_start_val > 0:
                    start_sample = int(trim_start_val * sample_rate)
                    start_sample = max(0, min(start_sample, len(audio_data)))
                
                if trim_end_val is not None and trim_end_val > 0:
                    end_sample = int(trim_end_val * sample_rate)
                    end_sample = max(start_sample, min(end_sample, len(audio_data)))
                
                # Extract trimmed portion
                trimmed_audio = audio_data[start_sample:end_sample]
                
                if len(trimmed_audio) == 0:
                    return "⚠️ Invalid trim range - no audio data", "", [], "", []
                
                trimmed_duration = len(trimmed_audio) / float(sample_rate)
                
                LOGGER.info(f"Processing audio: total={total_duration:.2f}s, trimmed={trimmed_duration:.2f}s (from {trim_start_val or 0}s to {trim_end_val or total_duration}s)")
                
                # Create tuple for transcribe function
                audio_to_process = (sample_rate, trimmed_audio)
                
                text, segments = transcribe(
                    audio_to_process,
                    language_choice,
                    task_choice,
                    ts_flag,
                    pipeline,
                    language_tokens,
                    segment_seconds,
                )
                
                # For uploads, replace everything (don't accumulate)
                # Use deep copy to ensure clean data
                clean_segments = copy.deepcopy(segments) if segments else []
                gc.collect()  # Clean up memory
                
                if trim_start_val or trim_end_val:
                    status_msg = f"✅ Processed {trimmed_duration:.2f}s (trimmed from {total_duration:.2f}s) - {len(text)} characters"
                else:
                    status_msg = f"✅ Processed {trimmed_duration:.2f}s of audio - {len(text)} characters"
                
                return status_msg, text, clean_segments, text, clean_segments
                
            except Exception as e:
                LOGGER.exception("Error in upload handler")
                gc.collect()
                error_msg = f"❌ Error: {str(e)}"
                return error_msg, f"Error: {str(e)}", [], f"Error: {str(e)}", []

        def clear_transcript():
            """Clear accumulated transcript"""
            return "🗑️ Cleared", "", [], "", []

        # Streaming: accumulate results as user speaks
        # Increased interval to 3.0 seconds to prevent memory corruption
        streaming_audio.stream(
            streaming_handler,
            inputs=[streaming_audio, accumulated_text, accumulated_segments, language_dd, task_radio, timestamps_checkbox],
            outputs=[accumulated_text, accumulated_segments, transcript_output, segments_output],
            stream_every=3.0,  # Increased from 2.0 to reduce memory pressure
        )

        # Upload: process when button is clicked (not automatic)
        process_btn.click(
            upload_handler,
            inputs=[upload_audio, trim_start, trim_end, language_dd, task_radio, timestamps_checkbox],
            outputs=[status_text, accumulated_text, accumulated_segments, transcript_output, segments_output],
        )

        # Clear button
        clear_btn.click(
            clear_transcript,
            outputs=[status_text, accumulated_text, accumulated_segments, transcript_output, segments_output],
        )

    return demo


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(args.threads))
    os.environ.setdefault("OV_CPU_THREADS_NUM", str(args.threads))

    language_tokens = load_language_tokens(args.model_dir)
    pipeline = build_pipeline(args.model_dir, args.device, args.threads, args.streams)

    interface = create_interface(pipeline, language_tokens, args.segment_seconds)
    interface.queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
