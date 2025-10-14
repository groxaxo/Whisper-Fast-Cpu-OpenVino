#!/usr/bin/env python3
"""Benchmark script for Whisper-Fast-CPU-OpenVINO"""

import time
import sys
import os
from pathlib import Path
import json

import numpy as np
import torchaudio
import openvino_genai as ov_genai


def get_audio_duration(audio_path):
    """Get audio duration in seconds"""
    waveform, sample_rate = torchaudio.load(audio_path)
    return waveform.shape[1] / sample_rate


def benchmark_file(pipeline, audio_path, device="CPU", threads=8):
    """Benchmark a single audio file"""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {Path(audio_path).name}")
    print(f"{'='*60}")
    
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Resample to 16kHz if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        sample_rate = 16000
    
    # Convert to numpy
    audio_data = waveform.squeeze().numpy().astype(np.float32).tolist()
    
    # Get duration
    duration = len(audio_data) / sample_rate
    
    print(f"Duration: {duration:.2f}s")
    print(f"Sample rate: {sample_rate}Hz")
    print(f"Device: {device}")
    print(f"Threads: {threads}")
    
    # Warm up (first run is slower)
    print("\nWarming up...")
    _ = pipeline.generate(audio_data[:16000])  # 1 second warmup
    
    # Benchmark
    print("Running benchmark...")
    start_time = time.time()
    result = pipeline.generate(audio_data)
    end_time = time.time()
    
    processing_time = end_time - start_time
    rtf = processing_time / duration  # Real-time factor
    speed = duration / processing_time  # Speed multiplier
    
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"{'='*60}")
    print(f"Audio duration:     {duration:.2f}s")
    print(f"Processing time:    {processing_time:.2f}s")
    print(f"Real-time factor:   {rtf:.2f}x (lower is better)")
    print(f"Speed:              {speed:.2f}x real-time")
    print(f"Transcript length:  {len(result)} characters")
    print(f"\nTranscript preview:")
    print(f"{result[:200]}..." if len(result) > 200 else result)
    
    return {
        "file": Path(audio_path).name,
        "duration_seconds": round(duration, 2),
        "processing_time_seconds": round(processing_time, 2),
        "real_time_factor": round(rtf, 2),
        "speed_multiplier": round(speed, 2),
        "transcript_length": len(result),
        "transcript_preview": result[:100]
    }


def get_model_info(model_dir="model"):
    """Get model information from MODEL_INFO.txt or config"""
    info_file = Path(model_dir) / "MODEL_INFO.txt"
    if info_file.exists():
        with open(info_file, 'r') as f:
            content = f.read()
            for line in content.split('\n'):
                if line.startswith('Model:'):
                    return line.split(':', 1)[1].strip()
    return "whisper-large-v3 (unknown quantization)"


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark Whisper models")
    parser.add_argument('--model-dir', default='model', help='Model directory (default: model)')
    parser.add_argument('--threads', type=int, default=8, help='Number of CPU threads (default: 8)')
    parser.add_argument('--device', default='CPU', help='Device (default: CPU)')
    args = parser.parse_args()
    
    # Test files
    test_files = [
        "/home/op/neutts-air/neutts-air/samples/dave.wav",
        "/home/op/happiness_book_output/chapter_01.wav",
        "/home/op/ov-whisper/sample.wav",
    ]
    
    # Filter existing files
    test_files = [f for f in test_files if os.path.exists(f)]
    
    if not test_files:
        print("No test files found!")
        sys.exit(1)
    
    # Configuration
    device = args.device
    threads = args.threads
    model_path = args.model_dir
    model_name = get_model_info(model_path)
    
    print(f"\n{'#'*60}")
    print(f"# Whisper-Fast-CPU-OpenVINO Benchmark")
    print(f"{'#'*60}")
    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Model Directory: {model_path}")
    print(f"  Device: {device}")
    print(f"  Threads: {threads}")
    print(f"  Test files: {len(test_files)}")
    
    # Load pipeline
    print(f"\nLoading pipeline...")
    config = {
        "INFERENCE_NUM_THREADS": str(threads),
        "PERFORMANCE_HINT": "THROUGHPUT"
    }
    pipeline = ov_genai.WhisperPipeline(model_path, device, **config)
    print("Pipeline loaded!")
    
    # Run benchmarks
    results = []
    for audio_file in test_files:
        try:
            result = benchmark_file(pipeline, audio_file, device, threads)
            results.append(result)
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            continue
    
    # Summary
    print(f"\n{'#'*60}")
    print(f"# BENCHMARK SUMMARY")
    print(f"{'#'*60}\n")
    
    if results:
        avg_rtf = sum(r["real_time_factor"] for r in results) / len(results)
        avg_speed = sum(r["speed_multiplier"] for r in results) / len(results)
        total_audio = sum(r["duration_seconds"] for r in results)
        total_processing = sum(r["processing_time_seconds"] for r in results)
        
        print(f"Files processed:        {len(results)}")
        print(f"Total audio duration:   {total_audio:.2f}s")
        print(f"Total processing time:  {total_processing:.2f}s")
        print(f"Average RTF:            {avg_rtf:.2f}x")
        print(f"Average speed:          {avg_speed:.2f}x real-time")
        print(f"\nInterpretation:")
        print(f"  - RTF < 1.0 means faster than real-time")
        print(f"  - Speed > 1.0 means processing faster than playback")
        print(f"  - Your system processes audio {avg_speed:.1f}x faster than real-time")
        
        # Save results
        output_file = f"benchmark_results_{model_name.replace(' ', '_').replace('/', '_')}.json"
        with open(output_file, 'w') as f:
            json.dump({
                "model": model_name,
                "model_directory": model_path,
                "device": device,
                "threads": threads,
                "summary": {
                    "files_processed": len(results),
                    "total_audio_seconds": round(total_audio, 2),
                    "total_processing_seconds": round(total_processing, 2),
                    "average_rtf": round(avg_rtf, 2),
                    "average_speed_multiplier": round(avg_speed, 2)
                },
                "details": results
            }, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    print(f"\n{'#'*60}\n")


if __name__ == "__main__":
    main()
