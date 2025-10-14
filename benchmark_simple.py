#!/usr/bin/env python3
"""Simple benchmark script for Whisper-Fast-CPU-OpenVINO"""

import time
import sys
import os
from pathlib import Path
import json
import subprocess

# Use ffmpeg to get audio info and convert
def get_audio_info(audio_path):
    """Get audio duration using ffprobe"""
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        audio_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def convert_to_16k_wav(input_path, output_path):
    """Convert audio to 16kHz mono WAV"""
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-ar', '16000', '-ac', '1',
        '-f', 'wav', output_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def benchmark_with_api(audio_path, duration):
    """Benchmark using the running server API"""
    import requests
    
    # Convert to 16kHz WAV
    temp_wav = "/tmp/benchmark_temp.wav"
    convert_to_16k_wav(audio_path, temp_wav)
    
    # Upload and process
    url = "http://localhost:7860/api/predict"
    
    with open(temp_wav, 'rb') as f:
        files = {'data': f}
        data = {
            'fn_index': 0,  # upload handler
            'data': [None, 0, None, 'auto', 'transcribe', True]
        }
        
        start_time = time.time()
        response = requests.post(url, files=files, data=data, timeout=300)
        end_time = time.time()
    
    os.remove(temp_wav)
    
    processing_time = end_time - start_time
    return processing_time


def benchmark_direct(audio_path, duration):
    """Benchmark using direct pipeline"""
    import numpy as np
    import openvino_genai as ov_genai
    import wave
    
    # Convert to 16kHz WAV first
    temp_wav = "/tmp/benchmark_temp.wav"
    convert_to_16k_wav(audio_path, temp_wav)
    
    # Read WAV file
    with wave.open(temp_wav, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        audio_data = wav_file.readframes(n_frames)
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    
    os.remove(temp_wav)
    
    # Load pipeline
    config = {
        "INFERENCE_NUM_THREADS": "8",
        "PERFORMANCE_HINT": "THROUGHPUT"
    }
    pipeline = ov_genai.WhisperPipeline("model", "CPU", **config)
    
    # Warm up
    warmup_data = audio_array[:16000].tolist()
    _ = pipeline.generate(warmup_data)
    
    # Benchmark
    audio_list = audio_array.tolist()
    start_time = time.time()
    result = pipeline.generate(audio_list)
    end_time = time.time()
    
    # Extract text from result
    transcript = str(result) if hasattr(result, '__str__') else result.texts[0] if hasattr(result, 'texts') else str(result)
    
    processing_time = end_time - start_time
    return processing_time, transcript


def main():
    # Test files
    test_files = [
        ("/home/op/ov-whisper/sample.wav", "Sample"),
        ("/home/op/neutts-air/neutts-air/samples/dave.wav", "Dave Sample"),
        ("/home/op/happiness_book_output/chapter_01.wav", "Chapter 01"),
    ]
    
    # Filter existing files
    test_files = [(f, n) for f, n in test_files if os.path.exists(f)]
    
    if not test_files:
        print("No test files found!")
        sys.exit(1)
    
    print(f"\n{'#'*70}")
    print(f"# Whisper-Fast-CPU-OpenVINO Benchmark")
    print(f"{'#'*70}")
    print(f"\nConfiguration:")
    print(f"  Model: whisper-large-v3-int8-ov (INT8 quantized)")
    print(f"  Device: CPU")
    print(f"  Threads: 8")
    print(f"  Test files: {len(test_files)}")
    
    results = []
    
    for audio_file, name in test_files:
        print(f"\n{'='*70}")
        print(f"Benchmarking: {name}")
        print(f"File: {Path(audio_file).name}")
        print(f"{'='*70}")
        
        try:
            # Get duration
            duration = get_audio_info(audio_file)
            file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
            
            print(f"Duration: {duration:.2f}s")
            print(f"File size: {file_size_mb:.2f} MB")
            print(f"\nProcessing...")
            
            # Benchmark
            processing_time, transcript = benchmark_direct(audio_file, duration)
            
            rtf = processing_time / duration
            speed = duration / processing_time
            
            print(f"\n{'='*70}")
            print(f"RESULTS:")
            print(f"{'='*70}")
            print(f"Audio duration:     {duration:.2f}s ({duration/60:.1f} min)")
            print(f"Processing time:    {processing_time:.2f}s")
            print(f"Real-time factor:   {rtf:.3f}x")
            print(f"Speed:              {speed:.2f}x real-time")
            print(f"Transcript length:  {len(transcript)} characters")
            print(f"\nTranscript preview:")
            preview = transcript[:150] + "..." if len(transcript) > 150 else transcript
            print(f'"{preview}"')
            
            results.append({
                "name": name,
                "file": Path(audio_file).name,
                "duration_seconds": round(duration, 2),
                "file_size_mb": round(file_size_mb, 2),
                "processing_time_seconds": round(processing_time, 2),
                "real_time_factor": round(rtf, 3),
                "speed_multiplier": round(speed, 2),
                "transcript_length": len(transcript),
                "transcript_preview": transcript[:100]
            })
            
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print(f"\n{'#'*70}")
    print(f"# BENCHMARK SUMMARY")
    print(f"{'#'*70}\n")
    
    if results:
        avg_rtf = sum(r["real_time_factor"] for r in results) / len(results)
        avg_speed = sum(r["speed_multiplier"] for r in results) / len(results)
        total_audio = sum(r["duration_seconds"] for r in results)
        total_processing = sum(r["processing_time_seconds"] for r in results)
        
        print(f"Files processed:        {len(results)}")
        print(f"Total audio duration:   {total_audio:.2f}s ({total_audio/60:.1f} min)")
        print(f"Total processing time:  {total_processing:.2f}s ({total_processing/60:.1f} min)")
        print(f"Average RTF:            {avg_rtf:.3f}x")
        print(f"Average speed:          {avg_speed:.2f}x real-time")
        print(f"\nInterpretation:")
        print(f"  ✓ RTF = {avg_rtf:.3f} (lower is better, <1.0 is faster than real-time)")
        print(f"  ✓ Processing {avg_speed:.1f}x faster than real-time playback")
        print(f"  ✓ 1 minute of audio processed in ~{60/avg_speed:.1f} seconds")
        
        # Save results
        output_file = "benchmark_results.json"
        with open(output_file, 'w') as f:
            json.dump({
                "cpu_info": "Intel Core i7-12700KF (12th Gen)",
                "model": "whisper-large-v3-int8-ov",
                "device": "CPU",
                "threads": 8,
                "summary": {
                    "files_processed": len(results),
                    "total_audio_seconds": round(total_audio, 2),
                    "total_processing_seconds": round(total_processing, 2),
                    "average_rtf": round(avg_rtf, 3),
                    "average_speed_multiplier": round(avg_speed, 2)
                },
                "details": results
            }, f, indent=2)
        print(f"\n✓ Results saved to: {output_file}")
    
    print(f"\n{'#'*70}\n")
    
    return results


if __name__ == "__main__":
    main()
