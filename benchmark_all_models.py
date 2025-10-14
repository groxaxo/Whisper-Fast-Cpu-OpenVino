#!/usr/bin/env python3
"""
Benchmark all available Whisper models
Tests each model and creates comprehensive comparison
"""

import os
import sys
import json
import time
from pathlib import Path
import subprocess

# Model configurations
MODELS = {
    "int8-turbo": {
        "dir": "model",
        "name": "INT8-Turbo (OpenVINO)",
        "repo": "OpenVINO/whisper-large-v3-turbo-int8-ov"
    },
    "int8-lite": {
        "dir": "model_int8_lite",
        "name": "INT8-Lite (bweng)",
        "repo": "bweng/whisper-large-v3-turbo-int8-ov"
    },
    "int4": {
        "dir": "model_int4",
        "name": "INT4 (OpenVINO)",
        "repo": "OpenVINO/whisper-large-v3-int4-ov"
    }
}

# Test files
TEST_FILES = [
    ("/home/op/ov-whisper/sample.wav", "Sample Audio"),
    ("/home/op/neutts-air/neutts-air/samples/dave.wav", "Dave Sample"),
    ("/home/op/happiness_book_output/chapter_01.wav", "Chapter 01"),
]


def check_model_exists(model_dir):
    """Check if model is downloaded"""
    config_file = Path(model_dir) / "generation_config.json"
    return config_file.exists()


def download_model(model_key, model_dir):
    """Download model if not present"""
    if check_model_exists(model_dir):
        print(f"✓ {MODELS[model_key]['name']} already downloaded")
        return True
    
    print(f"\nDownloading {MODELS[model_key]['name']}...")
    print(f"Repository: {MODELS[model_key]['repo']}")
    print("This may take several minutes...\n")
    
    try:
        result = subprocess.run(
            [sys.executable, "setup_model.py", "--model", model_key, "--target-dir", model_dir],
            timeout=600
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"✗ Download timeout for {model_key}")
        return False
    except Exception as e:
        print(f"✗ Error downloading {model_key}: {e}")
        return False


def run_benchmark(model_dir, threads=8):
    """Run benchmark for a specific model"""
    # Filter existing test files
    existing_files = [f for f, _ in TEST_FILES if os.path.exists(f)]
    
    if not existing_files:
        print("No test files found!")
        return None
    
    print(f"\nRunning benchmark with {len(existing_files)} test files...")
    
    try:
        result = subprocess.run(
            [sys.executable, "benchmark_simple.py", "--model-dir", model_dir, "--threads", str(threads)],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode != 0:
            print(f"Benchmark failed: {result.stderr}")
            return None
        
        # Find the output JSON file
        model_name = Path(model_dir).name
        result_files = list(Path(".").glob(f"benchmark_results_*.json"))
        
        if result_files:
            latest_result = max(result_files, key=lambda p: p.stat().st_mtime)
            with open(latest_result) as f:
                return json.load(f)
        
        return None
        
    except subprocess.TimeoutExpired:
        print("Benchmark timeout!")
        return None
    except Exception as e:
        print(f"Benchmark error: {e}")
        return None


def format_results_table(all_results):
    """Format results as comparison table"""
    print("\n" + "=" * 90)
    print("  BENCHMARK RESULTS COMPARISON")
    print("=" * 90 + "\n")
    
    # Header
    print(f"{'Model':<25} {'Avg Speed':<15} {'Avg RTF':<15} {'Files':<10} {'Total Audio':<15}")
    print("-" * 90)
    
    for model_key, result in all_results.items():
        if result:
            model_name = MODELS[model_key]["name"]
            summary = result.get("summary", {})
            avg_speed = summary.get("average_speed_multiplier", 0)
            avg_rtf = summary.get("average_rtf", 0)
            files = summary.get("files_processed", 0)
            total_audio = summary.get("total_audio_seconds", 0)
            
            print(f"{model_name:<25} {f'{avg_speed:.2f}x RT':<15} {f'{avg_rtf:.3f}x':<15} "
                  f"{files:<10} {f'{total_audio:.1f}s':<15}")
    
    print("\n" + "=" * 90 + "\n")


def save_comparison_json(all_results):
    """Save comprehensive comparison to JSON"""
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system": "Intel Core i7-12700KF (12th Gen)",
        "models": {}
    }
    
    for model_key, result in all_results.items():
        if result:
            output["models"][model_key] = {
                "name": MODELS[model_key]["name"],
                "repository": MODELS[model_key]["repo"],
                "results": result
            }
    
    output_file = "benchmark_comparison_all_models.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✓ Comprehensive results saved to: {output_file}\n")


def main():
    print("\n" + "=" * 90)
    print("  Whisper Models Benchmark Suite")
    print("=" * 90 + "\n")
    
    # Check and download models
    print("Step 1: Checking models...\n")
    available_models = {}
    
    for model_key, model_info in MODELS.items():
        model_dir = model_info["dir"]
        if download_model(model_key, model_dir):
            available_models[model_key] = model_dir
        else:
            print(f"⚠  Skipping {model_info['name']} (not available)")
    
    if not available_models:
        print("\n✗ No models available for benchmarking!")
        return 1
    
    print(f"\n✓ {len(available_models)} models ready for benchmarking\n")
    
    # Run benchmarks
    print("Step 2: Running benchmarks...\n")
    all_results = {}
    
    for model_key, model_dir in available_models.items():
        print(f"\n{'=' * 90}")
        print(f"  Benchmarking: {MODELS[model_key]['name']}")
        print(f"  Directory: {model_dir}")
        print(f"{'=' * 90}")
        
        result = run_benchmark(model_dir)
        all_results[model_key] = result
        
        if result:
            summary = result.get("summary", {})
            print(f"\n✓ Completed - Speed: {summary.get('average_speed_multiplier', 0):.2f}x RT")
        else:
            print(f"\n✗ Benchmark failed for {model_key}")
    
    # Display comparison
    print("\nStep 3: Results Summary\n")
    format_results_table(all_results)
    
    # Save comprehensive results
    save_comparison_json(all_results)
    
    print("=" * 90)
    print("  Benchmark suite completed!")
    print("=" * 90 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
