#!/usr/bin/env python3
"""
Model Setup Script for Whisper-Fast-CPU-OpenVINO
Allows selection and download of different Whisper model variants
"""

import os
import sys
import argparse
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: huggingface_hub is not installed")
    print("Install it with: pip install huggingface_hub")
    sys.exit(1)


# Available models with their specifications
AVAILABLE_MODELS = {
    "int8-turbo": {
        "name": "whisper-large-v3-turbo-int8",
        "repo_id": "OpenVINO/whisper-large-v3-turbo-int8-ov",
        "description": "Whisper Large V3 Turbo with INT8 quantization (balanced speed/accuracy)",
        "size": "~1.5 GB",
        "speed": "Fast (1.5-2.0x real-time)",
        "accuracy": "High",
        "recommended": True,
        "hardware": "Medium to High-end CPUs",
    },
    "int8-lite": {
        "name": "whisper-large-v3-turbo-int8-lite",
        "repo_id": "bweng/whisper-large-v3-turbo-int8-ov",
        "description": "Whisper Large V3 Turbo INT8 optimized for weaker hardware",
        "size": "~1.5 GB",
        "speed": "Moderate (1.2-1.8x real-time)",
        "accuracy": "High",
        "recommended": False,
        "hardware": "Low to Medium-end CPUs",
    },
    "int4": {
        "name": "whisper-large-v3-int4",
        "repo_id": "OpenVINO/whisper-large-v3-int4-ov",
        "description": "Whisper Large V3 with INT4 quantization (maximum speed)",
        "size": "~800 MB",
        "speed": "Very Fast (2.0-3.0x real-time)",
        "accuracy": "Good (slight degradation vs INT8)",
        "recommended": False,
        "hardware": "All CPUs (especially resource-constrained)",
    },
}


def print_banner():
    """Print setup banner"""
    print("\n" + "=" * 70)
    print("  Whisper-Fast-CPU-OpenVINO - Model Setup")
    print("=" * 70 + "\n")


def check_hf_cli():
    """Check if HuggingFace CLI is installed"""
    try:
        import subprocess
        result = subprocess.run(['huggingface-cli', '--version'], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def auto_download_if_needed(target_dir="model"):
    """Auto-download model if it doesn't exist and HF CLI is available"""
    target_path = Path(target_dir)
    config_file = target_path / "generation_config.json"
    
    # Check if model already exists
    if config_file.exists():
        print(f"✓ Model already exists in '{target_dir}'")
        return True
    
    print(f"Model not found in '{target_dir}'")
    
    # Check if HF CLI is available
    if not check_hf_cli():
        print("\n⚠️  HuggingFace CLI not available.")
        print("\nManual setup required:")
        print("  1. Install: pip install huggingface_hub")
        print("  2. Run: python setup_model.py")
        return False
    
    print("\n✓ HuggingFace CLI detected")
    print("Starting auto-download of recommended model...\n")
    
    # Auto-download the recommended model
    recommended_key = next((k for k, v in AVAILABLE_MODELS.items() if v["recommended"]), "int8-turbo")
    return download_model(recommended_key, target_dir, force=False, interactive=False)


def list_models():
    """Display available models"""
    print("Available Models:\n")
    
    for idx, (key, model) in enumerate(AVAILABLE_MODELS.items(), 1):
        recommended = " [RECOMMENDED]" if model["recommended"] else ""
        print(f"{idx}. {model['name']}{recommended}")
        print(f"   Repository: {model['repo_id']}")
        print(f"   Description: {model['description']}")
        print(f"   Size: {model['size']}")
        print(f"   Speed: {model['speed']}")
        print(f"   Accuracy: {model['accuracy']}")
        print(f"   Hardware: {model['hardware']}")
        print()


def download_model(model_key: str, target_dir: str = "model", force: bool = False, interactive: bool = True):
    """Download the selected model"""
    if model_key not in AVAILABLE_MODELS:
        print(f"Error: Unknown model key '{model_key}'")
        print(f"Available keys: {', '.join(AVAILABLE_MODELS.keys())}")
        return False
    
    model = AVAILABLE_MODELS[model_key]
    target_path = Path(target_dir)
    
    # Check if model already exists
    if target_path.exists() and not force:
        config_file = target_path / "generation_config.json"
        if config_file.exists():
            if interactive:
                print(f"\n⚠️  Model directory '{target_dir}' already exists!")
                response = input("Do you want to overwrite it? (yes/no): ").strip().lower()
                if response not in ['yes', 'y']:
                    print("Aborted.")
                    return False
            else:
                print(f"Model already exists in '{target_dir}'")
                return True
    
    print(f"\n{'=' * 70}")
    print(f"Downloading: {model['name']}")
    print(f"Repository: {model['repo_id']}")
    print(f"Target: {target_dir}")
    print(f"Size: {model['size']}")
    print(f"{'=' * 70}\n")
    
    try:
        print("Starting download... (this may take several minutes)\n")
        snapshot_download(
            repo_id=model['repo_id'],
            local_dir=target_dir,
            local_dir_use_symlinks=False,
        )
        
        print(f"\n✅ Successfully downloaded {model['name']} to '{target_dir}'")
        
        # Verify essential files
        essential_files = [
            "generation_config.json",
            "config.json",
            "tokenizer.json",
            "openvino_encoder_model.xml",
            "openvino_encoder_model.bin",
            "openvino_decoder_model.xml",
            "openvino_decoder_model.bin",
        ]
        
        missing_files = []
        for file in essential_files:
            if not (target_path / file).exists():
                missing_files.append(file)
        
        if missing_files:
            print(f"\n⚠️  Warning: Some files are missing: {', '.join(missing_files)}")
            print("The model may not work correctly.")
            return False
        
        print("\n✅ All essential files verified!")
        
        # Save model info
        info_file = target_path / "MODEL_INFO.txt"
        with open(info_file, 'w') as f:
            f.write(f"Model: {model['name']}\n")
            f.write(f"Repository: {model['repo_id']}\n")
            f.write(f"Quantization: {model_key.upper()}\n")
            f.write(f"Description: {model['description']}\n")
            f.write(f"Downloaded: {Path.cwd()}\n")
        
        print(f"\n📝 Model info saved to: {info_file}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        import traceback
        traceback.print_exc()
        return False


def interactive_setup():
    """Interactive model selection and download"""
    print_banner()
    list_models()
    
    print("=" * 70)
    print("\nSelect a model to download:")
    print("  1 - INT8 Turbo (Recommended - balanced speed/accuracy, high-end CPUs)")
    print("  2 - INT8 Lite (Optimized for weaker hardware)")
    print("  3 - INT4 (Maximum speed, smallest size)")
    print("  q - Quit")
    print()
    
    choice = input("Enter your choice (1/2/3/q): ").strip().lower()
    
    if choice == 'q':
        print("Aborted.")
        return False
    
    model_map = {
        '1': 'int8-turbo',
        '2': 'int8-lite',
        '3': 'int4',
    }
    
    if choice not in model_map:
        print(f"Invalid choice: {choice}")
        return False
    
    model_key = model_map[choice]
    
    # Ask for custom directory
    print(f"\nDefault installation directory: ./model")
    custom_dir = input("Press Enter to use default, or enter custom path: ").strip()
    target_dir = custom_dir if custom_dir else "model"
    
    return download_model(model_key, target_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Setup and download Whisper models for OpenVINO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (recommended)
  python setup_model.py
  
  # Direct download INT8 model
  python setup_model.py --model int8
  
  # Download INT4 model to custom directory
  python setup_model.py --model int4 --target-dir my_model
  
  # List available models
  python setup_model.py --list
        """
    )
    
    parser.add_argument(
        '--model',
        choices=['int8-turbo', 'int8-lite', 'int4'],
        help='Model to download (int8-turbo, int8-lite, or int4)'
    )
    parser.add_argument(
        '--target-dir',
        default='model',
        help='Target directory for model files (default: model)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available models and exit'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force overwrite if model already exists'
    )
    parser.add_argument(
        '--auto',
        action='store_true',
        help='Auto-download recommended model if not present'
    )
    
    args = parser.parse_args()
    
    # List models and exit
    if args.list:
        print_banner()
        list_models()
        return 0
    
    # Auto-download mode
    if args.auto:
        print_banner()
        success = auto_download_if_needed(args.target_dir)
        return 0 if success else 1
    
    # Direct download mode
    if args.model:
        print_banner()
        success = download_model(args.model, args.target_dir, args.force)
        return 0 if success else 1
    
    # Interactive mode
    success = interactive_setup()
    
    if success:
        print("\n" + "=" * 70)
        print("🎉 Setup complete!")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Start the server:")
        print("     ./start_server.sh")
        print("     or")
        print("     python serve_whisper.py --device CPU --port 7860")
        print()
        print("  2. Open your browser:")
        print("     http://localhost:7860")
        print()
        print("  3. Run benchmarks (optional):")
        print("     python benchmark_simple.py")
        print("=" * 70 + "\n")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
