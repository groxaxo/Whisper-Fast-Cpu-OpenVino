#!/usr/bin/env python3
"""
Model Comparison Tool
Compare INT8 vs INT4 models side-by-side
"""

import sys
from pathlib import Path

# Model specifications
MODELS = {
    "INT8-Turbo": {
        "name": "whisper-large-v3-turbo-int8",
        "repo": "OpenVINO/whisper-large-v3-turbo-int8-ov",
        "size_gb": 1.5,
        "speed_min": 1.5,
        "speed_max": 2.0,
        "accuracy": 100,
        "memory_gb": 2.0,
        "hardware": "Medium to High-end CPUs",
        "recommended": True,
        "pros": [
            "Best accuracy (same as original Whisper)",
            "Balanced speed/accuracy tradeoff",
            "Recommended for production use",
            "Stable and well-tested"
        ],
        "cons": [
            "Larger download size",
            "Slightly slower than INT4"
        ],
        "best_for": [
            "General transcription tasks",
            "Production deployments",
            "When accuracy is critical",
            "High-end systems with adequate storage"
        ]
    },
    "INT8-Lite": {
        "name": "whisper-large-v3-turbo-int8-lite",
        "repo": "bweng/whisper-large-v3-turbo-int8-ov",
        "size_gb": 1.5,
        "speed_min": 1.2,
        "speed_max": 1.8,
        "accuracy": 100,
        "memory_gb": 1.8,
        "hardware": "Low to Medium-end CPUs",
        "recommended": False,
        "pros": [
            "Same accuracy as Turbo variant",
            "Optimized for weaker hardware",
            "Lower memory pressure",
            "Better stability on older CPUs"
        ],
        "cons": [
            "Slower than Turbo variant",
            "Still requires 1.5GB storage"
        ],
        "best_for": [
            "Older or weaker CPUs",
            "Systems with <4 cores",
            "Embedded systems",
            "When INT4 accuracy isn't sufficient"
        ]
    },
    "INT4": {
        "name": "whisper-large-v3-int4",
        "repo": "OpenVINO/whisper-large-v3-int4-ov",
        "size_gb": 0.8,
        "speed_min": 2.0,
        "speed_max": 3.0,
        "accuracy": 96,
        "memory_gb": 1.5,
        "recommended": False,
        "pros": [
            "Fastest processing speed",
            "Smallest model size",
            "Lower memory usage",
            "Great for embedded systems"
        ],
        "cons": [
            "Slight accuracy degradation (2-5%)",
            "May struggle with accents/noise",
            "Less tested than INT8"
        ],
        "best_for": [
            "Speed-critical applications",
            "Resource-constrained systems",
            "Real-time streaming",
            "Batch processing large datasets"
        ]
    }
}


def print_header():
    """Print comparison header"""
    print("\n" + "=" * 80)
    print("  Whisper Model Comparison: INT8-Turbo vs INT8-Lite vs INT4")
    print("=" * 80 + "\n")


def print_specs():
    """Print technical specifications"""
    print("📊 Technical Specifications\n")
    print(f"{'Specification':<20} {'INT8-Turbo':<18} {'INT8-Lite':<18} {'INT4':<18}")
    print("-" * 80)
    
    specs = [
        ("Model Name", "name"),
        ("Download Size", lambda m: f"~{m['size_gb']} GB"),
        ("Speed Range", lambda m: f"{m['speed_min']}-{m['speed_max']}x RT"),
        ("Accuracy", lambda m: f"{m['accuracy']}%"),
        ("Memory Usage", lambda m: f"~{m['memory_gb']} GB"),
        ("Hardware", "hardware"),
        ("Recommended", lambda m: "✅" if m['recommended'] else " "),
    ]
    
    for label, key in specs:
        turbo = MODELS["INT8-Turbo"][key] if isinstance(key, str) else key(MODELS["INT8-Turbo"])
        lite = MODELS["INT8-Lite"][key] if isinstance(key, str) else key(MODELS["INT8-Lite"])
        int4 = MODELS["INT4"][key] if isinstance(key, str) else key(MODELS["INT4"])
        print(f"{label:<20} {str(turbo):<18} {str(lite):<18} {str(int4):<18}")
    
    print()


def print_pros_cons():
    """Print pros and cons"""
    print("✅ Pros & ❌ Cons\n")
    
    for model_key in ["INT8-Turbo", "INT8-Lite", "INT4"]:
        model = MODELS[model_key]
        print(f"{'─' * 40}")
        print(f"{model_key} Model")
        print(f"{'─' * 40}")
        
        print("\n✅ Pros:")
        for pro in model['pros']:
            print(f"  • {pro}")
        
        print("\n❌ Cons:")
        for con in model['cons']:
            print(f"  • {con}")
        
        print()


def print_use_cases():
    """Print best use cases"""
    print("🎯 Best Use Cases\n")
    
    for model_key in ["INT8-Turbo", "INT8-Lite", "INT4"]:
        model = MODELS[model_key]
        print(f"{model_key} Model:")
        for use_case in model['best_for']:
            print(f"  • {use_case}")
        print()


def print_performance_example():
    """Print performance examples (estimated)\n"""  
    print("⚡ Performance Examples (Estimated)\n")
    
    durations = [60, 300, 3600]  # 1 min, 5 min, 1 hour
    
    print(f"{'Audio':<12} {'INT8-Turbo':<18} {'INT8-Lite':<18} {'INT4':<18}")
    print("-" * 70)
    
    for duration in durations:
        if duration < 60:
            duration_str = f"{duration}s"
        elif duration < 3600:
            duration_str = f"{duration // 60} minutes"
        else:
            duration_str = f"{duration // 3600} hour"
        
        # Calculate processing times (using average speed)
        turbo_speed = (MODELS["INT8-Turbo"]["speed_min"] + MODELS["INT8-Turbo"]["speed_max"]) / 2
        lite_speed = (MODELS["INT8-Lite"]["speed_min"] + MODELS["INT8-Lite"]["speed_max"]) / 2
        int4_speed = (MODELS["INT4"]["speed_min"] + MODELS["INT4"]["speed_max"]) / 2
        
        turbo_time = duration / turbo_speed
        lite_time = duration / lite_speed
        int4_time = duration / int4_speed
        
        def format_time(seconds):
            if seconds < 60:
                return f"{seconds:.1f}s"
            elif seconds < 3600:
                return f"{seconds / 60:.1f} min"
            else:
                return f"{seconds / 3600:.1f} hours"
        
        print(f"{duration_str:<12} {format_time(turbo_time):<18} {format_time(lite_time):<18} {format_time(int4_time):<18}")
    
    print()


def print_recommendation():
    """Print recommendation"""
    print("💡 Recommendation\n")
    print("Choose INT8-Turbo if:")
    print("  • You have a modern CPU (4+ cores, 3+ GHz)")
    print("  • You need balanced speed and accuracy")
    print("  • You're deploying to production")
    print("  • ⭐ RECOMMENDED for most users")
    print()
    print("Choose INT8-Lite if:")
    print("  • You have an older or weaker CPU (<4 cores)")
    print("  • INT8-Turbo is too slow or unstable")
    print("  • You need full accuracy but have limited resources")
    print("  • INT4 accuracy isn't sufficient")
    print()
    print("Choose INT4 if:")
    print("  • You need maximum speed (2.0-3.0x real-time)")
    print("  • Storage is very limited (only 800 MB)")
    print("  • You're processing large batches")
    print("  • Slight accuracy loss (4-5%) is acceptable")
    print()


def print_installation():
    """Print installation commands"""
    print("📦 Installation Commands\n")
    print("INT8-Turbo (Recommended):")
    print("  python setup_model.py --model int8-turbo")
    print()
    print("INT8-Lite (For Weaker Hardware):")
    print("  python setup_model.py --model int8-lite")
    print()
    print("INT4 (Maximum Speed):")
    print("  python setup_model.py --model int4")
    print()
    print("Auto-download (if HF CLI available):")
    print("  python setup_model.py --auto")
    print()
    print("Interactive Selection:")
    print("  python setup_model.py")
    print()


def check_installed_models():
    """Check which models are installed"""
    print("📁 Installed Models\n")
    
    model_dirs = {
        "model": "Default location",
        "model_int8_turbo": "INT8-Turbo model",
        "model_int8_lite": "INT8-Lite model",
        "model_int4": "INT4 model"
    }
    
    found_any = False
    for dir_name, description in model_dirs.items():
        dir_path = Path(dir_name)
        if dir_path.exists():
            config_file = dir_path / "generation_config.json"
            info_file = dir_path / "MODEL_INFO.txt"
            
            if config_file.exists():
                found_any = True
                print(f"✅ {dir_name}/ - {description}")
                
                if info_file.exists():
                    with open(info_file, 'r') as f:
                        for line in f:
                            if line.startswith('Model:'):
                                print(f"   {line.strip()}")
                                break
            else:
                print(f"⚠️  {dir_name}/ - Incomplete (missing config)")
    
    if not found_any:
        print("❌ No models found. Run: python setup_model.py")
    
    print()


def main():
    """Main comparison display"""
    print_header()
    
    # Check for installed models
    check_installed_models()
    
    # Show comparison
    print_specs()
    print_pros_cons()
    print_use_cases()
    print_performance_example()
    print_recommendation()
    print_installation()
    
    print("=" * 80)
    print("\nFor detailed setup instructions, see: SETUP_GUIDE.md")
    print("For benchmarking, run: python benchmark_simple.py --model-dir <model_dir>")
    print()


if __name__ == "__main__":
    main()
