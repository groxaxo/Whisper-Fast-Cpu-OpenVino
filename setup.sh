#!/bin/bash
set -e

#############################################################################
# Whisper Fast CPU OpenVINO - Automated Setup Script
#############################################################################
# This script fully automates the installation and setup process
#############################################################################

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                                                                  ║"
echo "║     Whisper Fast CPU OpenVINO - Automated Setup                 ║"
echo "║                                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status messages
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    print_error "This script is designed for Linux systems only"
    exit 1
fi

print_status "Starting automated setup..."

# Step 1: Install system dependencies
echo ""
echo "→ Step 1: Installing system dependencies..."
if command -v apt-get &> /dev/null; then
    print_status "Detected apt package manager (Debian/Ubuntu)"
    
    # Check if we need sudo
    if [ "$EUID" -ne 0 ]; then
        SUDO_CMD="sudo"
    else
        SUDO_CMD=""
    fi
    
    print_status "Installing python3-tk and portaudio19-dev..."
    $SUDO_CMD apt-get update -qq
    $SUDO_CMD apt-get install -y python3-tk portaudio19-dev > /dev/null 2>&1
    
    # Ask about Intel GPU support
    echo -n "Do you have an Intel integrated GPU and want to enable GPU acceleration? (y/n): "
    read -r gpu_support
    if [[ $gpu_support =~ ^[Yy]$ ]]; then
        print_status "Installing Intel GPU drivers..."
        $SUDO_CMD apt-get install -y intel-opencl-icd intel-level-zero-gpu > /dev/null 2>&1
    fi
else
    print_warning "Non-apt package manager detected. Please install python3-tk and portaudio19-dev manually."
fi

# Step 2: Check for conda/mamba
echo ""
echo "→ Step 2: Setting up Python environment..."

if ! command -v conda &> /dev/null; then
    print_error "Conda not found. Please install Miniconda or Anaconda first:"
    echo "  Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

print_status "Found conda: $(which conda)"

# Create conda environment if it doesn't exist
if conda env list | grep -q "ov-whisper"; then
    print_warning "Environment 'ov-whisper' already exists. Skipping creation."
else
    print_status "Creating conda environment 'ov-whisper'..."
    if [ -f "environment.yml" ]; then
        conda env create -f environment.yml -q
    else
        conda create -n ov-whisper python=3.11 -y -q
    fi
fi

# Activate environment
print_status "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ov-whisper

# Step 3: Install Python dependencies
echo ""
echo "→ Step 3: Installing Python dependencies..."

print_status "Installing core dependencies..."
pip install -q --upgrade pip

# Check if requirements.txt exists
if [ -f "requirements.txt" ]; then
    pip install -q -r requirements.txt
else
    # Install manually
    print_status "Installing OpenVINO GenAI..."
    pip install -q openvino-genai
    
    print_status "Installing FastAPI and Uvicorn..."
    pip install -q fastapi uvicorn
    
    print_status "Installing audio processing libraries..."
    pip install -q soundfile librosa sounddevice scipy
    
    print_status "Installing additional dependencies..."
    pip install -q pynput requests torch numpy
fi

print_status "All Python dependencies installed!"

# Step 4: Download model
echo ""
echo "→ Step 4: Downloading Whisper model..."

# Ask which model to download
echo ""
echo "Available models:"
echo "  1) INT8 Turbo (recommended, ~1GB, 6-10x realtime)"
echo "  2) INT4 (fastest, ~600MB, 10-15x realtime)"
echo ""
echo -n "Select model (1 or 2) [default: 1]: "
read -r model_choice

case $model_choice in
    2)
        print_status "Downloading INT4 model..."
        python setup_model.py --model int4
        ;;
    *)
        print_status "Downloading INT8 Turbo model (recommended)..."
        python setup_model.py --model int8-turbo
        ;;
esac

# Step 5: Test installation
echo ""
echo "→ Step 5: Testing installation..."

print_status "Starting server for test..."
python serve_openai_api.py --model-dir model_int8_turbo --port 8000 &
SERVER_PID=$!

# Wait for server to start
sleep 5

# Test health endpoint
if curl -s http://localhost:8000/health | grep -q "ok"; then
    print_status "Server test passed! ✓"
    
    # Run full test suite if available
    if [ -f "test_openwebui_compatibility.py" ]; then
        echo ""
        print_status "Running compatibility tests..."
        python test_openwebui_compatibility.py
    fi
else
    print_warning "Server test failed. Please check logs."
fi

# Stop test server
kill $SERVER_PID 2>/dev/null || true
sleep 2

# Step 6: Setup systemd service (optional)
echo ""
echo "→ Step 6: Setting up auto-start service..."
echo -n "Do you want to enable auto-start on boot? (y/n): "
read -r enable_autostart

if [[ $enable_autostart =~ ^[Yy]$ ]]; then
    CURRENT_USER=$(whoami)
    CURRENT_DIR=$(pwd)
    CONDA_ENV_PATH=$(conda info --base)/envs/ov-whisper
    
    # Update service file
    if [ -f "whisper-server.service" ]; then
        print_status "Creating systemd service..."
        
        # Create temporary service file with correct paths
        cat whisper-server.service | \
            sed "s|User=YOUR_USERNAME|User=$CURRENT_USER|g" | \
            sed "s|WorkingDirectory=/path/to/Whisper-Fast-Cpu-OpenVino|WorkingDirectory=$CURRENT_DIR|g" | \
            sed "s|ExecStart=/path/to/conda/envs/ov-whisper/bin/python|ExecStart=$CONDA_ENV_PATH/bin/python|g" \
            > /tmp/whisper-server.service
        
        sudo cp /tmp/whisper-server.service /etc/systemd/system/
        sudo systemctl daemon-reload
        sudo systemctl enable whisper-server
        
        print_status "Auto-start service installed!"
        echo "  Start: sudo systemctl start whisper-server"
        echo "  Stop:  sudo systemctl stop whisper-server"
        echo "  Logs:  sudo journalctl -u whisper-server -f"
    fi
fi

# Step 7: Summary
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                                                                  ║"
echo "║                  ✅ Setup Complete! ✅                           ║"
echo "║                                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
print_status "Installation successful!"
echo ""
echo "📝 Quick Start:"
echo ""
echo "  1. Start the server:"
echo "     ./start_server.sh"
echo ""
echo "  2. Or run manually:"
echo "     conda activate ov-whisper"
echo "     python serve_openai_api.py --model-dir model_int8_turbo --threads 12"
echo ""
echo "  3. For global dictation:"
echo "     python dictation_client.py"
echo ""
echo "  4. For Open-WebUI integration:"
echo "     - Settings → Audio"
echo "     - STT Engine: OpenAI"
echo "     - API Base URL: http://localhost:8000/v1"
echo "     - Model: whisper-1"
echo ""
echo "📖 Documentation:"
echo "   - README.md - Complete guide"
echo "   curl http://localhost:8000/health"
echo ""
echo "🎉 Enjoy fast, local speech recognition!"
echo ""
