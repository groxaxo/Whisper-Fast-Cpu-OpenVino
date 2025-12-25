#!/bin/bash
# Start Dictation Client
# Global hotkey: Ctrl+Alt+Space

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Starting Dictation Client${NC}"
echo -e "${BLUE}Hotkey: Ctrl+Alt+Space${NC}"
echo -e "${YELLOW}Make sure the server is running on http://localhost:8000${NC}"
echo ""

# Check if server is running
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Warning: Server not responding at http://localhost:8000${NC}"
    echo -e "${YELLOW}   Start server first: ./start_server.sh${NC}"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Verify user is in input group
if ! groups | grep -q input; then
    echo -e "${YELLOW}⚠️  Warning: User not in 'input' group${NC}"
    echo -e "${YELLOW}   Run: sudo usermod -aG input \$USER && logout${NC}"
    echo ""
fi

# Start dictation client
/home/op/miniconda/envs/ov-whisper/bin/python dictation_client.py
