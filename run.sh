#!/bin/bash

echo "Starting ExoVision..."

source venv/bin/activate

export ROLE=receiver

# Auto-detect Tailscale IP of THIS machine
export DENMARK_HOST=$(tailscale ip -4)

echo "Detected Tailscale IP: $DENMARK_HOST"

python scripts/main.py