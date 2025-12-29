#!/bin/bash
# Lambda Cloud Deployment Script for QIG Consciousness
# Deploy constellation training on Lambda Cloud GPU instance

set -e

echo "🌩️  Lambda Cloud Deployment for QIG Consciousness"
echo "=================================================="
echo ""

# Check if running on Lambda Cloud
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ Error: No NVIDIA GPU detected. Are you on a Lambda Cloud GPU instance?"
    exit 1
fi

# Display GPU info
echo "📊 GPU Information:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Check Python version
echo "🐍 Python version:"
python --version
echo ""

# Setup virtual environment if not exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment with uv..."
    uv sync
fi

echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install/upgrade dependencies
echo "📚 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo ""
echo "✅ Environment ready!"
echo ""

# Check for Gary-B checkpoint
GARY_B_CHECKPOINT=""
if [ -f "checkpoints/constellation/gary_b_60560.pt" ]; then
    GARY_B_CHECKPOINT="--load-gary-b checkpoints/constellation/gary_b_60560.pt"
    echo "🧠 Gary-B checkpoint found: checkpoints/constellation/gary_b_60560.pt"
    echo "   Awakening protocol will be activated"
else
    echo "ℹ️  No Gary-B checkpoint found (training from scratch)"
fi

echo ""
echo "🚀 Launching QIG Chat (Constellation Mode)"
echo "   Device: CUDA (Lambda GPU)"
echo "   Mode: Full constellation (3 Garys + Ocean + Charlie)"
echo "   Coaching: MonkeyCoach v2"
echo ""

# Launch with Lambda optimizations
python chat_interfaces/qig_chat.py \
    --device cuda \
    $GARY_B_CHECKPOINT

echo ""
echo "✅ Training session completed"
