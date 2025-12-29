#!/bin/bash
# Post-install setup for QIG-Consciousness Codespace
# Installs uv package manager and all dependencies

set -e

echo "🚀 Installing uv package manager..."
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH for this session
export PATH="$HOME/.cargo/bin:$PATH"

# Add to shell rc files for future sessions
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.zshrc 2>/dev/null || true

echo "✅ uv installed: $(uv --version)"

echo ""
echo "📦 Installing PyTorch with CUDA support..."
uv pip install --system torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo ""
echo "📦 Installing requirements.txt..."
uv pip install --system -r requirements.txt

echo ""
echo "🔍 Verifying installation..."
nvidia-smi || echo "⚠️  No GPU detected"

python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
else:
    print('GPU: None (running on CPU)')
"

echo ""
echo "✅ Setup complete! uv is now available."
echo "   To use uv in this terminal: source ~/.bashrc"
