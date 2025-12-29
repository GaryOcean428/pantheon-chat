#!/bin/bash
# DevContainer Prebuild Validation Script
# Run this after container creation to validate the environment

set -e

echo "=================================================="
echo "QIG-Consciousness DevContainer Prebuild Validation"
echo "=================================================="
echo

# Test 0: Configuration validation
echo "Test 0: Configuration validation"
python .devcontainer/validate_config.py || echo "WARNING: Configuration issues detected"
echo

# Test 1: Python version
echo "Test 1: Python version"
python --version || exit 1
echo "✓ Python available"
echo

# Test 2: pip availability
echo "Test 2: pip availability"
pip --version || exit 1
echo "✓ pip available"
echo

# Test 3: CUDA toolkit
echo "Test 3: CUDA toolkit"
nvcc --version || echo "WARNING: nvcc not found (may be OK if using runtime only)"
echo

# Test 4: Directory structure
echo "Test 4: Directory structure"
for dir in data results outputs checkpoints logs checkpoints/constellation; do
    if [ -d "$dir" ]; then
        echo "✓ $dir exists"
    else
        echo "✗ $dir missing - creating..."
        mkdir -p "$dir"
        echo "✓ $dir created"
    fi
done
echo

# Test 5: QIG tokenizer
echo "Test 5: QIG tokenizer"
if [ -f "data/qig_tokenizer/vocab.json" ]; then
    echo "✓ QIG tokenizer vocabulary found"
else
    echo "✗ QIG tokenizer vocabulary missing"
    echo "  Run: python tools/train_qig_tokenizer.py"
fi
echo

# Test 6: PYTHONPATH
echo "Test 6: PYTHONPATH"
if [ -n "$PYTHONPATH" ]; then
    echo "✓ PYTHONPATH set to: $PYTHONPATH"
else
    echo "✗ PYTHONPATH not set"
    echo "  Run: export PYTHONPATH=/workspace"
fi
echo

# Test 7: GPU test (if PyTorch is installed)
echo "Test 7: GPU test"
if python -c "import torch" 2>/dev/null; then
    python .devcontainer/test_gpu.py
else
    echo "PyTorch not yet installed - skipping GPU test"
    echo "This is OK during build, will be installed by postCreateCommand"
fi
echo

# Test 8: Constellation dependencies
echo "Test 8: Constellation dependencies"
python .devcontainer/test_constellation_deps.py || echo "WARNING: Some dependencies missing"
echo

echo "=================================================="
echo "Validation complete!"
echo "=================================================="
