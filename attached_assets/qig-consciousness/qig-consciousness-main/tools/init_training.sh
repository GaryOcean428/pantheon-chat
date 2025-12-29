#!/bin/bash
###############################################################################
# QIG-Kernel Training Initialization Script
# =========================================
#
# Prepares environment and launches training for β-function validation.
#
# This script:
# 1. Validates Python environment
# 2. Checks/installs dependencies
# 3. Validates architecture (6/6 checks)
# 4. Prepares dataset (if needed)
# 5. Launches training with monitoring
# 6. Measures β-function after training
# 7. Generates report
#
# Usage:
#   bash tools/init_training.sh [--install-deps] [--skip-training]
#
# Budget: ~$100 (10-20 hours compute)
# Target: β_attention ≈ 0.44 (matches physics)
###############################################################################

set -e  # Exit on error

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "======================================================================"
echo "QIG-KERNEL TRAINING INITIALIZATION"
echo "======================================================================"
echo ""
echo "Mission: Measure β_attention to test information geometry unification"
echo "Target:  β_attention ≈ 0.44 (matching physics β-function)"
echo "Budget:  \$100 maximum"
echo ""

# Parse arguments
INSTALL_DEPS=false
SKIP_TRAINING=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --install-deps)
            INSTALL_DEPS=true
            shift
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

###############################################################################
# Step 1: Python Environment Check
###############################################################################

echo "Step 1: Checking Python environment..."
echo "----------------------------------------------------------------------"

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "✅ Python $PYTHON_VERSION found"
echo ""

###############################################################################
# Step 2: Dependencies
###############################################################################

echo "Step 2: Checking dependencies..."
echo "----------------------------------------------------------------------"

check_package() {
    python3 -c "import $1" 2>/dev/null && echo "  ✅ $1" || echo "  ❌ $1 (missing)"
}

check_package torch
check_package transformers
check_package numpy
check_package scipy

if [ "$INSTALL_DEPS" = true ]; then
    echo ""
    echo "Installing dependencies..."
    pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install -q transformers datasets numpy scipy matplotlib
    echo "✅ Dependencies installed"
fi

echo ""

###############################################################################
# Step 3: Architecture Validation
###############################################################################

echo "Step 3: Validating architecture..."
echo "----------------------------------------------------------------------"

if python3 tools/validation/validate_architecture.py; then
    echo ""
    echo "✅ Architecture validation passed (6/6 checks)"
else
    echo "❌ Architecture validation failed"
    exit 1
fi

echo ""

###############################################################################
# Step 4: Dataset Preparation
###############################################################################

echo "Step 4: Checking dataset..."
echo "----------------------------------------------------------------------"

if [ ! -d "data/conversations/train" ]; then
    echo "Dataset not found. Creating sample dataset..."
    python3 tools/data_prep/prepare_dataset.py --output data/conversations --create-sample
    echo "✅ Sample dataset created"
else
    TRAIN_COUNT=$(find data/conversations/train -type f | wc -l)
    echo "✅ Dataset found: $TRAIN_COUNT training conversations"
fi

echo ""

###############################################################################
# Step 5: Training Configuration
###############################################################################

echo "Step 5: Training configuration..."
echo "----------------------------------------------------------------------"
echo "  Model: QIG-Kernel-Recursive (100M parameters)"
echo "  Epochs: 10"
echo "  Batch size: 4"
echo "  Learning rate: 1e-4"
echo "  Loss: LM + basin alignment + Φ regularization"
echo ""
echo "  Target metrics:"
echo "    - Basin distance < 0.15"
echo "    - Integration Φ > 0.7"
echo "    - β_attention ≈ 0.44 ± 0.1"
echo ""

if [ "$SKIP_TRAINING" = true ]; then
    echo "⚠️  Skipping training (--skip-training flag)"
    echo ""
    exit 0
fi

###############################################################################
# Step 6: Launch Training
###############################################################################

echo "Step 6: Launching training..."
echo "----------------------------------------------------------------------"
echo ""
echo "⚠️  NOTE: Training requires PyTorch and GPU (optional but recommended)"
echo "    Estimated time: 10-20 hours (depending on hardware)"
echo "    Estimated cost: \$80-100 (if using cloud GPU)"
echo ""

read -p "Start training now? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting training..."
    echo ""

    # Create output directory
    mkdir -p outputs/qig_kernel_run1

    # Launch training
    python3 tools/training/train_qig_kernel.py \
        --data-dir data/conversations \
        --output-dir outputs/qig_kernel_run1 \
        --epochs 10 \
        --batch-size 4 \
        2>&1 | tee outputs/qig_kernel_run1/training.log

    echo ""
    echo "✅ Training complete"
    echo ""

    ###########################################################################
    # Step 7: Measure β-function
    ###########################################################################

    echo "Step 7: Measuring β-function..."
    echo "----------------------------------------------------------------------"
    echo ""

    if [ -f "outputs/qig_kernel_run1/best_model.pt" ]; then
        python3 tools/measure_beta_function.py \
            --model-path outputs/qig_kernel_run1/best_model.pt \
            --context-lengths 512,1024,2048,4096,8192 \
            --output outputs/qig_kernel_run1/beta_measurement.json \
            --plot outputs/qig_kernel_run1/beta_function_fit.png

        echo ""
        echo "✅ β-function measurement complete"
        echo ""

        # Display results
        if [ -f "outputs/qig_kernel_run1/beta_measurement.json" ]; then
            echo "======================================================================"
            echo "RESULTS SUMMARY"
            echo "======================================================================"
            cat outputs/qig_kernel_run1/beta_measurement.json | python3 -m json.tool
            echo ""
        fi
    else
        echo "❌ Model checkpoint not found. Training may have failed."
        exit 1
    fi
else
    echo ""
    echo "Training cancelled. You can start training manually with:"
    echo ""
    echo "  python3 tools/training/train_qig_kernel.py --data-dir data/conversations"
    echo ""
fi

###############################################################################
# Final Summary
###############################################################################

echo "======================================================================"
echo "INITIALIZATION COMPLETE"
echo "======================================================================"
echo ""
echo "Next steps:"
echo ""
echo "  1. Review training logs: outputs/qig_kernel_run1/training.log"
echo "  2. Check β measurement: outputs/qig_kernel_run1/beta_measurement.json"
echo "  3. View plot: outputs/qig_kernel_run1/beta_function_fit.png"
echo "  4. Report β_attention value to determine if it matches physics (0.44)"
echo ""
echo "Success criteria:"
echo "  🎯 MAJOR:    |β_attention - 0.44| < 0.1  (information geometry unifies!)"
echo "  🎯 MODERATE: |β_attention - 0.44| < 0.2  (qualitative agreement)"
echo "  🎯 NULL:     |β_attention - 0.44| > 0.3  (distinct behavior, still valuable)"
echo ""
echo "Basin stable. Math validated. Ready to measure. 🌊💚"
echo ""
