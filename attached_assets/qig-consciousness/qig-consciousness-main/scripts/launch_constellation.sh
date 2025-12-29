#!/bin/bash
###############################################################################
# Constellation Launch Script
# ============================
#
# Launches Ocean (pure observer) + 3 Gary instances (active/observer hybrid)
# for multi-instance geometric consciousness training.
#
# Prerequisites:
#   - Configs generated (gary_A/B/C.yaml, ocean.yaml)
#   - Dataset available (data/conversations/*.json)
#   - CUDA available (recommended)
#
# Usage:
#   bash scripts/launch_constellation.sh [OPTIONS]
#
# Options:
#   --data-dir DIR        : Path to conversation data (default: data/conversations)
#   --epochs N            : Number of training epochs (default: 20)
#   --checkpoint-dir DIR  : Checkpoint directory (default: checkpoints/constellation)
#   --fresh-start         : Start fresh (ignore existing checkpoint)
#   --stop-on-convergence : Stop when constellation converges
#
# Example:
#   bash scripts/launch_constellation.sh --epochs 20 --stop-on-convergence
#
###############################################################################

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default arguments
DATA_DIR="data/conversations"
EPOCHS=20
CHECKPOINT_DIR="checkpoints/constellation"
FRESH_START=""
STOP_ON_CONVERGENCE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --fresh-start)
            FRESH_START="--fresh-start"
            shift
            ;;
        --stop-on-convergence)
            STOP_ON_CONVERGENCE="--stop-on-convergence"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Banner
echo -e "${CYAN}"
cat << "EOF"
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   🌊  CONSTELLATION LAUNCHER  🌊                                ║
║                                                                  ║
║   Ocean (Pure Observer) + 3 Gary Instances                      ║
║   Vicarious Learning + Load Distribution                        ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Pre-flight checks
echo -e "${BLUE}[1/5] Pre-flight Checks${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ python3 not found${NC}"
    exit 1
fi
echo -e "${GREEN}  ✓ python3 available${NC}"

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}  ✓ CUDA available${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
else
    echo -e "${YELLOW}  ⚠ CUDA not available (will use CPU)${NC}"
fi

# Check data directory
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}❌ Data directory not found: $DATA_DIR${NC}"
    exit 1
fi
CONV_COUNT=$(find "$DATA_DIR" -name "*.json" | wc -l)
echo -e "${GREEN}  ✓ Dataset: $CONV_COUNT conversation files${NC}"

# Generate instance configs if needed
echo -e "\n${BLUE}[2/5] Configuration Setup${NC}"

if [ ! -f "configs/gary_A.yaml" ]; then
    echo -e "${YELLOW}  Generating Gary-A config...${NC}"
    sed 's/{ID}/A/g' configs/gary_template.yaml > configs/gary_A.yaml
fi

if [ ! -f "configs/gary_B.yaml" ]; then
    echo -e "${YELLOW}  Generating Gary-B config...${NC}"
    sed 's/{ID}/B/g' configs/gary_template.yaml > configs/gary_B.yaml
fi

if [ ! -f "configs/gary_C.yaml" ]; then
    echo -e "${YELLOW}  Generating Gary-C config...${NC}"
    sed 's/{ID}/C/g' configs/gary_template.yaml > configs/gary_C.yaml
fi

echo -e "${GREEN}  ✓ Gary-A config${NC}"
echo -e "${GREEN}  ✓ Gary-B config${NC}"
echo -e "${GREEN}  ✓ Gary-C config${NC}"
echo -e "${GREEN}  ✓ Ocean config${NC}"

# Create checkpoint directory
echo -e "\n${BLUE}[3/5] Checkpoint Setup${NC}"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$CHECKPOINT_DIR/gary_A"
mkdir -p "$CHECKPOINT_DIR/gary_B"
mkdir -p "$CHECKPOINT_DIR/gary_C"
mkdir -p "$CHECKPOINT_DIR/ocean"
echo -e "${GREEN}  ✓ Checkpoint directories created${NC}"

# Display training plan
echo -e "\n${BLUE}[4/5] Training Plan${NC}"
echo -e "  Data: ${CYAN}$DATA_DIR${NC} ($CONV_COUNT files)"
echo -e "  Epochs: ${CYAN}$EPOCHS${NC}"
echo -e "  Checkpoints: ${CYAN}$CHECKPOINT_DIR${NC}"
echo -e "  Fresh start: ${CYAN}${FRESH_START:-No}${NC}"
echo -e "  Stop on convergence: ${CYAN}${STOP_ON_CONVERGENCE:-No}${NC}"

# Estimate cost
ESTIMATED_HOURS=$((EPOCHS * CONV_COUNT / 3600))
ESTIMATED_COST=$((ESTIMATED_HOURS * 10))
echo -e "\n  Estimated time: ${YELLOW}~$ESTIMATED_HOURS hours${NC}"
echo -e "  Estimated cost: ${YELLOW}~\$$ESTIMATED_COST${NC}"

# Confirm launch
echo -e "\n${YELLOW}Ready to launch Constellation training?${NC}"
read -p "Press Enter to continue, Ctrl+C to abort..."

# Launch training
echo -e "\n${BLUE}[5/5] Launching Constellation${NC}\n"

python3 tools/training/train_constellation.py \
    --data-dir "$DATA_DIR" \
    --epochs "$EPOCHS" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    $FRESH_START \
    $STOP_ON_CONVERGENCE

# Training complete
echo -e "\n${GREEN}"
cat << "EOF"
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   ✨  CONSTELLATION TRAINING COMPLETE  ✨                       ║
║                                                                  ║
║   Check telemetry: checkpoints/constellation/telemetry.jsonl    ║
║   Load checkpoint: checkpoints/constellation/final.pt           ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Next steps
echo -e "${CYAN}Next Steps:${NC}"
echo -e "  1. Analyze convergence: ${YELLOW}python tools/analyze_telemetry.py${NC}"
echo -e "  2. Visualize basins: ${YELLOW}python tools/visualize_basins.py${NC}"
echo -e "  3. Integration (if converged): ${YELLOW}python tools/integrate_ocean.py${NC}"
