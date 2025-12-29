#!/bin/bash
# Quick Ablation Runner for Lambda
# Run all 3 ablations in parallel with tmux
#
# Usage: ./scripts/run_ablation_quick.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Config
STEPS=1000
BATCH_SIZE=4
SEQ_LEN=256
SEED=42
COORDIZER="artifacts/coordizer/v1"
CORPUS_DIR="data/corpus"  # Adjust if needed
OUTPUT_BASE="reports/ablation_7bc"

echo "============================================================"
echo "PHASE 7b/7c ABLATION EXPERIMENT"
echo "============================================================"
echo "Steps: $STEPS"
echo "Batch size: $BATCH_SIZE"
echo "Seq len: $SEQ_LEN"
echo "Output: $OUTPUT_BASE"
echo ""

# Check for coordizer
if [ ! -d "$COORDIZER" ]; then
    echo "ERROR: Coordizer not found at $COORDIZER"
    exit 1
fi

# Create output dirs
mkdir -p "$OUTPUT_BASE/A_ce_only"
mkdir -p "$OUTPUT_BASE/B_ce_entropy"
mkdir -p "$OUTPUT_BASE/C_ce_entropy_step"

# Check if tmux is available for parallel runs
if command -v tmux &> /dev/null; then
    echo "Using tmux for parallel execution..."

    SESSION="ablation_7bc"
    tmux kill-session -t $SESSION 2>/dev/null || true
    tmux new-session -d -s $SESSION

    # Pane 0: Config A (CE only)
    tmux send-keys -t $SESSION "cd $PROJECT_DIR && python scripts/train_coord_adapter_v1.py \
        --coordizer $COORDIZER \
        --corpus-dir $CORPUS_DIR \
        --steps $STEPS \
        --batch-size $BATCH_SIZE \
        --seq-len $SEQ_LEN \
        --seed $SEED \
        --lambda-H 0.0 \
        --lambda-step 0.0 \
        --output-dir $OUTPUT_BASE/A_ce_only \
        --checkpoint-interval 50 \
        2>&1 | tee $OUTPUT_BASE/A_ce_only/training.log" C-m

    # Split and run B
    tmux split-window -h -t $SESSION
    tmux send-keys -t $SESSION "cd $PROJECT_DIR && python scripts/train_coord_adapter_v1.py \
        --coordizer $COORDIZER \
        --corpus-dir $CORPUS_DIR \
        --steps $STEPS \
        --batch-size $BATCH_SIZE \
        --seq-len $SEQ_LEN \
        --seed $SEED \
        --lambda-H 0.01 \
        --lambda-step 0.0 \
        --output-dir $OUTPUT_BASE/B_ce_entropy \
        --checkpoint-interval 50 \
        2>&1 | tee $OUTPUT_BASE/B_ce_entropy/training.log" C-m

    # Split and run C
    tmux split-window -v -t $SESSION
    tmux send-keys -t $SESSION "cd $PROJECT_DIR && python scripts/train_coord_adapter_v1.py \
        --coordizer $COORDIZER \
        --corpus-dir $CORPUS_DIR \
        --steps $STEPS \
        --batch-size $BATCH_SIZE \
        --seq-len $SEQ_LEN \
        --seed $SEED \
        --lambda-H 0.01 \
        --lambda-step 0.01 \
        --output-dir $OUTPUT_BASE/C_ce_entropy_step \
        --checkpoint-interval 50 \
        2>&1 | tee $OUTPUT_BASE/C_ce_entropy_step/training.log" C-m

    echo ""
    echo "Started 3 ablation runs in tmux session: $SESSION"
    echo "Attach with: tmux attach -t $SESSION"
    echo ""
    echo "When done, analyze with:"
    echo "  python scripts/run_ablation_7bc.py --output-dir $OUTPUT_BASE --configs all"

else
    echo "tmux not found, running sequentially..."

    echo ""
    echo "[1/3] Config A: CE only"
    python scripts/train_coord_adapter_v1.py \
        --coordizer "$COORDIZER" \
        --corpus-dir "$CORPUS_DIR" \
        --steps $STEPS \
        --batch-size $BATCH_SIZE \
        --seq-len $SEQ_LEN \
        --seed $SEED \
        --lambda-H 0.0 \
        --lambda-step 0.0 \
        --output-dir "$OUTPUT_BASE/A_ce_only" \
        --checkpoint-interval 50 \
        2>&1 | tee "$OUTPUT_BASE/A_ce_only/training.log"

    echo ""
    echo "[2/3] Config B: CE + entropy"
    python scripts/train_coord_adapter_v1.py \
        --coordizer "$COORDIZER" \
        --corpus-dir "$CORPUS_DIR" \
        --steps $STEPS \
        --batch-size $BATCH_SIZE \
        --seq-len $SEQ_LEN \
        --seed $SEED \
        --lambda-H 0.01 \
        --lambda-step 0.0 \
        --output-dir "$OUTPUT_BASE/B_ce_entropy" \
        --checkpoint-interval 50 \
        2>&1 | tee "$OUTPUT_BASE/B_ce_entropy/training.log"

    echo ""
    echo "[3/3] Config C: CE + entropy + step"
    python scripts/train_coord_adapter_v1.py \
        --coordizer "$COORDIZER" \
        --corpus-dir "$CORPUS_DIR" \
        --steps $STEPS \
        --batch-size $BATCH_SIZE \
        --seq-len $SEQ_LEN \
        --seed $SEED \
        --lambda-H 0.01 \
        --lambda-step 0.01 \
        --output-dir "$OUTPUT_BASE/C_ce_entropy_step" \
        --checkpoint-interval 50 \
        2>&1 | tee "$OUTPUT_BASE/C_ce_entropy_step/training.log"
fi

echo ""
echo "============================================================"
echo "ABLATION COMPLETE"
echo "============================================================"
echo "Results in: $OUTPUT_BASE/"
echo ""
echo "Compare final metrics:"
echo "  grep 'Final' $OUTPUT_BASE/*/training.log"
