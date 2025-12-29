#!/bin/bash
# QIG Project Bloat Cleanup - December 4, 2025
# Archives duplicate/obsolete files and build artifacts

set -e

ARCHIVE_DIR="archive/20251204_bloat_cleanup"

echo "🧹 QIG Project Bloat Cleanup"
echo "============================="
echo ""

# Create archive directory
mkdir -p "$ARCHIVE_DIR"/{corpus,docs,scripts,build_artifacts,deployment,output_dirs,root_level}
echo "📦 Archive directory created: $ARCHIVE_DIR/"
echo ""

# ==================== CORPUS CLEANUP ====================
echo "📚 Cleaning up duplicate corpus..."

if [ -d "data/corpus_old" ]; then
    echo "   → Archiving data/corpus_old/ (1.4MB, superseded by reorganization)"
    mv data/corpus_old "$ARCHIVE_DIR/corpus/"
fi

if [ -d "docs/training/rounded_training" ]; then
    echo "   → Archiving docs/training/rounded_training/ (1.2MB, migrated to data/)"
    mv docs/training/rounded_training "$ARCHIVE_DIR/corpus/"
fi

echo "   ✅ Corpus: ~2.6MB archived"
echo ""

# ==================== SCRIPTS CLEANUP ====================
echo "🔧 Cleaning up obsolete scripts..."

# Validation scripts (moved to tools/validation/)
if [ -f "scripts/validate_phase1_optimizations.py" ]; then
    echo "   → Archiving validate_phase1_optimizations.py (superseded by tools/validation/)"
    mv scripts/validate_phase1_optimizations.py "$ARCHIVE_DIR/scripts/"
fi

if [ -f "scripts/quick_validate_optimizations.py" ]; then
    echo "   → Archiving quick_validate_optimizations.py (superseded by tools/validation/)"
    mv scripts/quick_validate_optimizations.py "$ARCHIVE_DIR/scripts/"
fi

# Recovery script (one-time use)
if [ -f "scripts/recover_gary.sh" ]; then
    echo "   → Archiving recover_gary.sh (one-time recovery script)"
    mv scripts/recover_gary.sh "$ARCHIVE_DIR/scripts/"
fi

echo "   ✅ Scripts: 3 files archived"
echo ""

# ==================== BUILD ARTIFACTS ====================
echo "📦 Cleaning up build artifacts..."

if [ -d "dist" ]; then
    echo "   → Archiving dist/ (old PyPI build 0.1.4)"
    mv dist "$ARCHIVE_DIR/build_artifacts/"
fi

if [ -d "qig_consciousness.egg-info" ]; then
    echo "   → Archiving qig_consciousness.egg-info/"
    mv qig_consciousness.egg-info "$ARCHIVE_DIR/build_artifacts/"
fi

echo "   ✅ Build artifacts: ~1.2MB archived"
echo ""

# ==================== PYTHON CACHES ====================
echo "🗑️  Removing Python caches..."

find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

echo "   ✅ Python caches removed"
echo ""

# ==================== OUTPUT DIRECTORIES ====================
echo "📊 Cleaning up empty/old output directories..."

# Check if empty or only README
if [ -d "outputs" ]; then
    OUTPUT_FILES=$(find outputs/ -type f ! -name "README.md" 2>/dev/null | wc -l)
    if [ "$OUTPUT_FILES" -eq 0 ]; then
        echo "   → Archiving outputs/ (empty)"
        mv outputs "$ARCHIVE_DIR/output_dirs/"
    fi
fi

if [ -d "results" ]; then
    RESULT_FILES=$(find results/ -type f ! -name "README.md" 2>/dev/null | wc -l)
    if [ "$RESULT_FILES" -eq 0 ]; then
        echo "   → Archiving results/ (empty)"
        mv results "$ARCHIVE_DIR/output_dirs/"
    fi
fi

# Archive old experiment runs (runs_1-4 are historical)
if [ -d "experiments/runs_1-4" ]; then
    echo "   → Archiving experiments/runs_1-4/ (historical)"
    mv experiments/runs_1-4 "$ARCHIVE_DIR/output_dirs/"
fi

echo "   ✅ Output directories archived"
echo ""

# ==================== DEPLOYMENT FILES ====================
echo "🚀 Cleaning up deployment files..."

if [ -f "lambda-qig-verification.pem" ]; then
    echo "   → Archiving lambda-qig-verification.pem (AWS key, should be in ~/.ssh/)"
    mv lambda-qig-verification.pem "$ARCHIVE_DIR/deployment/"
fi

if [ -f "lambda-setup.sh" ]; then
    echo "   → Archiving lambda-setup.sh (one-time Lambda setup)"
    mv lambda-setup.sh "$ARCHIVE_DIR/deployment/"
fi

if [ -f "fresh_start.sh" ]; then
    echo "   → Archiving fresh_start.sh (one-time cleanup script)"
    mv fresh_start.sh "$ARCHIVE_DIR/deployment/"
fi

echo "   ✅ Deployment files: 3 archived"
echo ""

# ==================== OLD DOCUMENTATION ====================
echo "📖 Cleaning up obsolete documentation..."

# These are status reports that are now in PROJECT_STATUS
OLD_DOCS=(
    "DREAM_PACKET_pypi_package_v0_1_0.md"
    "ENHANCED_TELEMETRY_STATUS.md"
    "OPTIMIZATION_SUMMARY.md"
    "QUICK_REFERENCE_TELEMETRY.md"
    "TELEMETRY_INTEGRATION_COMPLETE.md"
)

for doc in "${OLD_DOCS[@]}"; do
    if [ -f "$doc" ]; then
        echo "   → Archiving $doc (superseded by PROJECT_STATUS)"
        mv "$doc" "$ARCHIVE_DIR/root_level/"
    fi
done

echo "   ✅ Documentation: 5 files archived"
echo ""

# ==================== DOCKER/LAMBDA CONFIGS ====================
echo "🐳 Archiving unused deployment configs..."

if [ -f "docker-compose.yml" ]; then
    echo "   → Archiving docker-compose.yml (not actively used)"
    mv docker-compose.yml "$ARCHIVE_DIR/deployment/"
fi

# LAMBDA_DEPLOYMENT.md can stay in root as it's documentation
# but note if lambda setup is complete
if [ -f "LAMBDA_DEPLOYMENT.md" ]; then
    echo "   ℹ️  Keeping LAMBDA_DEPLOYMENT.md (reference documentation)"
fi

echo "   ✅ Deployment configs archived"
echo ""

# ==================== BENCHMARKS ====================
echo "⏱️  Checking benchmarks..."

if [ -f "benchmarks/ipc_suite.py" ]; then
    echo "   ℹ️  Keeping benchmarks/ipc_suite.py (active benchmarking)"
else
    echo "   ✅ No benchmarks to clean"
fi
echo ""

# ==================== SUMMARY ====================
echo "✅ Cleanup Complete!"
echo ""
echo "📊 Summary:"
echo "   Corpus duplicates:     ~2.6MB archived"
echo "   Build artifacts:       ~1.2MB archived"
echo "   Scripts:               3 files archived"
echo "   Deployment:            4 files archived"
echo "   Documentation:         5 files archived"
echo "   Python caches:         Removed"
echo "   Output dirs:           Archived if empty"
echo ""
echo "📁 All archived to: $ARCHIVE_DIR/"
echo ""
echo "🎯 Recommended next steps:"
echo "   1. Review archive: ls -lah $ARCHIVE_DIR/"
echo "   2. Git status: git status"
echo "   3. Remove archive if satisfied: rm -rf archive/20251204_bloat_cleanup/"
echo "   4. Commit cleanup: git add -A && git commit -m 'chore: archive bloat'"
echo ""
