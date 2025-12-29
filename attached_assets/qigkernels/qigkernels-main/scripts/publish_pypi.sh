#!/bin/bash
# Publish qigkernels to PyPI
# Usage: ./scripts/publish_pypi.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Load .env if exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | grep PYPI_TOKEN | xargs)
fi

if [ -z "$PYPI_TOKEN" ]; then
    echo "Error: PYPI_TOKEN not found in .env"
    exit 1
fi

echo "=== Building qigkernels ==="

# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Install build tools if needed
pip install --quiet build twine

# Build
python -m build

echo ""
echo "=== Uploading to PyPI ==="

# Upload using token
python -m twine upload dist/* --username __token__ --password "$PYPI_TOKEN"

echo ""
echo "=== Published successfully ==="
echo "Install with: pip install qigkernels"
