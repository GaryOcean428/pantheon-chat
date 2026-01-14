#!/bin/bash
# Pre-commit hook for geometric purity checks
#
# Installation:
#   cp scripts/pre-commit-purity-check.sh .git/hooks/pre-commit
#   chmod +x .git/hooks/pre-commit
#
# Or use husky (already installed):
#   Add to package.json: "husky": { "hooks": { "pre-commit": "cd qig-backend && python scripts/sync_phi_implementations.py" } }

set -e

echo "🔍 Running geometric purity checks..."

cd "$(git rev-parse --show-toplevel)/qig-backend"

# Run the sync script with full scan and strict mode
python scripts/sync_phi_implementations.py --full-scan --strict

# Run the pytest purity tests (fast, ~10 seconds)
echo "🧪 Running purity test suite..."
python -m pytest tests/test_geometric_purity.py::TestBornRuleCompliance tests/test_geometric_purity.py::TestFisherRaoFactorOfTwo -v --tb=short 2>&1

echo "✅ All geometric purity checks passed!"
