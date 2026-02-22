#!/usr/bin/env bash
# restore_qgs.sh — restores qig_generative_service.py to production state
# Run from repo root: bash scripts/restore_qgs.sh
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
FILE="qig-backend/qig_generative_service.py"
GOOD_COMMIT="7e2b6772c196f0df443cc6dd345a9c7e62d06cd2"
PATCH="docs/04-records/qgs_proxy_restoration.patch"

echo "[restore] Checking out original from $GOOD_COMMIT ..."
git checkout "$GOOD_COMMIT" -- "$FILE"

echo "[restore] Applying proxy_routed / proxy_kernels patch ..."
git apply "$REPO_ROOT/$PATCH"

echo "[restore] Verifying ..."
python3 - << 'PY'
import ast, sys
with open('qig-backend/qig_generative_service.py') as f:
    c = f.read()
ast.parse(c)
assert len(c) > 85000, f'File too small: {len(c)}'
assert 'PRR_AVAILABLE' in c
assert 'StreamingCollapseMonitor' in c
assert 'proxy_routed: bool = False' in c
assert 'proxy_routed=_proxy_routed' in c
print(f'[restore] OK: {len(c)} bytes, AST clean, proxy fields present')
PY

echo "[restore] Committing ..."
git add "$FILE"
git commit -m "restore(qig_generative_service): production file + proxy_routed/proxy_kernels (TCP v6.1)"
git push
echo "[restore] Done."
