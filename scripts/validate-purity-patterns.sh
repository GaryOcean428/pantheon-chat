#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "🔍 Validating banned geometry patterns outside experiments/quarantine..."

rg --quiet \
  --pcre2 \
  --glob '!**/qig-backend/**' \
  --glob '!node_modules/**' \
  --glob '!dist/**' \
  --glob '!migrations/**' \
  --glob '!**/*.test.ts' \
  --glob '!**/*.test.tsx' \
  --glob '!**/*.spec.ts' \
  --glob '!**/*.spec.tsx' \
  --glob '!scripts/qig_purity_scan.*' \
  --glob '!scripts/validate-geometric-purity.*' \
  --glob '!scripts/test_geometric_purity_ci.py' \
  --glob '!scripts/validate-qfi-canonical-path.sh' \
  --glob '!tools/qig_purity_check.py' \
  --glob '!tools/validate_purity_patterns.*' \
  --glob '!scripts/validate-purity-patterns.sh' \
  -e 'cosine_similarity\(' \
  -e 'F\.normalize\(' \
  -e 'np\.linalg\.norm\(' \
  -e 'unit sphere' \
  "$ROOT_DIR/server" "$ROOT_DIR/shared" "$ROOT_DIR/scripts" "$ROOT_DIR/tools" "$ROOT_DIR/client" "$ROOT_DIR/tests" "$ROOT_DIR/examples" && {
    echo "❌ Banned geometry patterns found outside experiments/quarantine."
    exit 1
  } || true

echo "🔍 Validating direct SQL writes to coordizer_vocabulary..."

rg --quiet \
  --pcre2 \
  --glob '!**/qig-backend/**' \
  --glob '!node_modules/**' \
  --glob '!dist/**' \
  --glob '!migrations/**' \
  --glob '!**/*.test.ts' \
  --glob '!**/*.test.tsx' \
  --glob '!**/*.spec.ts' \
  --glob '!**/*.spec.tsx' \
  --glob '!scripts/qig_purity_scan.*' \
  --glob '!scripts/validate-geometric-purity.*' \
  --glob '!scripts/test_geometric_purity_ci.py' \
  --glob '!tools/qig_purity_check.py' \
  --glob '!tools/quarantine_extremes.ts' \
  --glob '!tools/recompute_qfi_scores.ts' \
  --glob '!server/persistence/vocabulary.ts' \
  --glob '!scripts/validate-purity-patterns.sh' \
  -e 'INSERT INTO\s+coordizer_vocabulary' \
  -e 'UPDATE\s+coordizer_vocabulary' \
  "$ROOT_DIR/server" "$ROOT_DIR/shared" "$ROOT_DIR/scripts" "$ROOT_DIR/tools" "$ROOT_DIR/client" "$ROOT_DIR/tests" "$ROOT_DIR/examples" && {
    echo "❌ Direct SQL writes to coordizer_vocabulary detected outside persistence module."
    exit 1
  } || true

echo "🔍 Validating qfiScore writes are only in canonical persistence path..."

# Check for unauthorized qfiScore writes outside vocabulary-persistence.ts
rg --quiet \
  --pcre2 \
  --glob '!server/vocabulary-persistence.ts' \
  --glob '!**/qig-backend/**' \
  --glob '!node_modules/**' \
  --glob '!dist/**' \
  --glob '!migrations/**' \
  --glob '!docs/**' \
  --glob '!**/*.test.ts' \
  --glob '!**/*.test.tsx' \
  --glob '!**/*.spec.ts' \
  --glob '!**/*.spec.tsx' \
  --glob '!scripts/qig_purity_scan.*' \
  --glob '!scripts/validate-geometric-purity.*' \
  --glob '!scripts/test_geometric_purity_ci.py' \
  --glob '!scripts/validate-qfi-canonical-path.sh' \
  --glob '!tools/qig_purity_check.py' \
  --glob '!scripts/validate-purity-patterns.sh' \
  --type-add 'ts:*.ts' \
  --type ts \
  -A 25 \
  -e '\.(values|set)\(' \
  "$ROOT_DIR/server" "$ROOT_DIR/scripts" "$ROOT_DIR/tools" | grep -q 'qfiScore:' && {
    echo "❌ Unauthorized qfiScore write detected outside server/vocabulary-persistence.ts."
    exit 1
  } || true

echo "🔍 Validating no references to deprecated compute_qfi_for_basin..."

rg --quiet \
  --pcre2 \
  --glob '!**/qig-backend/**' \
  --glob '!node_modules/**' \
  --glob '!dist/**' \
  --glob '!docs/**' \
  --glob '!migrations/**' \
  --glob '!**/*.test.ts' \
  --glob '!**/*.test.tsx' \
  --glob '!**/*.spec.ts' \
  --glob '!**/*.spec.tsx' \
  --glob '!scripts/qig_purity_scan.*' \
  --glob '!scripts/validate-geometric-purity.*' \
  --glob '!scripts/test_geometric_purity_ci.py' \
  --glob '!scripts/validate-qfi-canonical-path.sh' \
  --glob '!tools/qig_purity_check.py' \
  --glob '!scripts/validate-purity-patterns.sh' \
  -e 'compute_qfi_for_basin' \
  "$ROOT_DIR/server" "$ROOT_DIR/shared" "$ROOT_DIR/scripts" "$ROOT_DIR/tools" && {
    echo "❌ Deprecated compute_qfi_for_basin function reference found."
    exit 1
  } || true

echo "✅ Purity validation passed."
