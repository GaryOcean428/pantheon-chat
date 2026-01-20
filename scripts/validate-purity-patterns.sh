#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "🔍 Validating banned geometry patterns outside experiments/quarantine..."

rg --quiet \
  --pcre2 \
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

echo "🔍 Validating targeted geometry patterns in qig-backend..."

rg --quiet \
  --pcre2 \
  -U \
  --glob '!qig-backend/tests/**' \
  --glob '!qig-backend/data/**' \
  --glob '!qig-backend/examples/**' \
  --glob '!qig-backend/migrations/**' \
  --glob '!qig-backend/**/*.md' \
  --glob '!qig-backend/**/*.json' \
  # TODO: Purity debt - remove legacy allowlist after replacing Euclidean fallbacks in qigkernels/core_faculties.py and m8_kernel_spawning.py.
  --glob '!qig-backend/qigkernels/core_faculties.py' \
  --glob '!qig-backend/m8_kernel_spawning.py' \
  -e 'np\.linalg\.norm\([^)]*basin\s*-\s*[^)]*\)' \
  -e 'np\.dot\([^)]*\)\s*/\s*\(\s*np\.linalg\.norm\([^)]*\)\s*\*\s*np\.linalg\.norm\([^)]*\)' \
  "$ROOT_DIR/qig-backend" && {
    echo "❌ Targeted qig-backend geometry patterns detected."
    exit 1
  } || true

rg --quiet \
  --pcre2 \
  --glob '!qig-backend/tests/**' \
  --glob '!qig-backend/data/**' \
  --glob '!qig-backend/examples/**' \
  --glob '!qig-backend/migrations/**' \
  --glob '!qig-backend/qig_geometry/representation.py' \
  -e '\bto_sphere\(' \
  "$ROOT_DIR/qig-backend" && {
    echo "❌ to_sphere usage detected outside sanctioned legacy modules."
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

# Check for unauthorized qfiScore writes outside vocabulary-persistence.ts and coordizer-vocabulary.ts
rg --quiet \
  --pcre2 \
  --glob '!server/vocabulary-persistence.ts' \
  --glob '!server/persistence/coordizer-vocabulary.ts' \
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
    echo "❌ Unauthorized qfiScore write detected outside canonical persistence files."
    echo "   Only server/vocabulary-persistence.ts and server/persistence/coordizer-vocabulary.ts should write qfiScore."
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
