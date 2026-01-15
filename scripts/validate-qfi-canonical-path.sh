#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "🔍 Validating QFI canonical path enforcement..."

# Check for deprecated compute_qfi_for_basin function references
echo "  Checking for deprecated compute_qfi_for_basin..."
if command -v rg &> /dev/null; then
  rg --quiet \
    --glob '!node_modules/**' \
    --glob '!dist/**' \
    --glob '!docs/**' \
    --glob '!migrations/**' \
    --glob '!**/*.test.ts' \
    --glob '!**/*.test.tsx' \
    --glob '!**/*.spec.ts' \
    --glob '!**/*.spec.tsx' \
    --glob '!scripts/validate-qfi-canonical-path.sh' \
    --glob '!scripts/validate-purity-patterns.sh' \
    -e 'compute_qfi_for_basin' \
    "$ROOT_DIR/server" "$ROOT_DIR/shared" "$ROOT_DIR/scripts" "$ROOT_DIR/tools" && {
      echo "❌ Deprecated compute_qfi_for_basin function found in canonical paths."
      echo "   Only compute_qfi_score_simplex should be used."
      exit 1
    } || true
else
  # Fallback to grep if ripgrep is not available
  grep -r \
    --exclude-dir=node_modules \
    --exclude-dir=dist \
    --exclude-dir=docs \
    --exclude-dir=migrations \
    --exclude-dir=tests \
    --exclude="validate-qfi-canonical-path.sh" \
    --exclude="validate-purity-patterns.sh" \
    --exclude="*.test.ts" \
    --exclude="*.test.tsx" \
    --exclude="*.spec.ts" \
    --exclude="*.spec.tsx" \
    'compute_qfi_for_basin' \
    "$ROOT_DIR/server" "$ROOT_DIR/shared" "$ROOT_DIR/scripts" "$ROOT_DIR/tools" 2>/dev/null && {
      echo "❌ Deprecated compute_qfi_for_basin function found in canonical paths."
      echo "   Only compute_qfi_score_simplex should be used."
      exit 1
    } || true
fi

# Check for qfiScore writes outside the canonical persistence module
echo "  Checking for qfiScore writes outside canonical upsert path..."

# Find all TypeScript files in the target directories
TMPFILE=$(mktemp)
find "$ROOT_DIR/server" "$ROOT_DIR/scripts" "$ROOT_DIR/tools" \
  -name "*.ts" \
  ! -path "*/node_modules/*" \
  ! -path "*/dist/*" \
  ! -path "*/docs/*" \
  ! -path "*/migrations/*" \
  ! -path "*/tests/*" \
  ! -name "*.test.ts" \
  ! -name "*.test.tsx" \
  ! -name "*.spec.ts" \
  ! -name "*.spec.tsx" \
  > "$TMPFILE"

# Check each file for qfiScore writes outside the allowed file
while IFS= read -r file; do
  # Skip the canonical persistence file
  if [[ "$file" == *"server/vocabulary-persistence.ts" ]] || [[ "$file" == *"server/persistence/coordizer-vocabulary.ts" ]]; then
    continue
  fi
  
  # Look for write patterns in the context of database operations
  # We check for lines containing 'qfiScore:' that are near .values() or .set() calls
  # This detects patterns like: .values({ qfiScore: ... }) or .set({ qfiScore: ... })
  if grep -n 'qfiScore:' "$file" > /dev/null 2>&1; then
    # Get lines around qfiScore mentions and check for write operations
    CONTEXT=$(grep -B 5 -A 5 'qfiScore:' "$file" 2>/dev/null || true)
    if echo "$CONTEXT" | grep -E '\.(values|set)\s*\(' > /dev/null 2>&1; then
      REL_PATH="${file#$ROOT_DIR/}"
      echo "❌ Potential unauthorized qfiScore write detected in: $REL_PATH"
      echo "   Only server/vocabulary-persistence.ts should write qfiScore values."
      echo "   If this is a false positive (e.g., reading qfiScore), please review manually."
      rm -f "$TMPFILE"
      exit 1
    fi
  fi
done < "$TMPFILE"

rm -f "$TMPFILE"

echo "✅ QFI canonical path validation passed."
