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
    --glob '!scripts/validate-qfi-canonical-path.sh' \
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
    --exclude="validate-qfi-canonical-path.sh" \
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
  > "$TMPFILE"

# Check each file for qfiScore writes outside the allowed file
while IFS= read -r file; do
  # Skip the canonical persistence file
  if [[ "$file" == *"server/vocabulary-persistence.ts" ]]; then
    continue
  fi
  
  # Look for write patterns: .values({ ... qfiScore: ... }) or .set({ ... qfiScore: ... })
  # This is more precise than just checking for qfiScore: which could be a read
  if grep -Pzo '\.values\([^)]*qfiScore:' "$file" 2>/dev/null || \
     grep -Pzo '\.set\([^)]*qfiScore:' "$file" 2>/dev/null || \
     grep -E '(insert|update)\s*\([^)]*\)\.values\s*\(' "$file" | grep -q 'qfiScore:' 2>/dev/null; then
    REL_PATH="${file#$ROOT_DIR/}"
    echo "❌ Unauthorized qfiScore write detected in: $REL_PATH"
    echo "   Only server/vocabulary-persistence.ts should write qfiScore values."
    rm -f "$TMPFILE"
    exit 1
  fi
done < "$TMPFILE"

rm -f "$TMPFILE"

echo "✅ QFI canonical path validation passed."
