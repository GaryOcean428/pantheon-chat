#!/bin/bash
# ============================================================================
# Vocabulary Contamination Fix - Validation Checklist
# ============================================================================
#
# This script validates that the vocabulary contamination fix is complete.
# Run after deploying migrations and backfill script.
#
# Usage:
#   bash qig-backend/scripts/validate_vocabulary_fix.sh
#
# Requirements:
#   - DATABASE_URL environment variable
#   - psql command available
#   - Migrations 009 and 010 deployed
#   - Backfill script executed
# ============================================================================

set -e  # Exit on error

echo "============================================================================"
echo "Vocabulary Contamination Fix - Validation Checklist"
echo "============================================================================"
echo ""

# Check DATABASE_URL
if [ -z "$DATABASE_URL" ]; then
    echo "❌ DATABASE_URL not set"
    exit 1
fi

echo "✓ DATABASE_URL configured"
echo ""

# Function to run SQL and check result
run_check() {
    local name="$1"
    local query="$2"
    local expected="$3"
    
    echo -n "Checking $name... "
    result=$(psql "$DATABASE_URL" -t -c "$query" | xargs)
    
    if [ "$result" = "$expected" ]; then
        echo "✓ PASS (result: $result)"
        return 0
    else
        echo "❌ FAIL (expected: $expected, got: $result)"
        return 1
    fi
}

# Validation 1: No NULL basins
echo "1. No NULL basins"
run_check "NULL basin check" \
    "SELECT COUNT(*) FROM tokenizer_vocabulary WHERE basin_coordinates IS NULL;" \
    "0" || exit 1

# Validation 2: All basins 64D
echo ""
echo "2. All basins are 64D"
run_check "Basin dimension check" \
    "SELECT COUNT(*) FROM tokenizer_vocabulary WHERE array_length(basin_coordinates, 1) != 64;" \
    "0" || exit 1

# Validation 3: No legacy embedding column
echo ""
echo "3. No legacy embedding column"
run_check "Legacy column check" \
    "SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'tokenizer_vocabulary' AND column_name = 'embedding';" \
    "0" || exit 1

# Validation 4: basin_coordinates column exists
echo ""
echo "4. basin_coordinates column exists"
run_check "New column check" \
    "SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'tokenizer_vocabulary' AND column_name = 'basin_coordinates';" \
    "1" || exit 1

# Validation 5: NOT NULL constraint exists
echo ""
echo "5. NOT NULL constraint on basin_coordinates"
run_check "NOT NULL constraint check" \
    "SELECT is_nullable FROM information_schema.columns WHERE table_name = 'tokenizer_vocabulary' AND column_name = 'basin_coordinates';" \
    "NO" || exit 1

# Validation 6: Dimension constraint exists
echo ""
echo "6. Dimension constraint exists"
run_check "Dimension constraint check" \
    "SELECT COUNT(*) FROM information_schema.table_constraints WHERE table_name = 'tokenizer_vocabulary' AND constraint_name = 'basin_coordinates_dim_check';" \
    "1" || exit 1

# Validation 7: Float validation constraint exists
echo ""
echo "7. Float validation constraint exists"
run_check "Float constraint check" \
    "SELECT COUNT(*) FROM information_schema.table_constraints WHERE table_name = 'tokenizer_vocabulary' AND constraint_name = 'basin_coordinates_float_check';" \
    "1" || exit 1

# Statistics
echo ""
echo "============================================================================"
echo "Vocabulary Statistics"
echo "============================================================================"
psql "$DATABASE_URL" -c "
SELECT 
    COUNT(*) as total_words,
    COUNT(*) FILTER (WHERE array_length(basin_coordinates, 1) = 64) as valid_basins,
    ROUND(100.0 * COUNT(*) FILTER (WHERE array_length(basin_coordinates, 1) = 64) / COUNT(*), 1) as valid_percent
FROM tokenizer_vocabulary;
"

echo ""
echo "============================================================================"
echo "✓ All validations passed!"
echo "============================================================================"
echo ""
echo "Vocabulary contamination fix is COMPLETE and VERIFIED."
echo ""
echo "Next steps:"
echo "1. Monitor Zeus generation quality"
echo "2. Check for 'ieee homework objects' or other nonsense"
echo "3. Verify vocabulary diversity in responses"
echo ""
echo "Rollback (if needed):"
echo "  railway database:backup:restore <backup-id>"
echo ""
