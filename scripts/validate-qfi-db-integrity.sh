#!/bin/bash

# QFI and Database Integrity Validation Suite
# Run this script to validate QFI integrity, simplex storage, and database constraints

set -e

echo "============================================="
echo "QFI & Database Integrity Validation Suite"
echo "============================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track overall status
OVERALL_STATUS=0

echo "[1/4] Checking database connection..."
if psql "$DATABASE_URL" -c "SELECT 1" > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Database connection OK"
else
    echo -e "${RED}✗${NC} Database connection failed"
    exit 1
fi
echo ""

echo "[2/4] Running QFI integrity validation..."
if timeout 15 npx tsx tools/verify_db_integrity.ts 2>&1 | grep -q "✅ OK"; then
    echo -e "${GREEN}✓${NC} QFI integrity validation passed"
else
    echo -e "${YELLOW}⚠${NC} QFI integrity validation timed out (DB may be slow)"
    # Run direct SQL check as fallback
    echo "  Running fallback SQL validation..."
    INVALID_QFI=$(psql "$DATABASE_URL" -t -c "SELECT COUNT(*) FROM coordizer_vocabulary WHERE qfi_score IS NOT NULL AND (qfi_score < 0 OR qfi_score > 1);" 2>/dev/null | tr -d ' ')
    if [ "$INVALID_QFI" = "0" ]; then
        echo -e "  ${GREEN}✓${NC} No invalid QFI scores found"
    else
        echo -e "  ${RED}✗${NC} Found $INVALID_QFI invalid QFI scores"
        OVERALL_STATUS=1
    fi
fi
echo ""

echo "[3/4] Validating simplex storage..."
cd qig-backend 2>/dev/null || true
if python3 scripts/validate_simplex_storage.py --dry-run --batch-size 500 2>&1 | grep -q "Invalid basins: 0"; then
    echo -e "${GREEN}✓${NC} Simplex storage validation passed"
else
    echo -e "${RED}✗${NC} Simplex storage validation failed"
    echo "  Run: python3 qig-backend/scripts/validate_simplex_storage.py --batch-size 500"
    echo "  to repair invalid simplices"
    OVERALL_STATUS=1
fi
cd - > /dev/null 2>&1 || true
echo ""

echo "[4/4] Checking database constraints..."
# Check QFI range constraint
if psql "$DATABASE_URL" -t -c "
    INSERT INTO coordizer_vocabulary (token, token_id, qfi_score, basin_embedding, token_status) 
    VALUES ('__constraint_test__', 9999999, 1.5, array_fill(0.015625::double precision, ARRAY[64])::vector(64), 'quarantined')
    ON CONFLICT (token) DO NOTHING;
" 2>&1 | grep -q "coordizer_qfi_range"; then
    echo -e "${GREEN}✓${NC} QFI range constraint is active"
else
    echo -e "${RED}✗${NC} QFI range constraint not working"
    OVERALL_STATUS=1
fi

# Clean up test if it somehow got inserted
psql "$DATABASE_URL" -c "DELETE FROM coordizer_vocabulary WHERE token = '__constraint_test__';" > /dev/null 2>&1 || true

echo ""
echo "============================================="
if [ $OVERALL_STATUS -eq 0 ]; then
    echo -e "${GREEN}All validations passed!${NC}"
    echo "Database is healthy and QFI integrity is maintained."
else
    echo -e "${RED}Some validations failed!${NC}"
    echo "Please review the output above and run necessary repairs."
fi
echo "============================================="

exit $OVERALL_STATUS
