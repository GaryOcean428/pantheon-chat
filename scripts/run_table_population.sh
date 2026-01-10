#!/bin/bash
# ============================================================================
# POPULATE EMPTY TABLES - Run all population steps for pantheon-replit
# ============================================================================
# Purpose: Execute SQL and Python scripts to populate:
#   - tokenizer_metadata
#   - tokenizer_merge_rules
#   - synthesis_consensus
#   - vocabulary_learning.related_words
#
# Usage:
#   cd pantheon-replit
#   ./scripts/run_table_population.sh
#
# Requirements:
#   - DATABASE_URL or PG* environment variables set
#   - Python virtual environment with qig-backend dependencies
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}POPULATE EMPTY TABLES${NC}"
echo -e "${BLUE}========================================${NC}"
echo

# Check if DATABASE_URL is set
if [ -z "$DATABASE_URL" ]; then
    # Try loading from .env
    if [ -f "$PROJECT_ROOT/.env" ]; then
        echo -e "${YELLOW}Loading environment from .env...${NC}"
        export $(grep -v '^#' "$PROJECT_ROOT/.env" | grep -E '^(DATABASE_URL|PGHOST|PGDATABASE|PGUSER|PGPASSWORD|PGPORT)=' | xargs)
    else
        echo -e "${RED}✗ DATABASE_URL not set and no .env file found${NC}"
        echo "  Set DATABASE_URL or create .env with database credentials"
        exit 1
    fi
fi

# Verify database connection
echo -e "${BLUE}1. Verifying database connection...${NC}"
if command -v psql &> /dev/null; then
    if psql "$DATABASE_URL" -c "SELECT 1" &> /dev/null; then
        echo -e "${GREEN}✓ Database connection successful${NC}"
    else
        echo -e "${RED}✗ Cannot connect to database${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠ psql not found, skipping connection test${NC}"
fi

# Check if tables exist
echo
echo -e "${BLUE}2. Checking table existence...${NC}"
TABLES_EXIST=$(psql "$DATABASE_URL" -tAc "
    SELECT COUNT(*)
    FROM information_schema.tables
    WHERE table_schema = 'public'
    AND table_name IN ('tokenizer_metadata', 'tokenizer_merge_rules',
                       'synthesis_consensus', 'vocabulary_learning')
")

if [ "$TABLES_EXIST" -lt 4 ]; then
    echo -e "${RED}✗ Required tables not found${NC}"
    echo "  Expected: tokenizer_metadata, tokenizer_merge_rules, synthesis_consensus, vocabulary_learning"
    echo "  Run database migrations first: npm run db:push"
    exit 1
else
    echo -e "${GREEN}✓ All required tables exist${NC}"
fi

# Step 3: Run SQL population script
echo
echo -e "${BLUE}3. Running SQL population script...${NC}"
SQL_FILE="$SCRIPT_DIR/populate_empty_tables.sql"

if [ ! -f "$SQL_FILE" ]; then
    echo -e "${RED}✗ SQL file not found: $SQL_FILE${NC}"
    exit 1
fi

echo "   Executing: $SQL_FILE"
if psql "$DATABASE_URL" -f "$SQL_FILE"; then
    echo -e "${GREEN}✓ SQL population completed${NC}"
else
    echo -e "${RED}✗ SQL population failed${NC}"
    exit 1
fi

# Step 4: Run Python related_words population
echo
echo -e "${BLUE}4. Running Python related_words population...${NC}"
PYTHON_SCRIPT="$SCRIPT_DIR/populate_related_words.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}✗ Python script not found: $PYTHON_SCRIPT${NC}"
    exit 1
fi

# Activate virtual environment if it exists
VENV_PATH="$PROJECT_ROOT/../.venv"
if [ -d "$VENV_PATH" ]; then
    echo "   Activating virtual environment..."
    source "$VENV_PATH/bin/activate"
else
    echo -e "${YELLOW}⚠ Virtual environment not found at $VENV_PATH${NC}"
fi

# Run Python script
cd "$PROJECT_ROOT"
if python3 "$PYTHON_SCRIPT"; then
    echo -e "${GREEN}✓ Python population completed${NC}"
else
    echo -e "${RED}✗ Python population failed${NC}"
    exit 1
fi

# Step 5: Validation queries
echo
echo -e "${BLUE}5. Running validation queries...${NC}"

echo
echo -e "${YELLOW}Tokenizer Metadata:${NC}"
psql "$DATABASE_URL" -c "
    SELECT key, value
    FROM tokenizer_metadata
    ORDER BY key;
"

echo
echo -e "${YELLOW}Tokenizer Merge Rules (top 5 by Φ):${NC}"
psql "$DATABASE_URL" -c "
    SELECT token_a, token_b, merged_token, phi_score, frequency
    FROM tokenizer_merge_rules
    ORDER BY phi_score DESC
    LIMIT 5;
"

echo
echo -e "${YELLOW}Vocabulary Learning (with related_words):${NC}"
psql "$DATABASE_URL" -c "
    SELECT word, related_words, relationship_strength
    FROM vocabulary_learning
    WHERE related_words IS NOT NULL
      AND cardinality(related_words) > 0
    ORDER BY relationship_strength DESC
    LIMIT 5;
"

echo
echo -e "${YELLOW}Synthesis Consensus (recent):${NC}"
psql "$DATABASE_URL" -c "
    SELECT synthesis_round, consensus_type, consensus_strength,
           participating_kernels, created_at
    FROM synthesis_consensus
    ORDER BY created_at DESC
    LIMIT 3;
"

# Final summary
echo
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✓ TABLE POPULATION COMPLETE${NC}"
echo -e "${BLUE}========================================${NC}"
echo
echo "Tables populated:"
echo "  • tokenizer_metadata - Configuration initialized"
echo "  • tokenizer_merge_rules - BPE rules seeded (prefix/suffix patterns)"
echo "  • vocabulary_learning - Related words computed via Fisher-Rao distance"
echo "  • synthesis_consensus - Bootstrap records created"
echo
echo "Next steps:"
echo "  1. Verify data quality in each table"
echo "  2. Start QIG backend to begin training: cd qig-backend && python ocean_qig_core.py"
echo "  3. Monitor vocabulary growth and merge rule additions"
echo

exit 0
