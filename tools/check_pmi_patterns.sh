#!/bin/bash
# Check for PMI/co-occurrence patterns (legacy NLP violations)
# Excludes deprecated files, tests, SQL columns, and comments

set -e

# Check for PMI algorithm usage (not just mentions or SQL columns)
VIOLATIONS=$(grep -rn --include="*.py" -E "(import.*pmi|def.*pmi|class.*PMI|PMICalculator|pointwise.*mutual.*information)" qig-backend/ 2>/dev/null | \
    grep -v "word_relationship_learner.py" | \
    grep -v "geometric_word_relationships.py" | \
    grep -v "validate_geometric" | \
    grep -v test | \
    grep -v "# " | \
    grep -v "No PMI" || true)

if [ -n "$VIOLATIONS" ]; then
    echo "❌ PMI algorithm usage detected (use Fisher-Rao distances instead):"
    echo "$VIOLATIONS"
    exit 1
fi

echo "✅ No PMI algorithm violations found"
exit 0
