#!/bin/bash
# Check for hard-coded stopword lists (legacy NLP violations)
# Excludes deprecated files and tests

set -e

# Check for various stopword patterns
VIOLATIONS=$(grep -rn --include="*.py" qig-backend/ 2>/dev/null | \
    grep -E "STOPWORDS\s*=\s*(\{|\[)" | \
    grep -v "word_relationship_learner.py" | \
    grep -v "contextualized_filter.py" | \
    grep -v "SEMANTIC_CRITICAL" | \
    grep -v test || true)

if [ -n "$VIOLATIONS" ]; then
    echo "❌ Hard-coded stopword list detected (use contextualized filter instead):"
    echo "$VIOLATIONS"
    exit 1
fi

echo "✅ No hard-coded stopword violations found"
exit 0
