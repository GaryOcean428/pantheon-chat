#!/usr/bin/env python3
"""Fix vocabulary_observations table data issues.

Issues addressed:
1. type column contains the word text instead of word/phrase/sequence
2. phrase_category is NULL
3. is_real_word is FALSE for real words
4. max_phi and efficiency_gain are all 0s

Auto-detects schema columns - works with any table structure.
"""

import os
import sys
from pathlib import Path

# Add qig-backend to path for imports
_qig_backend = Path(__file__).parent.parent
if str(_qig_backend) not in sys.path:
    sys.path.insert(0, str(_qig_backend))

BATCH_SIZE = 200

# BPE fragments that aren't real words
BPE_FRAGMENTS = {
    'ing', 'tion', 'ness', 'ment', 'able', 'ible', 'ful', 'less',
    'ly', 'er', 'ed', 'es', 'al', 'ive', 'ous', 'ity', 'ism',
    'ist', 'ure', 'ance', 'ence', 'ant', 'ent', 'ary', 'ory',
    're', 'un', 'dis', 'mis', 'pre', 'non', 'anti', 'de',
    'sub', 'super', 'semi', 'mid', 'over', 'under', 'out',
    'pro', 'ex', 'co', 'inter', 'trans', 'post', 'multi',
}

# Code artifacts to filter
CODE_ARTIFACTS = {
    'const', 'let', 'var', 'function', 'return', 'import', 'export',
    'class', 'interface', 'type', 'enum', 'async', 'await', 'try',
    'catch', 'throw', 'new', 'this', 'super', 'extends', 'implements',
    'null', 'undefined', 'true', 'false', 'if', 'else', 'for', 'while',
    'switch', 'case', 'break', 'continue', 'default', 'typeof', 'instanceof',
    'console', 'log', 'debug', 'error', 'warn', 'info',
    'def', 'self', 'cls', 'lambda', 'yield', 'assert', 'raise',
    'from', 'as', 'with', 'pass', 'global', 'nonlocal',
}

# Category keywords for phrase_category classification
CATEGORY_KEYWORDS = {
    'topic': ['about', 'regarding', 'concerning', 'related', 'topic', 'subject', 'area'],
    'concept': ['idea', 'concept', 'theory', 'principle', 'abstract', 'notion', 'understanding'],
    'pattern': ['pattern', 'structure', 'format', 'template', 'schema', 'model', 'form'],
    'entity': ['name', 'person', 'place', 'organization', 'company', 'product', 'brand'],
    'action': ['do', 'make', 'create', 'build', 'run', 'process', 'execute', 'perform'],
    'property': ['is', 'has', 'can', 'attribute', 'property', 'feature', 'characteristic'],
}


def classify_type(text: str) -> str:
    """Classify text as word, phrase, or sequence."""
    if not text:
        return 'word'

    # Multiple words = phrase
    words = text.split()
    if len(words) > 1:
        return 'phrase'

    # Contains special characters = sequence
    if any(c in text for c in ['_', '-', '.', '/', ':', '@', '#']):
        if len(text) > 15:
            return 'sequence'
        # Short hyphenated = compound word (phrase)
        if '-' in text:
            return 'phrase'

    # Single word
    return 'word'


def classify_phrase_category(text: str) -> str:
    """Classify phrase into topic/concept/pattern/entity/action/property/unknown."""
    if not text:
        return 'unknown'

    text_lower = text.lower()

    # Check for category keywords
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                return category

    # Check by structure
    words = text.split()
    if len(words) >= 2:
        # "X of Y" patterns are often concepts
        if 'of' in words or 'for' in words:
            return 'concept'
        # Capitalized words suggest entity
        if any(w[0].isupper() for w in words if w):
            return 'entity'

    # Single word heuristics
    if len(words) == 1:
        word = words[0]
        # Verbs (ending in -ing, -ed) suggest action
        if word.endswith('ing') or word.endswith('ed'):
            return 'action'
        # Abstract endings suggest concept
        if word.endswith('ness') or word.endswith('ity') or word.endswith('ism'):
            return 'concept'

    return 'unknown'


def is_real_word_strict(text: str) -> bool:
    """Check if text is a real word (not BPE fragment or code artifact)."""
    if not text or len(text) < 2:
        return False

    word = text.lower().strip()

    # Filter code artifacts
    if word in CODE_ARTIFACTS:
        return False

    # Filter BPE fragments (short only)
    if len(word) <= 5 and word in BPE_FRAGMENTS:
        return False

    # Filter camelCase and snake_case
    if '_' in text or (any(c.isupper() for c in text[1:])):
        return False

    # Filter words starting with numbers
    if text[0].isdigit():
        return False

    # BPE markers
    if text.startswith('##') or text.startswith('@@') or text.startswith('▁'):
        return False

    # Byte tokens
    if text.startswith('<') or text.endswith('>'):
        return False

    # Must be mostly alphabetic (allow hyphens and apostrophes)
    clean = text.replace('-', '').replace("'", '').replace("'", '')
    if not clean.isalpha():
        # Allow single space for phrases
        if ' ' in text:
            parts = text.split()
            if all(p.replace('-', '').replace("'", '').isalpha() for p in parts if p):
                return True
        return False

    # Real words are typically 3-30 characters
    if len(word) < 3 or len(word) > 30:
        return False

    return True


def compute_efficiency_gain(avg_phi: float, frequency: int) -> float:
    """Compute efficiency gain based on phi and frequency.

    Efficiency = how much value this observation adds.
    High phi + high frequency = high efficiency.
    """
    if avg_phi <= 0 or frequency <= 0:
        return 0.0

    # Log scale for frequency to avoid domination by common words
    import math
    freq_factor = math.log1p(frequency) / 10.0  # Normalize to ~0-1 range

    # Efficiency = phi * frequency_factor
    efficiency = avg_phi * min(freq_factor, 1.0)

    return round(efficiency, 4)


def fix_vocabulary_observations(limit: int = 0, dry_run: bool = False):
    """Fix vocabulary_observations table issues."""
    import psycopg2
    from psycopg2.extras import RealDictCursor

    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        print("ERROR: DATABASE_URL not set")
        return

    conn = psycopg2.connect(database_url)

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Check if table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'vocabulary_observations'
            )
        """)
        if not cur.fetchone()['exists']:
            print("vocabulary_observations table does not exist")
            conn.close()
            return

        # Get actual columns
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'vocabulary_observations'
            ORDER BY ordinal_position
        """)
        columns = [r['column_name'] for r in cur.fetchall()]
        print(f"Table columns: {columns}")

        # Find required columns
        has_type = 'type' in columns
        has_phrase_cat = 'phrase_category' in columns
        has_real_word = 'is_real_word' in columns
        has_max_phi = 'max_phi' in columns
        has_avg_phi = 'avg_phi' in columns
        has_efficiency = 'efficiency_gain' in columns
        has_frequency = 'frequency' in columns

        print(f"Has type: {has_type}, phrase_category: {has_phrase_cat}, is_real_word: {has_real_word}")
        print(f"Has max_phi: {has_max_phi}, avg_phi: {has_avg_phi}, efficiency_gain: {has_efficiency}")

        # Build query to find rows needing fixes
        where_parts = []

        # Type field contains text instead of word/phrase/sequence
        if has_type:
            where_parts.append("(type IS NULL OR type NOT IN ('word', 'phrase', 'sequence'))")

        # phrase_category is NULL
        if has_phrase_cat:
            where_parts.append("(phrase_category IS NULL OR phrase_category = 'unknown')")

        # is_real_word is NULL or needs recomputing
        if has_real_word:
            where_parts.append("(is_real_word = false)")

        # max_phi is 0 but avg_phi might have value
        if has_max_phi and has_avg_phi:
            where_parts.append("(max_phi = 0 AND avg_phi > 0)")

        # efficiency_gain is 0 but could be computed
        if has_efficiency and has_avg_phi and has_frequency:
            where_parts.append("(efficiency_gain = 0 OR efficiency_gain IS NULL)")

        if not where_parts:
            print("No fixable columns found")
            conn.close()
            return

        # Select all rows and fix any that have issues
        query = f"""
            SELECT id, text, type, phrase_category, is_real_word,
                   avg_phi, max_phi, efficiency_gain, frequency
            FROM vocabulary_observations
            ORDER BY id
        """
        if limit > 0:
            query += f" LIMIT {limit}"

        print(f"\nQuerying rows...")
        cur.execute(query)
        rows = cur.fetchall()

    if not rows:
        print("No rows found in vocabulary_observations")
        conn.close()
        return

    print(f"Found {len(rows)} total rows, analyzing...")

    # Analyze which rows need fixes
    to_fix = []
    stats = {'type': 0, 'phrase_cat': 0, 'real_word': 0, 'max_phi': 0, 'efficiency': 0}

    for row in rows:
        text = row.get('text', '')
        needs_fix = False
        fixes = {}

        # Fix type
        current_type = row.get('type', '')
        if current_type not in ('word', 'phrase', 'sequence'):
            correct_type = classify_type(text)
            fixes['type'] = correct_type
            stats['type'] += 1
            needs_fix = True

        # Fix phrase_category
        current_cat = row.get('phrase_category')
        if current_cat is None or current_cat == 'unknown':
            correct_cat = classify_phrase_category(text)
            if correct_cat != 'unknown':
                fixes['phrase_category'] = correct_cat
                stats['phrase_cat'] += 1
                needs_fix = True

        # Fix is_real_word
        current_real = row.get('is_real_word', False)
        if not current_real:
            is_real = is_real_word_strict(text)
            if is_real:
                fixes['is_real_word'] = True
                stats['real_word'] += 1
                needs_fix = True

        # Fix max_phi
        avg_phi = row.get('avg_phi', 0) or 0
        max_phi = row.get('max_phi', 0) or 0
        if max_phi == 0 and avg_phi > 0:
            fixes['max_phi'] = avg_phi
            stats['max_phi'] += 1
            needs_fix = True

        # Fix efficiency_gain
        efficiency = row.get('efficiency_gain', 0) or 0
        frequency = row.get('frequency', 1) or 1
        if efficiency == 0 and avg_phi > 0:
            correct_efficiency = compute_efficiency_gain(avg_phi, frequency)
            if correct_efficiency > 0:
                fixes['efficiency_gain'] = correct_efficiency
                stats['efficiency'] += 1
                needs_fix = True

        if needs_fix:
            to_fix.append((row['id'], fixes))

    print(f"\nRows needing fixes: {len(to_fix)}")
    print(f"  - type field: {stats['type']}")
    print(f"  - phrase_category: {stats['phrase_cat']}")
    print(f"  - is_real_word (false->true): {stats['real_word']}")
    print(f"  - max_phi (0->avg_phi): {stats['max_phi']}")
    print(f"  - efficiency_gain: {stats['efficiency']}")

    if dry_run:
        print("\nDRY RUN - showing first 10 fixes:")
        for row_id, fixes in to_fix[:10]:
            print(f"  id={row_id}: {fixes}")
        conn.close()
        return

    if not to_fix:
        print("No rows need fixing")
        conn.close()
        return

    # Apply fixes
    success = 0
    errors = 0

    with conn.cursor() as cur:
        for i in range(0, len(to_fix), BATCH_SIZE):
            batch = to_fix[i:i+BATCH_SIZE]

            for row_id, fixes in batch:
                try:
                    set_parts = []
                    params = []

                    for col, val in fixes.items():
                        set_parts.append(f"{col} = %s")
                        params.append(val)

                    if set_parts:
                        params.append(row_id)
                        cur.execute(f"""
                            UPDATE vocabulary_observations
                            SET {', '.join(set_parts)}
                            WHERE id = %s
                        """, params)
                        success += 1

                except Exception as e:
                    print(f"Error fixing id={row_id}: {e}")
                    errors += 1

            conn.commit()
            print(f"Progress: {min(i + BATCH_SIZE, len(to_fix))}/{len(to_fix)} (success={success}, errors={errors})")

    print(f"\nCompleted: {success} fixed, {errors} errors")
    conn.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Fix vocabulary_observations table issues')
    parser.add_argument('--limit', type=int, default=0,
                        help='Maximum rows to process (0 = all)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be updated without making changes')
    args = parser.parse_args()

    fix_vocabulary_observations(limit=args.limit, dry_run=args.dry_run)
