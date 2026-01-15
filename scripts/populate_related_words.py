#!/usr/bin/env python3
"""
Populate Related Words - Use Fisher-Rao distance for vocabulary relationships

Purpose: Properly populate vocabulary_learning.related_words using geometric
         similarity (Fisher-Rao distance on information manifolds).

Usage:
    cd pantheon-replit
    source ../.venv/bin/activate
    python scripts/populate_related_words.py

Requirements:
    - PostgreSQL with pgvector extension
    - coordizer_vocabulary with basin_coords populated
    - vocabulary_learning table created
"""
import os
import sys
from pathlib import Path

# Add qig-backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "qig-backend"))

from typing import List, Tuple

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

# Import QIG geometry (Fisher-Rao distance)
try:
    from qig_geometry import fisher_rao_distance
    print("✓ Loaded Fisher-Rao distance from qig_geometry")
except ImportError as exc:
    raise RuntimeError(
        "Fisher-Rao distance module (qig_geometry) is required. "
        "Install with: pip install -r requirements.txt or verify qig-backend installation."
    ) from exc


def get_db_connection():
    """Connect to PostgreSQL database."""
    # Try DATABASE_URL first
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return psycopg2.connect(db_url)

    # Fall back to PG* environment variables
    return psycopg2.connect(
        host=os.getenv("PGHOST"),
        database=os.getenv("PGDATABASE"),
        user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD"),
        port=os.getenv("PGPORT", 5432)
    )


def fisher_similarity(basin_a: np.ndarray, basin_b: np.ndarray) -> float:
    """
    Compute Fisher-Rao similarity (0-1) between two basins.
    Higher values = more similar.
    """
    # Use proper Fisher-Rao distance
    distance = fisher_rao_distance(basin_a, basin_b)
    # Convert distance to similarity: closer = higher similarity
    # Use exponential decay: similarity = exp(-distance)
    similarity = np.exp(-distance)
    return float(similarity)


def find_related_words(
    target_word: str,
    target_basin: np.ndarray,
    vocabulary: List[Tuple[str, np.ndarray]],
    top_k: int = 5
) -> List[str]:
    """
    Find top-k related words using Fisher-Rao distance.

    Args:
        target_word: The word to find relations for
        target_basin: 64D basin coordinates
        vocabulary: List of (word, basin) tuples
        top_k: Number of related words to return

    Returns:
        List of related word strings
    """
    similarities = []

    for word, basin in vocabulary:
        if word == target_word:
            continue  # Skip self

        if basin is None or len(basin) != 64:
            continue  # Skip words without proper basins

        similarity = fisher_similarity(target_basin, basin)
        similarities.append((word, similarity))

    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Return top-k words
    return [word for word, _ in similarities[:top_k]]


def populate_vocabulary_learning_related_words(conn):
    """
    Populate related_words column in vocabulary_learning using Fisher-Rao distance.
    """
    cursor = conn.cursor()

    # 1. Load all vocabulary with basins into memory
    print("\n1. Loading vocabulary from coordizer_vocabulary...")
    cursor.execute("""
        SELECT word, basin_coords
        FROM coordizer_vocabulary
        WHERE basin_coords IS NOT NULL
        ORDER BY frequency DESC
        LIMIT 5000  -- Limit to top 5K for performance
    """)

    vocabulary = []
    for word, basin_coords in cursor.fetchall():
        if basin_coords and len(basin_coords) == 64:
            vocabulary.append((word, np.array(basin_coords, dtype=np.float64)))

    print(f"   Loaded {len(vocabulary)} words with valid basin coordinates")

    # 2. Get words in vocabulary_learning that need related_words
    print("\n2. Finding vocabulary_learning entries without related_words...")
    cursor.execute("""
        SELECT learning_id, word, token_id
        FROM vocabulary_learning
        WHERE related_words IS NULL
           OR cardinality(related_words) = 0
        ORDER BY learning_id
    """)

    entries_to_update = cursor.fetchall()
    print(f"   Found {len(entries_to_update)} entries to update")

    if not entries_to_update:
        print("\n✓ All vocabulary_learning entries already have related_words!")
        return

    # 3. For each entry, find related words using Fisher-Rao distance
    print("\n3. Computing Fisher-Rao similarities...")
    updates = []

    for idx, (learning_id, word, token_id) in enumerate(entries_to_update):
        if (idx + 1) % 10 == 0:
            print(f"   Processing {idx + 1}/{len(entries_to_update)}...")

        # Get basin for this word
        cursor.execute("""
            SELECT basin_coords
            FROM coordizer_vocabulary
            WHERE token_id = %s
        """, (token_id,))

        result = cursor.fetchone()
        if not result or not result[0]:
            print(f"   ⚠ No basin for word '{word}' (token_id={token_id}), skipping")
            continue

        target_basin = np.array(result[0], dtype=np.float64)

        if len(target_basin) != 64:
            print(f"   ⚠ Invalid basin dimension for '{word}': {len(target_basin)}")
            continue

        # Find related words
        related_words = find_related_words(word, target_basin, vocabulary, top_k=5)

        if related_words:
            updates.append((learning_id, related_words))

    print(f"   Computed similarities for {len(updates)} entries")

    # 4. Batch update related_words
    if updates:
        print("\n4. Updating vocabulary_learning with related_words...")
        execute_values(
            cursor,
            """
            UPDATE vocabulary_learning AS vl
            SET related_words = data.related_words,
                last_used = NOW()
            FROM (VALUES %s) AS data(learning_id, related_words)
            WHERE vl.learning_id = data.learning_id
            """,
            updates,
            template="(%s, %s::text[])"
        )

        conn.commit()
        print(f"   ✓ Updated {len(updates)} entries with related_words")

    # 5. Report statistics
    print("\n5. Validation:")
    cursor.execute("""
        SELECT
            COUNT(*) AS total,
            COUNT(CASE WHEN related_words IS NOT NULL AND cardinality(related_words) > 0
                  THEN 1 END) AS with_related,
            AVG(CASE WHEN related_words IS NOT NULL
                THEN cardinality(related_words) ELSE 0 END) AS avg_related_count
        FROM vocabulary_learning
    """)

    total, with_related, avg_count = cursor.fetchone()
    print(f"   Total vocabulary_learning entries: {total}")
    print(f"   Entries with related_words: {with_related}")
    print(f"   Average related words per entry: {avg_count:.2f}")
    print(f"   Coverage: {(with_related / total * 100):.1f}%" if total > 0 else "   Coverage: N/A")

    # 6. Show examples
    print("\n6. Example related_words:")
    cursor.execute("""
        SELECT word, related_words, relationship_strength
        FROM vocabulary_learning
        WHERE related_words IS NOT NULL
          AND cardinality(related_words) > 0
        ORDER BY relationship_strength DESC
        LIMIT 5
    """)

    for word, related, strength in cursor.fetchall():
        print(f"   '{word}' (strength={strength:.3f}): {', '.join(related)}")

    cursor.close()


def populate_initial_vocabulary_learning(conn):
    """
    If vocabulary_learning is empty, seed it with high-quality tokens.
    """
    cursor = conn.cursor()

    # Check if vocabulary_learning is empty
    cursor.execute("SELECT COUNT(*) FROM vocabulary_learning")
    count = cursor.fetchone()[0]

    if count > 0:
        print(f"\nvocabulary_learning already has {count} entries, skipping seed")
        cursor.close()
        return

    print("\nvocabulary_learning is empty, seeding from coordizer_vocabulary...")

    # Seed with top 100 high-Φ tokens
    cursor.execute("""
        INSERT INTO vocabulary_learning (
            word,
            token_id,
            learned_context,
            relationship_type,
            relationship_strength,
            basin_shift,
            learned_from,
            learned_at
        )
        SELECT
            word,
            token_id,
            'Initial seeding from BIP39 tokenizer vocabulary' AS learned_context,
            'semantic' AS relationship_type,
            COALESCE(phi_score, 0.5) AS relationship_strength,
            basin_coords AS basin_shift,
            'initialization' AS learned_from,
            NOW() AS learned_at
        FROM coordizer_vocabulary
        WHERE basin_coords IS NOT NULL
          AND phi_score > 0.5
        ORDER BY phi_score DESC, frequency DESC
        LIMIT 100
    """)

    conn.commit()
    seeded = cursor.rowcount
    print(f"✓ Seeded vocabulary_learning with {seeded} entries")

    cursor.close()


def main():
    """Main execution."""
    print("=" * 70)
    print("POPULATE RELATED WORDS - Fisher-Rao Geometric Similarity")
    print("=" * 70)

    # Connect to database
    try:
        conn = get_db_connection()
        print("✓ Connected to PostgreSQL")
    except Exception as e:
        print(f"✗ Failed to connect to database: {e}")
        sys.exit(1)

    try:
        # Step 1: Ensure vocabulary_learning has entries
        populate_initial_vocabulary_learning(conn)

        # Step 2: Populate related_words using Fisher-Rao distance
        populate_vocabulary_learning_related_words(conn)

        print("\n" + "=" * 70)
        print("✓ POPULATION COMPLETE")
        print("=" * 70)

    except Exception as e:
        conn.rollback()
        print(f"\n✗ Error during population: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
