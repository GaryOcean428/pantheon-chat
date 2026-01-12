#!/usr/bin/env python3
"""
Backfill learned_words table with complete metadata.

This script populates NULL columns in learned_words:
- basin_coords: Computed via coordizer (QIG-pure)
- phrase_category: Classified via QIGPhraseClassifier
- is_integrated: Set TRUE for words with valid basins
- qfi_score, basin_distance, curvature_std, entropy_score: Computed from basin
- integrated_at: Set to NOW() for newly integrated words

Usage:
    python scripts/backfill_learned_words.py [--batch-size 100] [--limit 1000] [--dry-run]
"""

import os
import sys
import argparse
import numpy as np
import logging
from typing import Optional, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_connection():
    """Get database connection."""
    import psycopg2
    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        raise RuntimeError("DATABASE_URL not set")
    return psycopg2.connect(db_url)


def compute_entropy(basin: np.ndarray) -> float:
    """Compute Shannon entropy of basin embedding."""
    probs = np.abs(basin) / (np.sum(np.abs(basin)) + 1e-10)
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    return float(entropy)


def classify_phrase(word: str, basin: np.ndarray) -> str:
    """Classify phrase category using QIG classifier or heuristics."""
    try:
        from qig_phrase_classifier import classify_phrase_qig_pure
        category, _ = classify_phrase_qig_pure(word, basin)
        return category
    except Exception:
        if word.isupper() and len(word) > 1:
            return 'ACRONYM'
        elif len(word) > 0 and word[0].isupper() and not word.isupper():
            return 'PROPER_NOUN'
        elif len(word) >= 3 and word.isalpha():
            return 'COMMON_WORD'
        else:
            return 'UNKNOWN'


def compute_qfi(basin: np.ndarray) -> float:
    """Compute Quantum Fisher Information score using trace-based QFI.
    
    QIG-pure formula: QFI = tr(|basin><basin|) = trace of outer product
    This measures the information content of the basin embedding.
    """
    qfi = np.trace(np.outer(basin, basin))
    return float(qfi)


def qfi_to_phi(qfi: float) -> float:
    """Convert QFI to phi score."""
    phi = 0.5 + 0.3 * np.tanh(np.log10(abs(qfi) + 1e-10))
    return max(0.0, min(1.0, float(phi)))


def get_words_needing_backfill(conn, limit: int = 1000) -> List[Tuple[str, Optional[List[float]]]]:
    """Get words that need backfill (missing basin_coords, phrase_category, or is_integrated)."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT word, basin_coords 
            FROM learned_words
            WHERE (is_integrated IS NULL OR is_integrated = FALSE)
               OR phrase_category IS NULL
               OR basin_coords IS NULL
            ORDER BY frequency DESC NULLS LAST
            LIMIT %s
        """, (limit,))
        return [(row[0], row[1]) for row in cur.fetchall()]


def get_basin_from_tokenizer(conn, word: str) -> Optional[np.ndarray]:
    """Try to get basin from tokenizer_vocabulary first."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT basin_embedding FROM tokenizer_vocabulary
            WHERE token = %s AND basin_embedding IS NOT NULL
            LIMIT 1
        """, (word,))
        row = cur.fetchone()
        if row and row[0]:
            return np.array(row[0], dtype=np.float64)
    return None


def compute_basin_via_coordizer(word: str) -> Optional[np.ndarray]:
    """Compute basin via coordizer (QIG-pure)."""
    try:
        from coordizers.fallback_vocabulary import compute_basin_embedding
        basin = compute_basin_embedding(word)
        if isinstance(basin, np.ndarray) and basin.shape == (64,):
            return basin
    except Exception as e:
        logger.warning(f"Coordizer failed for '{word}': {e}")
    return None


def parse_basin(basin_data) -> Optional[np.ndarray]:
    """Parse basin data from various database formats."""
    if basin_data is None:
        return None
    try:
        if isinstance(basin_data, (list, tuple)):
            arr = np.array(basin_data, dtype=np.float64)
        elif isinstance(basin_data, str):
            clean = basin_data.strip('[](){}')
            arr = np.array([float(x) for x in clean.split(',')], dtype=np.float64)
        elif isinstance(basin_data, np.ndarray):
            arr = basin_data
        else:
            arr = np.array(list(basin_data), dtype=np.float64)
        
        if len(arr) == 64:
            return arr
    except Exception:
        pass
    return None


def backfill_word(conn, word: str, existing_basin, dry_run: bool = False) -> bool:
    """Backfill a single word with complete metadata."""
    try:
        basin = parse_basin(existing_basin)
        if basin is None:
            basin = get_basin_from_tokenizer(conn, word)
            if basin is None:
                basin = compute_basin_via_coordizer(word)
        
        if basin is None:
            logger.warning(f"Could not compute basin for '{word}'")
            return False
        
        basin_distance = float(np.linalg.norm(basin))
        curvature_std = float(np.std(basin))
        entropy_score = compute_entropy(basin)
        qfi_score = compute_qfi(basin)
        phi_score = qfi_to_phi(qfi_score)
        phrase_category = classify_phrase(word, basin)
        
        basin_str = '[' + ','.join(str(x) for x in basin.tolist()) + ']'
        
        if dry_run:
            logger.info(f"[DRY RUN] Would update '{word}': category={phrase_category}, qfi={qfi_score:.4f}")
            return True
        
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE learned_words SET
                    basin_coords = COALESCE(basin_coords, %s::vector(64)),
                    is_integrated = TRUE,
                    integrated_at = COALESCE(integrated_at, NOW()),
                    qfi_score = COALESCE(qfi_score, %s),
                    basin_distance = COALESCE(basin_distance, %s),
                    curvature_std = COALESCE(curvature_std, %s),
                    entropy_score = COALESCE(entropy_score, %s),
                    is_geometrically_valid = TRUE,
                    phrase_category = COALESCE(phrase_category, %s),
                    avg_phi = COALESCE(avg_phi, %s),
                    max_phi = COALESCE(max_phi, %s)
                WHERE word = %s
            """, (basin_str, qfi_score, basin_distance, curvature_std, 
                  entropy_score, phrase_category, phi_score, phi_score, word))
        
        return True
    except Exception as e:
        logger.error(f"Failed to backfill '{word}': {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return False


def run_backfill(batch_size: int = 100, limit: int = 1000, dry_run: bool = False):
    """Run the backfill process."""
    conn = get_connection()
    
    words = get_words_needing_backfill(conn, limit)
    logger.info(f"Found {len(words)} words needing backfill")
    
    if not words:
        logger.info("No words to backfill")
        return
    
    success_count = 0
    fail_count = 0
    
    for i, (word, existing_basin) in enumerate(words):
        if backfill_word(conn, word, existing_basin, dry_run):
            success_count += 1
        else:
            fail_count += 1
        
        if (i + 1) % batch_size == 0:
            if not dry_run:
                conn.commit()
            logger.info(f"Progress: {i + 1}/{len(words)} (success={success_count}, fail={fail_count})")
    
    if not dry_run:
        conn.commit()
    
    conn.close()
    
    logger.info(f"\nBackfill complete: {success_count} succeeded, {fail_count} failed")


def main():
    parser = argparse.ArgumentParser(description='Backfill learned_words metadata')
    parser.add_argument('--batch-size', type=int, default=100, help='Commit every N words')
    parser.add_argument('--limit', type=int, default=1000, help='Maximum words to process')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    
    args = parser.parse_args()
    
    logger.info(f"Starting backfill: batch_size={args.batch_size}, limit={args.limit}, dry_run={args.dry_run}")
    run_backfill(batch_size=args.batch_size, limit=args.limit, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
