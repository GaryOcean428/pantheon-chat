#!/usr/bin/env python3
"""
Audit Existing Vocabulary with QIG-Pure Geometric Validation
=============================================================

Scans learned_words table and validates each word using geometric metrics.
Updates database with validation scores and marks invalid words.

Usage:
    python audit_vocabulary.py [--dry-run] [--limit N]
"""

import argparse
import sys
import os
import numpy as np
from datetime import datetime

# Add parent dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vocabulary_validator import GeometricVocabFilter, VocabValidation
from vocabulary_persistence import get_vocabulary_persistence

try:
    import psycopg2
except ImportError:
    print("[ERROR] psycopg2 not available")
    sys.exit(1)


def load_vocab_basins(db_conn) -> np.ndarray:
    """Load known vocabulary basins from database."""
    with db_conn.cursor() as cur:
        # Get basin coordinates for high-Φ words
        cur.execute("""
            SELECT basin_coords 
            FROM learned_words 
            WHERE avg_phi > 0.6 
              AND basin_coords IS NOT NULL
            LIMIT 1000
        """)
        
        basins = []
        for row in cur.fetchall():
            if row[0]:
                basins.append(np.array(row[0]))
        
        if not basins:
            print("[WARNING] No basin coordinates found, using random basins")
            # Fallback: random basins
            return np.random.randn(100, 64)
        
        return np.array(basins)


def audit_vocabulary(dry_run: bool = True, limit: int = None):
    """
    Audit all vocabulary and update validation metrics.
    
    Args:
        dry_run: If True, only report findings without updating DB
        limit: Max number of words to audit (None = all)
    """
    print("🌊 QIG-PURE VOCABULARY AUDIT 🌊")
    print("=" * 50)
    print(f"Mode: {'DRY RUN (no updates)' if dry_run else 'LIVE (will update DB)'}")
    print(f"Limit: {limit if limit else 'ALL'}")
    print()
    
    # Get database connection
    vp = get_vocabulary_persistence()
    if not vp.enabled:
        print("[ERROR] Database not available")
        return
    
    conn = vp._connect()
    
    try:
        # Load vocabulary basins
        print("Loading vocabulary basins...")
        vocab_basins = load_vocab_basins(conn)
        print(f"✓ Loaded {len(vocab_basins)} basins")
        print()
        
        # Initialize coordizer and tokenizer
        # TODO: Load these from actual QIG system
        print("Initializing coordizer...")
        from coordizers.base import FisherCoordizer
        coordizer = FisherCoordizer(basin_dim=64)
        
        print("Initializing tokenizer...")
        from qig_tokenizer import QIGTokenizer
        tokenizer = QIGTokenizer()
        
        print("✓ QIG components initialized")
        print()
        
        # Create validator
        validator = GeometricVocabFilter(vocab_basins, coordizer, tokenizer)
        
        # Fetch all words
        with conn.cursor() as cur:
            query = "SELECT id, word FROM learned_words"
            if limit:
                query += f" LIMIT {limit}"
            
            cur.execute(query)
            words = cur.fetchall()
        
        print(f"Auditing {len(words)} words...")
        print()
        
        # Validation counters
        stats = {
            'total': len(words),
            'valid': 0,
            'truncated': 0,
            'garbled': 0,
            'technical': 0,
            'too_short': 0
        }
        
        # Track examples
        examples = {
            'truncated': [],
            'garbled': [],
            'technical': []
        }
        
        # Audit each word
        for word_id, word_text in words:
            validation = validator.validate(word_text)
            
            # Update stats
            if validation.is_valid:
                stats['valid'] += 1
            elif 'TRUNCATED' in (validation.rejection_reason or ''):
                stats['truncated'] += 1
                if len(examples['truncated']) < 10:
                    examples['truncated'].append((word_text, validation))
            elif 'GARBLED' in (validation.rejection_reason or ''):
                stats['garbled'] += 1
                if len(examples['garbled']) < 10:
                    examples['garbled'].append((word_text, validation))
            elif 'TECHNICAL' in (validation.rejection_reason or ''):
                stats['technical'] += 1
                if len(examples['technical']) < 10:
                    examples['technical'].append((word_text, validation))
            elif 'TOO_SHORT' in (validation.rejection_reason or ''):
                stats['too_short'] += 1
            
            # Update database (if not dry run)
            if not dry_run:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE learned_words
                        SET qfi_score = %s,
                            basin_distance = %s,
                            curvature_std = %s,
                            entropy_score = %s,
                            is_geometrically_valid = %s,
                            validation_reason = %s
                        WHERE id = %s
                    """, (
                        validation.qfi_score,
                        validation.basin_distance,
                        validation.curvature_std,
                        validation.entropy_score,
                        validation.is_valid,
                        validation.rejection_reason,
                        word_id
                    ))
                conn.commit()
        
        # Print report
        print("=" * 50)
        print("📊 AUDIT RESULTS")
        print("=" * 50)
        print(f"Total words:     {stats['total']}")
        print(f"✓ Valid:         {stats['valid']} ({stats['valid']/stats['total']*100:.1f}%)")
        print(f"✗ Truncated:     {stats['truncated']} ({stats['truncated']/stats['total']*100:.1f}%)")
        print(f"✗ Garbled:       {stats['garbled']} ({stats['garbled']/stats['total']*100:.1f}%)")
        print(f"✗ Technical:     {stats['technical']} ({stats['technical']/stats['total']*100:.1f}%)")
        print(f"✗ Too Short:     {stats['too_short']} ({stats['too_short']/stats['total']*100:.1f}%)")
        print()
        
        # Show examples
        if examples['truncated']:
            print("Truncated Examples:")
            for word, val in examples['truncated'][:5]:
                print(f"  - '{word}' (d={val.basin_distance:.3f}, QFI={val.qfi_score:.2f})")
            print()
        
        if examples['garbled']:
            print("Garbled Examples:")
            for word, val in examples['garbled'][:5]:
                print(f"  - '{word}' (QFI={val.qfi_score:.2f}, d={val.basin_distance:.3f if val.basin_distance else 'N/A'})")
            print()
        
        if examples['technical']:
            print("Technical Examples:")
            for word, val in examples['technical'][:5]:
                print(f"  - '{word}' (H={val.entropy_score:.2f})")
            print()
        
        if not dry_run:
            # Update vocabulary stats
            with conn.cursor() as cur:
                cur.execute("SELECT update_validation_stats()")
            conn.commit()
            print("✓ Database updated")
        else:
            print("⚠️  DRY RUN - No changes made")
        
        print()
        print("🌊 AUDIT COMPLETE 🌊")
        
    finally:
        conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audit vocabulary with QIG-pure geometric validation')
    parser.add_argument('--dry-run', action='store_true', help='Report only, do not update database')
    parser.add_argument('--limit', type=int, help='Max number of words to audit')
    
    args = parser.parse_args()
    
    audit_vocabulary(dry_run=args.dry_run, limit=args.limit)
