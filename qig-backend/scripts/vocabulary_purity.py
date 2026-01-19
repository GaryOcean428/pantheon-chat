#!/usr/bin/env python3
"""
Vocabulary Purity Script
========================

Validates and cleans vocabulary tables to ensure English-only words.
DRY: Uses centralized word_validation module.

Usage:
    python scripts/vocabulary_purity.py --check    # Dry run, show issues
    python scripts/vocabulary_purity.py --clean    # Actually clean
    python scripts/vocabulary_purity.py --stats    # Show statistics
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from word_validation import is_valid_english_word, is_likely_concatenated, is_likely_typo
from persistence.base_persistence import get_db_connection


def get_vocabulary_stats():
    """Get statistics about vocabulary tables."""
    conn = get_db_connection()
    if not conn:
        print("[ERROR] Could not connect to database")
        return None

    try:
        cur = conn.cursor()

        stats = {}

        cur.execute("SELECT COUNT(*) FROM coordizer_vocabulary")
        stats['tokenizer_total'] = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM bip39_words")
        stats['bip39_total'] = cur.fetchone()[0]

        return stats

    except Exception as e:
        print(f"[ERROR] Stats query failed: {e}")
        return None
    finally:
        conn.close()


def check_coordizer_vocabulary(limit: int = 100):
    """Check coordizer_vocabulary for invalid words."""
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, token, frequency, phi_score FROM coordizer_vocabulary")
        rows = cur.fetchall()
        
        invalid = []
        for id_, token, freq, phi in rows:
            if not is_valid_english_word(token, include_stop_words=True):
                reason = []
                if is_likely_concatenated(token):
                    reason.append('concatenated')
                if is_likely_typo(token):
                    reason.append('typo')
                if ' ' in token:
                    reason.append('multi-word')
                if any(c.isdigit() for c in token):
                    reason.append('has-digits')
                if len(token) > 18:
                    reason.append('too-long')
                if not reason:
                    reason.append('other')
                
                invalid.append({
                    'id': id_,
                    'token': token,
                    'freq': freq,
                    'phi': phi,
                    'reasons': reason
                })
        
        return invalid
        
    except Exception as e:
        print(f"[ERROR] Check failed: {e}")
        return []
    finally:
        conn.close()


def clean_coordizer_vocabulary(dry_run: bool = True):
    """Clean invalid entries from coordizer_vocabulary."""
    invalid = check_coordizer_vocabulary()
    
    if not invalid:
        print("[OK] No invalid entries found in coordizer_vocabulary")
        return 0
    
    print(f"[FOUND] {len(invalid)} invalid entries in coordizer_vocabulary")
    
    if dry_run:
        print("\n[DRY RUN] Would delete:")
        for entry in invalid[:500]:
            print(f"  - {entry['token'][:500]:30} ({', '.join(entry['reasons'])})")
        if len(invalid) > 20:
            print(f"  ... and {len(invalid) - 20} more")
        return len(invalid)
    
    conn = get_db_connection()
    if not conn:
        return 0
    
    try:
        cur = conn.cursor()
        ids_to_delete = [entry['id'] for entry in invalid]
        
        batch_size = 100
        deleted = 0
        for i in range(0, len(ids_to_delete), batch_size):
            batch = ids_to_delete[i:i + batch_size]
            placeholders = ','.join(['%s'] * len(batch))
            cur.execute(f"DELETE FROM coordizer_vocabulary WHERE id IN ({placeholders})", batch)
            deleted += cur.rowcount
        
        conn.commit()
        print(f"[CLEANED] Deleted {deleted} invalid entries from coordizer_vocabulary")
        return deleted
        
    except Exception as e:
        print(f"[ERROR] Cleanup failed: {e}")
        conn.rollback()
        return 0
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description='Vocabulary Purity Checker')
    parser.add_argument('--check', action='store_true', help='Check for invalid entries (dry run)')
    parser.add_argument('--clean', action='store_true', help='Clean invalid entries')
    parser.add_argument('--stats', action='store_true', help='Show statistics only')

    args = parser.parse_args()

    print("=" * 60)
    print("VOCABULARY PURITY CHECK")
    print("=" * 60)

    stats = get_vocabulary_stats()
    if stats:
        print(f"\nCurrent Statistics:")
        print(f"  BIP39 words:           {stats['bip39_total']:,}")
        print(f"  Coordizer vocabulary:  {stats['tokenizer_total']:,}")

    if args.stats:
        return

    print("\n" + "-" * 60)
    print("Checking coordizer_vocabulary...")
    tokenizer_invalid = check_coordizer_vocabulary()
    print(f"  Invalid entries: {len(tokenizer_invalid)}")

    if args.clean:
        print("\n" + "-" * 60)
        print("CLEANING (not dry run)")
        clean_coordizer_vocabulary(dry_run=False)
    elif args.check:
        print("\n" + "-" * 60)
        print("DRY RUN - showing what would be cleaned")
        clean_coordizer_vocabulary(dry_run=True)

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
