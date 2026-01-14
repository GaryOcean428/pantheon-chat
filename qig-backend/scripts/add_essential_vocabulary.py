#!/usr/bin/env python3
"""Add missing essential words (pronouns, prepositions, articles) to coordizer_vocabulary.

These fundamental function words are critical for proper phrase classification
and Fisher-Rao distance computation.
"""

import os
import sys
from pathlib import Path

_qig_backend = Path(__file__).parent.parent
if str(_qig_backend) not in sys.path:
    sys.path.insert(0, str(_qig_backend))

ESSENTIAL_WORDS = {
    'pronouns': ['i', 'me', 'my', 'mine', 'myself', 'you', 'your', 'yours', 'yourself',
                 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
                 'it', 'its', 'itself', 'we', 'us', 'our', 'ours', 'ourselves',
                 'they', 'them', 'their', 'theirs', 'themselves',
                 'who', 'whom', 'whose', 'which', 'what', 'that', 'this', 'these', 'those'],
    
    'prepositions': ['in', 'on', 'at', 'to', 'by', 'for', 'of', 'with', 'from',
                     'up', 'out', 'off', 'down', 'over', 'under', 'into', 'onto',
                     'through', 'during', 'before', 'after', 'between', 'among',
                     'about', 'against', 'above', 'below', 'beside', 'behind',
                     'beyond', 'near', 'around', 'across', 'along'],
    
    'articles': ['a', 'an', 'the'],
    
    'conjunctions': ['and', 'but', 'or', 'nor', 'so', 'yet', 'for', 'because',
                     'although', 'while', 'if', 'unless', 'until', 'when', 'where'],
    
    'auxiliary_verbs': ['is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
                        'do', 'does', 'did', 'have', 'has', 'had', 'having',
                        'will', 'would', 'shall', 'should', 'may', 'might',
                        'can', 'could', 'must'],
    
    'common_adverbs': ['not', 'no', 'yes', 'so', 'very', 'too', 'also', 'just',
                       'now', 'then', 'here', 'there', 'where', 'when', 'how', 'why'],
    
    'interjections': ['oh', 'ah', 'uh', 'um', 'hey', 'hi', 'wow', 'ouch', 'aha',
                      'alas', 'hurray', 'oops', 'yay', 'ugh'],
    
    'numbers': ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
                'nine', 'ten', 'hundred', 'thousand', 'million', 'billion', 'dozen'],
}


def format_vector(basin) -> str:
    """Format numpy array as pgvector string."""
    return '[' + ','.join(f'{x:.8f}' for x in basin) + ']'


def add_essential_vocabulary(dry_run: bool = False):
    """Add missing essential words to coordizer_vocabulary."""
    import psycopg2
    import numpy as np
    
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        print("ERROR: DATABASE_URL not set")
        return
    
    try:
        from coordizers.fallback_vocabulary import compute_basin_embedding
    except ImportError as e:
        print(f"ERROR: Cannot import compute_basin_embedding: {e}")
        return
    
    conn = psycopg2.connect(database_url)
    cur = conn.cursor()
    
    cur.execute("SELECT token FROM coordizer_vocabulary")
    existing_tokens = {row[0].lower() for row in cur.fetchall()}
    print(f"Existing vocabulary size: {len(existing_tokens)}")
    
    cur.execute("SELECT COALESCE(MAX(token_id), 0) FROM coordizer_vocabulary")
    max_token_id = cur.fetchone()[0]
    
    all_words = []
    for category, words in ESSENTIAL_WORDS.items():
        all_words.extend([(word, category) for word in words])
    
    missing_words = [(w, c) for w, c in all_words if w.lower() not in existing_tokens]
    print(f"Found {len(missing_words)} missing essential words")
    
    if dry_run:
        print("\nDRY RUN - Words to be added:")
        for word, category in missing_words[:30]:
            print(f"  {word} ({category})")
        if len(missing_words) > 30:
            print(f"  ... and {len(missing_words) - 30} more")
        cur.close()
        conn.close()
        return
    
    added = 0
    errors = 0
    next_token_id = max_token_id + 1
    
    for word, category in missing_words:
        try:
            basin = compute_basin_embedding(word)
            
            if basin is None or len(basin) != 64:
                print(f"  SKIP {word}: Invalid basin embedding")
                errors += 1
                continue
            
            basin_str = format_vector(basin)
            
            cur.execute("""
                INSERT INTO coordizer_vocabulary 
                    (token, token_id, basin_embedding, phi_score, frequency, source_type)
                VALUES (%s, %s, %s::vector, %s, %s, %s)
                ON CONFLICT (token) DO NOTHING
            """, (word.lower(), next_token_id, basin_str, 0.8, 1000, category))
            
            if cur.rowcount > 0:
                added += 1
                next_token_id += 1
                if added % 20 == 0:
                    conn.commit()
                    print(f"  Progress: {added} added, {errors} errors")
        
        except Exception as e:
            print(f"  ERROR adding '{word}': {e}")
            errors += 1
    
    conn.commit()
    
    print(f"\nCompleted: {added} added, {errors} errors")
    
    cur.execute("SELECT COUNT(*) FROM coordizer_vocabulary")
    new_count = cur.fetchone()[0]
    print(f"New vocabulary size: {new_count}")
    
    cur.close()
    conn.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Add essential vocabulary words')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be added without making changes')
    args = parser.parse_args()
    
    add_essential_vocabulary(dry_run=args.dry_run)
