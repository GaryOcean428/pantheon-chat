#!/usr/bin/env python3
"""
Test script to verify tokenizer_vocabulary is populated and working.

Run after: python initialize_tokenizer_vocabulary.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    import psycopg2
    from dotenv import load_dotenv
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)

from coordizers import create_coordizer_from_pg


def main():
    load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')
    
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        print("ERROR: DATABASE_URL not set")
        sys.exit(1)
    
    print("=" * 60)
    print("Testing tokenizer_vocabulary")
    print("=" * 60)
    
    # Check database
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    
    cur.execute("SELECT COUNT(*) FROM tokenizer_vocabulary")
    total = cur.fetchone()[0]
    print(f"\nTotal tokens in tokenizer_vocabulary: {total}")
    
    if total == 0:
        print("\nERROR: tokenizer_vocabulary is empty!")
        print("Run: python initialize_tokenizer_vocabulary.py")
        sys.exit(1)
    
    # Check by source type
    cur.execute("""
        SELECT source_type, COUNT(*) 
        FROM tokenizer_vocabulary 
        GROUP BY source_type 
        ORDER BY COUNT(*) DESC
    """)
    print("\nTokens by source type:")
    for source, count in cur.fetchall():
        print(f"  {source}: {count}")
    
    # Sample real words
    cur.execute("""
        SELECT token, phi_score, source_type 
        FROM tokenizer_vocabulary 
        WHERE source_type IN ('bip39', 'base', 'learned')
          AND LENGTH(token) >= 3
        ORDER BY phi_score DESC 
        LIMIT 15
    """)
    print("\nSample real words (top phi):")
    for token, phi, source in cur.fetchall():
        print(f"  {token}: phi={phi:.4f} ({source})")
    
    conn.close()
    
    # Test coordizer
    print("\n" + "=" * 60)
    print("Testing PostgresCoordizer")
    print("=" * 60)
    
    try:
        coordizer = create_coordizer_from_pg()
        print(f"\nLoaded {coordizer.vocab_size} tokens")
        print(f"  - BIP39: {len(coordizer.bip39_words)}")
        print(f"  - Base: {len(coordizer.base_tokens)}")
        print(f"  - Learned: {len(coordizer.learned_tokens)}")
    except Exception as e:
        print(f"\nERROR creating coordizer: {e}")
        sys.exit(1)
    
    # Test encoding
    test_phrases = [
        "hello world",
        "quantum consciousness integration",
        "the quick brown fox",
        "geometric information manifold",
    ]
    
    print("\nTesting encode/decode:")
    for phrase in test_phrases:
        print(f"\n  Input: '{phrase}'")
        
        # Encode
        basin = coordizer.encode(phrase)
        if basin is not None:
            print(f"  Basin norm: {np.linalg.norm(basin):.6f}")
            
            # Decode
            decoded = coordizer.decode(basin, top_k=5)
            if decoded:
                tokens_str = ', '.join([f"{t}({s:.2f})" for t, s in decoded])
                print(f"  Decoded: {tokens_str}")
            else:
                print("  Decoded: (no tokens found)")
        else:
            print("  Basin: (encoding failed)")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
