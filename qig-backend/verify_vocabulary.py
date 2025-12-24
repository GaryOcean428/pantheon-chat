#!/usr/bin/env python3
"""Verify tokenizer_vocabulary has real words."""

import os
from pathlib import Path
import psycopg2
from dotenv import load_dotenv

load_dotenv()
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

def verify():
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        print("ERROR: DATABASE_URL not set")
        return
    
    conn = psycopg2.connect(database_url)
    cur = conn.cursor()
    
    # Count by source type
    print("=== Vocabulary Distribution ===")
    cur.execute("""
        SELECT source_type, COUNT(*) as count
        FROM tokenizer_vocabulary
        GROUP BY source_type
        ORDER BY count DESC
    """)
    for row in cur.fetchall():
        print(f"  {row[0]}: {row[1]}")
    
    # Count real words (alphabetic, 3+ chars)
    print("\n=== Real Words ===")
    cur.execute("""
        SELECT COUNT(*) 
        FROM tokenizer_vocabulary 
        WHERE source_type IN ('bip39', 'common', 'learned')
          AND LENGTH(token) >= 3
          AND token ~ '^[a-zA-Z]+$'
    """)
    real_count = cur.fetchone()[0]
    print(f"Real English words: {real_count}")
    
    # Sample words
    print("\n=== Sample Words ===")
    cur.execute("""
        SELECT token, source_type, phi_score, basin_embedding IS NOT NULL as has_embedding
        FROM tokenizer_vocabulary
        WHERE source_type IN ('bip39', 'common')
          AND LENGTH(token) >= 3
        ORDER BY phi_score DESC
        LIMIT 20
    """)
    for row in cur.fetchall():
        print(f"  {row[0]}: source={row[1]}, phi={row[2]:.3f}, has_embedding={row[3]}")
    
    # Check for BPE fragments
    print("\n=== BPE Fragment Check ===")
    cur.execute("""
        SELECT COUNT(*) 
        FROM tokenizer_vocabulary 
        WHERE source_type IN ('byte_level', 'checkpoint_byte')
           OR token LIKE '<%'
           OR token LIKE '%+%'
    """)
    bpe_count = cur.fetchone()[0]
    if bpe_count > 0:
        print(f"WARNING: {bpe_count} BPE fragments still in database")
    else:
        print("No BPE fragments found ✓")
    
    conn.close()
    print("\n=== Verification Complete ===")

if __name__ == '__main__':
    verify()
