#!/usr/bin/env python3
"""
Fast Migration: 32K Checkpoint → tokenizer_vocabulary

Uses PostgreSQL COPY for bulk insertion (100x faster than INSERT).
"""

import os
import json
import io
import numpy as np
from dotenv import load_dotenv

load_dotenv(dotenv_path='../.env')

import psycopg2
from psycopg2 import sql

CHECKPOINT_PATH = '../shared/coordizer/checkpoint_32000.json'

def normalize_vector(v):
    """Normalize to unit sphere."""
    arr = np.array(v, dtype=np.float64)
    norm = np.linalg.norm(arr)
    if norm > 1e-10:
        arr = arr / norm
    return arr.tolist()

def compute_phi(v):
    """Compute phi from vector entropy."""
    arr = np.abs(np.array(v)) + 1e-10
    arr = arr / arr.sum()
    entropy = -np.sum(arr * np.log(arr))
    return float(np.clip(entropy / np.log(64), 0, 1))

def main():
    print("Loading checkpoint...")
    with open(CHECKPOINT_PATH) as f:
        data = json.load(f)
    
    vocab = data.get('vocab', {})
    print(f"Loaded {len(vocab)} tokens")
    
    # Connect to database
    conn = psycopg2.connect(os.getenv('DATABASE_URL'))
    cur = conn.cursor()
    
    # Get current max token_id
    cur.execute("SELECT COALESCE(MAX(token_id), 0) FROM tokenizer_vocabulary")
    max_id = cur.fetchone()[0]
    print(f"Current max token_id: {max_id}")
    
    # Get existing tokens to avoid duplicates
    cur.execute("SELECT token FROM tokenizer_vocabulary")
    existing = {row[0] for row in cur.fetchall()}
    print(f"Existing tokens: {len(existing)}")
    
    # Track tokens we're adding to avoid duplicates within checkpoint
    adding = set()
    
    # Prepare data for COPY
    print("Preparing data...")
    buffer = io.StringIO()
    new_count = 0
    
    for idx_str, token_data in vocab.items():
        token = token_data.get('name', f'<token_{idx_str}>')
        vector = token_data.get('vector', [])
        
        if token in existing or token in adding:
            continue
        if not vector or len(vector) != 64:
            continue
        
        adding.add(token)
        
        max_id += 1
        new_count += 1
        
        # Normalize and compute phi
        coords = normalize_vector(vector)
        phi = compute_phi(vector)
        weight = phi * 1.5
        
        # Determine source_type
        if token.startswith('<byte_'):
            source_type = 'byte_level'
        elif '+' in token or '<byte_' in token:
            source_type = 'checkpoint_byte'
        else:
            source_type = 'checkpoint_word'
        
        # Format vector for PostgreSQL
        basin_str = '[' + ','.join(f'{x:.8f}' for x in coords) + ']'
        
        # Write tab-separated line: token, token_id, weight, frequency, phi_score, basin_embedding, source_type
        # Escape special characters in token and skip problematic tokens
        if '\x00' in token or len(token) > 100:
            continue  # Skip null bytes and extremely long tokens
        safe_token = token.replace('\\', '\\\\').replace('\t', '\\t').replace('\n', '\\n').replace('\r', '\\r')
        buffer.write(f"{safe_token}\t{max_id}\t{weight:.6f}\t1\t{phi:.6f}\t{basin_str}\t{source_type}\n")
    
    print(f"New tokens to insert: {new_count}")
    
    if new_count == 0:
        print("No new tokens to insert!")
        conn.close()
        return
    
    # Use COPY for bulk insert
    print("Bulk inserting with COPY...")
    buffer.seek(0)
    
    cur.copy_expert(
        "COPY tokenizer_vocabulary (token, token_id, weight, frequency, phi_score, basin_embedding, source_type) FROM STDIN WITH (FORMAT text)",
        buffer
    )
    
    conn.commit()
    
    # Verify
    cur.execute("SELECT COUNT(*) FROM tokenizer_vocabulary")
    total = cur.fetchone()[0]
    print(f"✅ Migration complete! Total tokens: {total}")
    
    # Show sample
    cur.execute("""
        SELECT source_type, COUNT(*) 
        FROM tokenizer_vocabulary 
        GROUP BY source_type 
        ORDER BY COUNT(*) DESC
    """)
    print("\nTokens by source:")
    for row in cur.fetchall():
        print(f"  {row[0]}: {row[1]}")
    
    conn.close()

if __name__ == '__main__':
    main()
