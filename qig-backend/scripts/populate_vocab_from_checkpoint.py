#!/usr/bin/env python3
"""
Populate tokenizer_vocabulary with 64D embeddings from checkpoint file.
"""
import json
import os
import sys
import psycopg2
from psycopg2.extras import execute_batch

import pathlib
WORKSPACE_ROOT = pathlib.Path(__file__).parent.parent.parent
CHECKPOINT_PATH = WORKSPACE_ROOT / "attached_assets/checkpoint_24000_1767012247122.json"

def main():
    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        print("[ERROR] DATABASE_URL not set")
        sys.exit(1)
    
    print(f"[PopulateVocab] Loading checkpoint from {CHECKPOINT_PATH}...")
    with open(CHECKPOINT_PATH, 'r') as f:
        checkpoint = json.load(f)
    
    vocab = checkpoint.get('vocab', {})
    basin_dim = checkpoint.get('basin_dim', 64)
    
    print(f"[PopulateVocab] Loaded {len(vocab)} tokens with {basin_dim}D embeddings")
    
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tokenizer_vocabulary (
            id SERIAL PRIMARY KEY,
            token TEXT NOT NULL UNIQUE,
            token_id INTEGER,
            phi_score REAL DEFAULT 0,
            weight REAL DEFAULT 1.0,
            embedding REAL[],
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            frequency INTEGER DEFAULT 1,
            basin_embedding vector(64),
            source_type VARCHAR(50) DEFAULT 'checkpoint',
            updated_at TIMESTAMP DEFAULT NOW()
        )
    """)
    conn.commit()
    
    batch = []
    skipped = 0
    for token_id_str, entry in vocab.items():
        token_id = int(token_id_str)
        name = entry.get('name', '')
        vector = entry.get('vector', [])
        scale = entry.get('scale', 'byte')
        
        if not name or len(name) < 1:
            skipped += 1
            continue
        
        if len(vector) != basin_dim:
            print(f"[WARN] Token {name} has wrong vector dim: {len(vector)}")
            continue
        
        batch.append({
            'token': name,
            'token_id': token_id,
            'embedding': vector,
            'source_type': f'checkpoint_{scale}',
            'phi_score': 0.5,
            'weight': 1.0 if scale == 'word' else 0.5
        })
    
    print(f"[PopulateVocab] Inserting {len(batch)} tokens (skipped {skipped})...")
    
    cur.execute("DELETE FROM tokenizer_vocabulary WHERE source_type LIKE 'checkpoint%'")
    deleted = cur.rowcount
    print(f"[PopulateVocab] Deleted {deleted} old checkpoint entries")
    
    insert_sql = """
        INSERT INTO tokenizer_vocabulary (token, token_id, embedding, source_type, phi_score, weight)
        VALUES (%(token)s, %(token_id)s, %(embedding)s, %(source_type)s, %(phi_score)s, %(weight)s)
        ON CONFLICT (token) DO UPDATE SET
            embedding = EXCLUDED.embedding,
            source_type = EXCLUDED.source_type,
            token_id = EXCLUDED.token_id,
            updated_at = NOW()
    """
    
    execute_batch(cur, insert_sql, batch, page_size=500)
    conn.commit()
    
    cur.execute("SELECT COUNT(*) FROM tokenizer_vocabulary WHERE embedding IS NOT NULL")
    count = cur.fetchone()[0]
    print(f"[PopulateVocab] SUCCESS: {count} tokens now have embeddings")
    
    cur.execute("""
        SELECT token, array_length(embedding, 1) as dim, source_type 
        FROM tokenizer_vocabulary 
        WHERE embedding IS NOT NULL 
        LIMIT 5
    """)
    for row in cur.fetchall():
        print(f"  Sample: '{row[0]}' dim={row[1]} source={row[2]}")
    
    cur.close()
    conn.close()
    print("[PopulateVocab] Done!")

if __name__ == "__main__":
    main()
