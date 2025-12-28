#!/usr/bin/env python3
"""
Load Pre-trained 32K Coordizer into QIG System

This script imports a pre-trained coordizer with:
- 32,000 tokens with 64D unit-normalized basin embeddings
- 31,744 BPE merge rules
- Multi-scale tokens (char/subword/concept)

Usage:
    python load_pretrained_coordizer.py [--dry-run]
"""

import os
import sys
import json
import numpy as np
import argparse
from typing import Dict, List, Tuple, Optional

# Default paths
COORDIZER_JSON = "attached_assets/coordizer_1766884537919.json"
VECTORS_NPY = "attached_assets/vectors_1766884537920.npy"


def load_pretrained_data(json_path: str, npy_path: str) -> Tuple[Dict, np.ndarray]:
    """Load coordizer JSON and vectors numpy file."""
    print(f"Loading coordizer from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loading vectors from {npy_path}...")
    vectors = np.load(npy_path)
    
    print(f"  Vocab size: {data.get('vocab_size')}")
    print(f"  Basin dim: {data.get('basin_dim')}")
    print(f"  Merge rules: {len(data.get('merge_rules', []))}")
    print(f"  Vectors shape: {vectors.shape}")
    
    return data, vectors


def extract_tokens(vocab: Dict) -> List[Dict]:
    """Extract token info from vocab dictionary."""
    tokens = []
    
    for token_id, token_data in vocab.items():
        if isinstance(token_data, dict):
            tokens.append({
                'id': int(token_id),
                'coord_id': token_data.get('coord_id', int(token_id)),
                'name': token_data.get('name', ''),
                'scale': token_data.get('scale', 'unknown'),
                'vector': np.array(token_data.get('vector', [])),
            })
        else:
            # Simple string token
            tokens.append({
                'id': int(token_id),
                'name': str(token_data),
                'scale': 'unknown',
                'vector': None,
            })
    
    return sorted(tokens, key=lambda x: x['id'])


def analyze_vocabulary(tokens: List[Dict]) -> Dict:
    """Analyze vocabulary composition."""
    stats = {
        'total': len(tokens),
        'by_scale': {},
        'word_candidates': [],
        'with_vectors': 0,
    }
    
    for token in tokens:
        scale = token.get('scale', 'unknown')
        stats['by_scale'][scale] = stats['by_scale'].get(scale, 0) + 1
        
        if token.get('vector') is not None and len(token['vector']) > 0:
            stats['with_vectors'] += 1
        
        # Check if it's a word candidate (alphabetic, no special chars)
        name = token.get('name', '')
        clean_name = name.replace('+', '').replace(' ', '')
        if clean_name.isalpha() and len(clean_name) >= 3:
            stats['word_candidates'].append(token)
    
    return stats


def import_to_database(tokens: List[Dict], vectors: np.ndarray, dry_run: bool = False):
    """Import tokens into PostgreSQL tokenizer_vocabulary table."""
    import psycopg2
    
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        print("ERROR: DATABASE_URL not set")
        return False
    
    if dry_run:
        print("\n[DRY RUN] Would import tokens to database:")
        print(f"  Total tokens: {len(tokens)}")
        print(f"  Vector dimensions: {vectors.shape[1]}")
        return True
    
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    
    # Create table if not exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS pretrained_coordizer (
            token_id INTEGER PRIMARY KEY,
            token TEXT NOT NULL,
            scale TEXT,
            basin_embedding TEXT,
            phi_score FLOAT DEFAULT 0.75,
            source_type TEXT DEFAULT 'pretrained',
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    
    # Clear existing pretrained data
    cur.execute("DELETE FROM pretrained_coordizer")
    
    # Insert tokens
    batch_size = 1000
    imported = 0
    
    for i in range(0, len(tokens), batch_size):
        batch = tokens[i:i+batch_size]
        values = []
        
        for token in batch:
            token_id = token['id']
            name = token.get('name', '')
            scale = token.get('scale', 'unknown')
            
            # Get vector from either token data or numpy array
            if token.get('vector') is not None and len(token['vector']) > 0:
                vec = token['vector']
            elif token_id < len(vectors):
                vec = vectors[token_id]
            else:
                vec = np.zeros(64)
            
            # Format as PostgreSQL array string
            embedding_str = '[' + ','.join(f'{v:.8f}' for v in vec) + ']'
            
            values.append((token_id, name, scale, embedding_str))
        
        cur.executemany("""
            INSERT INTO pretrained_coordizer (token_id, token, scale, basin_embedding)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (token_id) DO UPDATE 
            SET token = EXCLUDED.token, scale = EXCLUDED.scale, basin_embedding = EXCLUDED.basin_embedding
        """, values)
        
        imported += len(batch)
        if imported % 5000 == 0:
            print(f"  Imported {imported}/{len(tokens)} tokens...")
    
    conn.commit()
    cur.close()
    conn.close()
    
    print(f"\n✓ Imported {imported} tokens to pretrained_coordizer table")
    return True


def export_merge_rules(data: Dict, output_path: str = "qig-backend/data/merge_rules.json"):
    """Export merge rules to JSON file for BPE encoding."""
    merge_rules = data.get('merge_rules', [])
    
    output = {
        'vocab_size': data.get('vocab_size', 32000),
        'merge_rules': merge_rules,
        'source': 'pretrained_coordizer',
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f)
    
    print(f"✓ Exported {len(merge_rules)} merge rules to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Load pretrained coordizer')
    parser.add_argument('--dry-run', action='store_true', help='Analyze without importing')
    parser.add_argument('--json', default=COORDIZER_JSON, help='Path to coordizer JSON')
    parser.add_argument('--npy', default=VECTORS_NPY, help='Path to vectors NPY')
    args = parser.parse_args()
    
    # Load data
    data, vectors = load_pretrained_data(args.json, args.npy)
    
    # Extract and analyze tokens
    vocab = data.get('vocab', {})
    tokens = extract_tokens(vocab)
    stats = analyze_vocabulary(tokens)
    
    print(f"\nVocabulary Analysis:")
    print(f"  Total tokens: {stats['total']}")
    print(f"  With vectors: {stats['with_vectors']}")
    print(f"  Word candidates: {len(stats['word_candidates'])}")
    print(f"  By scale:")
    for scale, count in sorted(stats['by_scale'].items()):
        print(f"    {scale}: {count}")
    
    # Export merge rules
    export_merge_rules(data)
    
    # Import to database
    if not args.dry_run:
        import_to_database(tokens, vectors, dry_run=False)
    else:
        import_to_database(tokens, vectors, dry_run=True)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
