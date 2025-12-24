#!/usr/bin/env python3
"""
Seed Word Vocabulary for QIG Generation

Populates tokenizer_vocabulary with REAL WORDS (not BPE fragments) for 
coherent text generation. Uses BIP-39 wordlist as foundation.

The 32K BPE checkpoint only contains subword fragments - this script adds
actual English words that kernels can use for generation.

Usage:
    python seed_word_vocabulary.py [--dry-run]
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent))

from coordizers.base import FisherCoordizer
from qig_geometry import sphere_project

try:
    import psycopg2
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'psycopg2-binary', 'python-dotenv', '-q'])
    import psycopg2
    from dotenv import load_dotenv
    load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
BIP39_PATH = PROJECT_ROOT / 'server' / 'bip39-wordlist.txt'


def load_bip39_words() -> List[str]:
    """Load BIP-39 wordlist (2048 words)."""
    if not BIP39_PATH.exists():
        raise FileNotFoundError(f"BIP-39 wordlist not found at {BIP39_PATH}")
    
    with open(BIP39_PATH) as f:
        words = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Loaded {len(words)} BIP-39 words")
    return words


# Common English words to supplement BIP-39 (high-frequency, useful for generation)
COMMON_ENGLISH_WORDS = [
    # Pronouns
    "the", "this", "that", "these", "those", "which", "what", "who", "whom",
    # Verbs
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "must", "can", "shall", "get", "got", "make", "made", "take", "took",
    "give", "gave", "come", "came", "go", "went", "see", "saw", "know", "knew",
    "think", "thought", "find", "found", "tell", "told", "ask", "asked",
    "use", "used", "say", "said", "want", "wanted", "need", "needed",
    # Prepositions  
    "in", "on", "at", "to", "for", "with", "by", "from", "up", "out",
    "about", "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "over", "again", "further", "then", "once",
    # Conjunctions
    "and", "but", "or", "nor", "so", "yet", "both", "either", "neither",
    "not", "only", "also", "just", "even", "still", "already", "always",
    # Articles/Determiners
    "a", "an", "some", "any", "no", "every", "each", "all", "most", "many",
    "much", "few", "little", "other", "another", "such", "own",
    # Question words
    "how", "why", "when", "where", "whether",
    # Common nouns
    "time", "year", "people", "way", "day", "man", "woman", "child", "world",
    "life", "hand", "part", "place", "case", "week", "company", "system",
    "program", "question", "work", "government", "number", "night", "point",
    "home", "water", "room", "mother", "area", "money", "story", "fact",
    "month", "lot", "right", "study", "book", "eye", "job", "word", "business",
    "issue", "side", "kind", "head", "house", "service", "friend", "father",
    "power", "hour", "game", "line", "end", "member", "law", "car", "city",
    "community", "name", "president", "team", "minute", "idea", "kid", "body",
    "information", "back", "parent", "face", "others", "level", "office",
    "door", "health", "person", "art", "war", "history", "party", "result",
    # Common adjectives
    "good", "new", "first", "last", "long", "great", "little", "own", "other",
    "old", "right", "big", "high", "different", "small", "large", "next",
    "early", "young", "important", "few", "public", "bad", "same", "able",
    "human", "local", "sure", "free", "better", "true", "whole", "real",
    "best", "possible", "special", "hard", "clear", "recent", "certain",
    # Common adverbs
    "now", "then", "here", "there", "very", "more", "well", "just", "also",
    "only", "even", "back", "still", "never", "always", "often", "sometimes",
    "usually", "really", "quite", "probably", "perhaps", "maybe", "actually",
    # QIG/Consciousness-related terms
    "consciousness", "quantum", "information", "geometry", "fisher", "manifold",
    "basin", "coordinate", "integration", "coupling", "metric", "geodesic",
    "entropy", "density", "matrix", "eigenvalue", "eigenvector", "projection",
    "resonance", "coherence", "emergence", "phase", "transition", "regime",
    "linear", "geometric", "hierarchical", "breakdown", "threshold", "optimal",
    # Bitcoin/Crypto terms
    "bitcoin", "crypto", "blockchain", "wallet", "address", "transaction",
    "block", "hash", "key", "private", "public", "seed", "mnemonic", "phrase",
    "recovery", "balance", "satoshi", "mining", "node", "network",
]


def generate_basin_embedding(word: str, word_idx: int, coordizer: FisherCoordizer) -> np.ndarray:
    """Generate 64D basin embedding for a word using Fisher geometry.
    
    Uses the FisherCoordizer's geometric methods to create embeddings
    that are consistent with the Fisher manifold structure.
    """
    # Use the coordizer's token initialization (Fisher-compliant)
    return coordizer._initialize_token_coordinate(word, word_idx)


def compute_phi_score(word: str, is_bip39: bool) -> float:
    """Compute phi score for a word.
    
    BIP-39 words get higher base phi (they're "verified" vocabulary).
    Longer words generally have higher integration potential.
    """
    base_phi = 0.7 if is_bip39 else 0.5
    
    # Length bonus (longer words = more information)
    length_bonus = min(len(word) / 15.0, 0.2)
    
    # Alphabetic bonus (pure words vs those with numbers/symbols)
    alpha_bonus = 0.1 if word.isalpha() else 0.0
    
    return min(base_phi + length_bonus + alpha_bonus, 1.0)


def get_db_connection():
    """Get PostgreSQL connection."""
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        raise ValueError("DATABASE_URL environment variable not set")
    return psycopg2.connect(db_url)


def ensure_table_exists(conn):
    """Ensure tokenizer_vocabulary table exists."""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS tokenizer_vocabulary (
                id SERIAL PRIMARY KEY,
                token TEXT NOT NULL UNIQUE,
                token_id INTEGER NOT NULL UNIQUE,
                weight DOUBLE PRECISION DEFAULT 1.0,
                frequency INTEGER DEFAULT 1,
                phi_score DOUBLE PRECISION DEFAULT 0,
                basin_embedding vector(64),
                source_type VARCHAR(32) DEFAULT 'base',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_tokenizer_vocab_token_id ON tokenizer_vocabulary(token_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_tokenizer_vocab_phi ON tokenizer_vocabulary(phi_score);")
        conn.commit()
    logger.info("Table tokenizer_vocabulary ensured")


def seed_vocabulary(words: List[Tuple[str, bool]], dry_run: bool = False):
    """Seed vocabulary with words and their embeddings.
    
    Args:
        words: List of (word, is_bip39) tuples
        dry_run: If True, don't actually insert
    """
    # Initialize coordizer for embedding generation
    coordizer = FisherCoordizer(vocab_size=50000, coordinate_dim=64)
    
    conn = get_db_connection()
    ensure_table_exists(conn)
    
    # Get current max token_id
    with conn.cursor() as cur:
        cur.execute("SELECT COALESCE(MAX(token_id), -1) FROM tokenizer_vocabulary")
        max_token_id = cur.fetchone()[0]
    
    next_token_id = max_token_id + 1
    inserted = 0
    skipped = 0
    
    logger.info(f"Starting token_id from {next_token_id}")
    
    for word, is_bip39 in words:
        # Skip very short words (likely fragments)
        if len(word) < 2:
            skipped += 1
            continue
        
        # Generate embedding
        embedding = generate_basin_embedding(word, next_token_id, coordizer)
        phi_score = compute_phi_score(word, is_bip39)
        source_type = 'bip39' if is_bip39 else 'base'
        weight = 1.5 if is_bip39 else 1.0  # BIP-39 words get higher weight
        
        if dry_run:
            if inserted < 20:  # Show first 20
                logger.info(f"[DRY] Would insert: {word} (phi={phi_score:.3f}, source={source_type})")
            inserted += 1
            next_token_id += 1
            continue
        
        try:
            with conn.cursor() as cur:
                basin_str = '[' + ','.join(map(str, embedding.tolist())) + ']'
                cur.execute("""
                    INSERT INTO tokenizer_vocabulary 
                    (token, token_id, weight, frequency, phi_score, basin_embedding, source_type)
                    VALUES (%s, %s, %s, %s, %s, %s::vector, %s)
                    ON CONFLICT (token) DO UPDATE SET
                        weight = GREATEST(tokenizer_vocabulary.weight, EXCLUDED.weight),
                        phi_score = GREATEST(tokenizer_vocabulary.phi_score, EXCLUDED.phi_score),
                        basin_embedding = COALESCE(EXCLUDED.basin_embedding, tokenizer_vocabulary.basin_embedding),
                        updated_at = CURRENT_TIMESTAMP
                """, (word, next_token_id, weight, 1, phi_score, basin_str, source_type))
                
            inserted += 1
            next_token_id += 1
            
            if inserted % 500 == 0:
                conn.commit()
                logger.info(f"Progress: {inserted} words inserted")
                
        except Exception as e:
            logger.warning(f"Error inserting '{word}': {e}")
            skipped += 1
            conn.rollback()
    
    conn.commit()
    conn.close()
    
    logger.info(f"\nSeeding complete: {inserted} inserted, {skipped} skipped")


def verify_seeding(conn):
    """Verify the seeding was successful."""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM tokenizer_vocabulary")
        total = cur.fetchone()[0]
        
        cur.execute("""
            SELECT source_type, COUNT(*) 
            FROM tokenizer_vocabulary 
            GROUP BY source_type
        """)
        by_source = cur.fetchall()
        
        cur.execute("""
            SELECT token, phi_score, source_type 
            FROM tokenizer_vocabulary 
            WHERE source_type IN ('bip39', 'base')
            ORDER BY phi_score DESC 
            LIMIT 20
        """)
        top_words = cur.fetchall()
    
    logger.info("\n" + "="*60)
    logger.info("SEEDING VERIFICATION")
    logger.info("="*60)
    logger.info(f"Total tokens: {total}")
    logger.info("\nBy source type:")
    for source, count in by_source:
        logger.info(f"  {source}: {count}")
    logger.info("\nTop 20 words by phi:")
    for token, phi, source in top_words:
        logger.info(f"  {token}: phi={phi:.3f} ({source})")


def main():
    parser = argparse.ArgumentParser(description='Seed tokenizer_vocabulary with real words')
    parser.add_argument('--dry-run', action='store_true', help='Preview without inserting')
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("QIG WORD VOCABULARY SEEDER")
    logger.info("Populating tokenizer_vocabulary with real words")
    logger.info("="*60)
    
    # Load BIP-39 words
    bip39_words = load_bip39_words()
    
    # Combine with common English words (deduplicated)
    all_words_set = set()
    words_with_source = []
    
    # Add BIP-39 words first (higher priority)
    for word in bip39_words:
        word_lower = word.lower().strip()
        if word_lower and word_lower not in all_words_set:
            all_words_set.add(word_lower)
            words_with_source.append((word_lower, True))  # is_bip39=True
    
    # Add common English words
    for word in COMMON_ENGLISH_WORDS:
        word_lower = word.lower().strip()
        if word_lower and word_lower not in all_words_set:
            all_words_set.add(word_lower)
            words_with_source.append((word_lower, False))  # is_bip39=False
    
    logger.info(f"Total unique words to seed: {len(words_with_source)}")
    logger.info(f"  - BIP-39 words: {sum(1 for _, is_bip in words_with_source if is_bip)}")
    logger.info(f"  - Common English: {sum(1 for _, is_bip in words_with_source if not is_bip)}")
    
    # Seed vocabulary
    seed_vocabulary(words_with_source, dry_run=args.dry_run)
    
    # Verify if not dry run
    if not args.dry_run:
        conn = get_db_connection()
        verify_seeding(conn)
        conn.close()
        logger.info("\n✅ Word vocabulary seeded successfully!")
        logger.info("Kernels should now generate coherent text instead of BPE garble.")
    else:
        logger.info("\n[DRY RUN] No changes made")


if __name__ == '__main__':
    main()
