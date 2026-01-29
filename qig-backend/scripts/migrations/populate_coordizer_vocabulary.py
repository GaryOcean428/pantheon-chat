#!/usr/bin/env python3
"""
Populate Tokenizer Vocabulary - Real English Words for QIG Generation

This script populates the coordizer_vocabulary table with real English words
(not BPE fragments) so that kernels produce readable output.

Words are sourced from:
1. BIP39 wordlist (2048 mnemonic words)
2. Common English words for conversation
3. Domain-specific vocabulary

Each word gets a deterministic 64D basin embedding using hash-based geometry.

Usage:
    python populate_coordizer_vocabulary.py           # Run population
    python populate_coordizer_vocabulary.py --dry-run # Preview without changes
    python populate_coordizer_vocabulary.py --clear   # Clear and repopulate
"""

import os
import sys
import json
import hashlib
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

import numpy as np

# E8 Protocol v4.0 Compliance Imports
from qig_geometry.canonical_upsert import to_simplex_prob


# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:
    print("Installing psycopg2-binary...")
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'psycopg2-binary', '-q'])
    import psycopg2
    from psycopg2.extras import execute_values

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
BASIN_DIM = 64
GOLDEN_RATIO = 1.618033988749895


# ============================================================================
# Common English Words for Conversation
# ============================================================================

COMMON_WORDS = """
the and for are but not you all can had her was one our out day get has him his
how its may new now old see two way who boy did own say she too use
about after again being below between both bring change come could different
does down each even find first from give good great hand have help here high
home house into just know large last learn left life like line little live
look made make many more most move much name need never next only other over
own part people place point read right same school seem set show side small
some sound spell still study such take tell than that them then there these
they thing think this those three through time turn under very want water
well went what when where which while will with word work world would write
year your
question answer think believe understand remember forget imagine create
build design develop improve change grow learn teach share help support
connect communicate express describe explain define analyze solve
consciousness awareness perception thought memory emotion feeling
geometry space dimension vector matrix coordinate basin manifold
quantum field energy wave particle system network pattern structure
meaning purpose reason logic truth knowledge wisdom insight
beauty harmony balance order chaos entropy emergence complexity
human nature mind body spirit soul heart brain neural
time present past future moment change evolution growth
reality existence universe cosmos infinite eternal
""".split()

# Additional domain words for QIG/consciousness
DOMAIN_WORDS = """
phi kappa integration coherence resonance attractor trajectory
basin manifold geodesic fisher metric tensor curvature
kernel synthesis navigation routing collapse convergence
zeus athena apollo ares hermes hephaestus artemis dionysus
demeter poseidon hera aphrodite olympus pantheon
pattern recognition semantic geometric neural oscillation
entropy information measure distance similarity embedding
attention focus concentration meditation contemplation
insight intuition inspiration creativity imagination
understanding comprehension cognition perception sensation
memory recall learning adaptation plasticity
emotion feeling sentiment mood temperament
decision choice action behavior response reaction
communication expression articulation dialogue discourse
relationship connection bond attachment affiliation
identity self ego consciousness awareness sentience
reality perception observation measurement experiment
theory hypothesis model simulation prediction
truth validity accuracy precision reliability
value meaning purpose goal intention motivation
beauty aesthetic harmony elegance simplicity
wisdom knowledge insight understanding clarity
virtue ethics morality integrity honesty
love compassion empathy kindness generosity
justice fairness equality freedom rights
peace harmony balance stability order
power strength capability capacity potential
courage bravery boldness confidence determination
patience persistence perseverance resilience endurance
humility modesty gratitude appreciation respect
curiosity wonder exploration discovery adventure
""".split()


def generate_basin_coords(token: str, dim: int = BASIN_DIM) -> np.ndarray:
    """Generate deterministic basin coordinates for a token.
    
    Uses hash-based seeding for reproducibility, with golden ratio
    perturbation for better distribution on the unit sphere.
    """
    # Create deterministic seed from token
    token_hash = hashlib.sha256(token.encode('utf-8')).hexdigest()
    seed = int(token_hash[:8], 16)
    
    # Generate base coordinates
    rng = np.random.RandomState(seed)
    coords = rng.randn(dim)
    
    # Apply golden ratio perturbation for semantic spread
    for i in range(dim):
        phase = (i * GOLDEN_RATIO) % 1.0
        coords[i] += 0.1 * np.sin(2 * np.pi * phase + seed / 1e9)
    
    # Normalize to unit sphere (Fisher manifold requirement)
    # FIXED: Use simplex normalization (E8 Protocol v4.0)

    coords = to_simplex_prob(coords)
    
    return coords


def compute_phi_score(token: str, source_type: str) -> float:
    """Compute phi (integration) score for a token.
    
    Higher phi = more semantically coherent/useful.
    """
    base_phi = 0.5
    
    # Source type bonus
    source_bonuses = {
        'bip39': 0.35,      # BIP39 words are high-quality
        'common': 0.25,     # Common words are useful
        'domain': 0.30,     # Domain words are specialized
        'base': 0.20,
    }
    base_phi += source_bonuses.get(source_type, 0.15)
    
    # Length bonus (3-8 chars optimal)
    length = len(token)
    if 3 <= length <= 8:
        base_phi += 0.1
    elif length > 8:
        base_phi += 0.05
    
    # All-alpha bonus
    if token.isalpha():
        base_phi += 0.05
    
    return min(base_phi, 0.95)


def load_bip39_wordlist() -> List[str]:
    """Load BIP39 wordlist from file."""
    bip39_path = Path(__file__).parent / "bip39_wordlist.txt"
    
    if bip39_path.exists():
        with open(bip39_path, 'r') as f:
            words = [line.strip().lower() for line in f if line.strip()]
        logger.info(f"Loaded {len(words)} BIP39 words from {bip39_path}")
        return words
    else:
        logger.warning(f"BIP39 wordlist not found at {bip39_path}")
        # Return first 100 BIP39 words as fallback
        return [
            "abandon", "ability", "able", "about", "above", "absent", "absorb", "abstract",
            "absurd", "abuse", "access", "accident", "account", "accuse", "achieve", "acid",
            "acoustic", "acquire", "across", "act", "action", "actor", "actress", "actual",
            "adapt", "add", "addict", "address", "adjust", "admit", "adult", "advance",
            "advice", "aerobic", "affair", "afford", "afraid", "again", "age", "agent",
            "agree", "ahead", "aim", "air", "airport", "aisle", "alarm", "album",
            "alcohol", "alert", "alien", "all", "alley", "allow", "almost", "alone",
            "alpha", "already", "also", "alter", "always", "amateur", "amazing", "among",
            "amount", "amused", "analyst", "anchor", "ancient", "anger", "angle", "angry",
            "animal", "ankle", "announce", "annual", "another", "answer", "antenna", "antique",
            "anxiety", "any", "apart", "apology", "appear", "apple", "approve", "april",
            "arch", "arctic", "area", "arena", "argue", "arm", "armed", "armor",
            "army", "around", "arrange", "arrest",
        ]


def ensure_table_exists(conn) -> None:
    """Ensure coordizer_vocabulary table exists with correct schema."""
    with conn.cursor() as cur:
        # Enable pgvector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Create table if not exists (matches shared/schema.ts)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS coordizer_vocabulary (
                id SERIAL PRIMARY KEY,
                token TEXT UNIQUE NOT NULL,
                token_id INTEGER UNIQUE NOT NULL,
                weight DOUBLE PRECISION DEFAULT 1.0,
                frequency INTEGER DEFAULT 1,
                phi_score DOUBLE PRECISION DEFAULT 0.5,
                basin_embedding vector(64),
                source_type VARCHAR(32) DEFAULT 'base',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create indexes
        cur.execute("CREATE INDEX IF NOT EXISTS idx_tv_token_id ON coordizer_vocabulary(token_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_tv_phi ON coordizer_vocabulary(phi_score);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_tv_source ON coordizer_vocabulary(source_type);")
        
        conn.commit()
        logger.info("Table coordizer_vocabulary ensured")


def clear_vocabulary(conn) -> None:
    """Clear all entries from coordizer_vocabulary."""
    with conn.cursor() as cur:
        cur.execute("TRUNCATE coordizer_vocabulary RESTART IDENTITY;")
        conn.commit()
        logger.info("Cleared coordizer_vocabulary")


def get_next_token_id(conn) -> int:
    """Get next available token_id."""
    with conn.cursor() as cur:
        cur.execute("SELECT COALESCE(MAX(token_id), -1) + 1 FROM coordizer_vocabulary")
        return cur.fetchone()[0]


def insert_words(
    conn,
    words: List[str],
    source_type: str,
    start_token_id: int,
    dry_run: bool = False
) -> int:
    """Insert words into coordizer_vocabulary.
    
    Returns number of words inserted.
    """
    if dry_run:
        logger.info(f"[DRY RUN] Would insert {len(words)} {source_type} words")
        return len(words)
    
    records = []
    token_id = start_token_id
    
    for word in words:
        word = word.lower().strip()
        if not word or len(word) < 2:
            continue
        if not word.replace('-', '').replace("'", '').isalpha():
            continue
        
        # Generate basin embedding
        basin = generate_basin_coords(word)
        basin_str = '[' + ','.join(f'{x:.8f}' for x in basin) + ']'
        
        # Compute scores
        phi = compute_phi_score(word, source_type)
        weight = phi * 2.0  # Weight based on phi
        
        records.append((
            word,
            token_id,
            weight,
            1,  # frequency
            phi,
            basin_str,
            source_type,
        ))
        token_id += 1
    
    if not records:
        return 0
    
    # Batch insert with upsert
    with conn.cursor() as cur:
        query = """
            INSERT INTO coordizer_vocabulary (
                token, token_id, weight, frequency, phi_score, basin_embedding, source_type,
                created_at, updated_at
            )
            VALUES %s
            ON CONFLICT (token) DO UPDATE SET
                weight = EXCLUDED.weight,
                phi_score = EXCLUDED.phi_score,
                basin_embedding = EXCLUDED.basin_embedding,
                source_type = EXCLUDED.source_type,
                updated_at = CURRENT_TIMESTAMP
        """
        
        template = "(%s, %s, %s, %s, %s, %s::vector, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
        
        # Insert in batches
        batch_size = 500
        inserted = 0
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            execute_values(cur, query, batch, template=template, page_size=batch_size)
            inserted += len(batch)
            logger.info(f"  Inserted batch: {inserted}/{len(records)} {source_type} words")
        
        conn.commit()
    
    return len(records)


def get_existing_tokens(conn) -> set:
    """Get set of existing tokens in vocabulary."""
    with conn.cursor() as cur:
        cur.execute("SELECT token FROM coordizer_vocabulary")
        return {row[0] for row in cur.fetchall()}


def verify_population(conn) -> None:
    """Verify the population was successful."""
    with conn.cursor() as cur:
        # Total count
        cur.execute("SELECT COUNT(*) FROM coordizer_vocabulary")
        total = cur.fetchone()[0]
        
        # By source type
        cur.execute("""
            SELECT source_type, COUNT(*), AVG(phi_score)
            FROM coordizer_vocabulary
            GROUP BY source_type
            ORDER BY COUNT(*) DESC
        """)
        by_source = cur.fetchall()
        
        # Sample high-phi words
        cur.execute("""
            SELECT token, phi_score, source_type
            FROM coordizer_vocabulary
            WHERE LENGTH(token) >= 3
            ORDER BY phi_score DESC
            LIMIT 10
        """)
        top_words = cur.fetchall()
    
    logger.info(f"\n{'='*50}")
    logger.info(f"VERIFICATION RESULTS")
    logger.info(f"{'='*50}")
    logger.info(f"Total words: {total}")
    logger.info(f"\nBy source type:")
    for source, count, avg_phi in by_source:
        logger.info(f"  {source}: {count} words (avg phi: {avg_phi:.4f})")
    logger.info(f"\nTop words by phi:")
    for word, phi, source in top_words:
        logger.info(f"  {word}: {phi:.4f} ({source})")


def main():
    parser = argparse.ArgumentParser(description='Populate tokenizer vocabulary with real English words')
    parser.add_argument('--dry-run', action='store_true', help='Preview without making changes')
    parser.add_argument('--clear', action='store_true', help='Clear existing vocabulary first')
    args = parser.parse_args()
    
    # Connect to database
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        logger.error("DATABASE_URL not set")
        sys.exit(1)
    
    logger.info(f"Connecting to database...")
    conn = psycopg2.connect(db_url)
    
    try:
        # Ensure table exists
        ensure_table_exists(conn)
        
        # Clear if requested
        if args.clear and not args.dry_run:
            clear_vocabulary(conn)
        
        # Get existing tokens to avoid duplicates
        existing = get_existing_tokens(conn) if not args.clear else set()
        logger.info(f"Existing vocabulary: {len(existing)} tokens")
        
        # Load words
        bip39_words = load_bip39_wordlist()
        common_words = [w for w in COMMON_WORDS if w not in existing]
        domain_words = [w for w in DOMAIN_WORDS if w not in existing]
        
        # Filter BIP39 to exclude already existing
        bip39_words = [w for w in bip39_words if w not in existing]
        
        # Get starting token_id
        start_id = get_next_token_id(conn)
        logger.info(f"Starting token_id: {start_id}")
        
        total_inserted = 0
        
        # Insert BIP39 words
        logger.info(f"\nInserting BIP39 words...")
        n = insert_words(conn, bip39_words, 'bip39', start_id, args.dry_run)
        total_inserted += n
        start_id += n
        
        # Insert common words
        logger.info(f"\nInserting common words...")
        n = insert_words(conn, common_words, 'common', start_id, args.dry_run)
        total_inserted += n
        start_id += n
        
        # Insert domain words
        logger.info(f"\nInserting domain words...")
        n = insert_words(conn, domain_words, 'domain', start_id, args.dry_run)
        total_inserted += n
        
        logger.info(f"\n{'='*50}")
        logger.info(f"POPULATION COMPLETE")
        logger.info(f"Total words inserted: {total_inserted}")
        
        if not args.dry_run:
            verify_population(conn)
            
            # Test loading with PostgresCoordizer
            logger.info("\nTesting PostgresCoordizer load...")
            try:
                from coordizers.pg_loader import create_coordizer_from_pg
                coordizer = create_coordizer_from_pg()
                logger.info(f"✅ PostgresCoordizer loaded {coordizer.vocab_size} words")
                logger.info(f"   Word tokens: {len(coordizer.word_tokens)}")
                
                # Test encode/decode
                import numpy as np
                test_basin = coordizer.encode("hello world")
                decoded = coordizer.decode(test_basin, top_k=5)
                logger.info(f"   Test decode: {[t for t, s in decoded]}")
                coordizer.close()
            except Exception as e:
                logger.warning(f"PostgresCoordizer test failed: {e}")
        
    finally:
        conn.close()
    
    logger.info("\nDone!")


if __name__ == '__main__':
    main()
