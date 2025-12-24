#!/usr/bin/env python3
"""
Populate tokenizer_vocabulary with words from the 32K checkpoint.

This script:
1. Reads the checkpoint JSON from shared/coordizer/checkpoint_32000.json
2. Extracts tokens with their basin vectors
3. Identifies real English words (not just BPE fragments)
4. Inserts them into tokenizer_vocabulary table

Run: python populate_tokenizer_from_checkpoint.py
"""

import os
import sys
import json
import logging
import hashlib
import numpy as np
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

import psycopg2
from psycopg2.extras import execute_values


# Common English words to supplement BPE vocabulary
COMMON_WORDS = [
    # Articles and pronouns
    "the", "a", "an", "this", "that", "these", "those",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their", "mine", "yours", "ours", "theirs",
    
    # Verbs
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing", "done",
    "will", "would", "could", "should", "may", "might", "must", "shall",
    "can", "cannot",
    "go", "goes", "went", "going", "gone",
    "come", "comes", "came", "coming",
    "see", "sees", "saw", "seen", "seeing",
    "know", "knows", "knew", "known", "knowing",
    "think", "thinks", "thought", "thinking",
    "want", "wants", "wanted", "wanting",
    "use", "uses", "used", "using",
    "find", "finds", "found", "finding",
    "give", "gives", "gave", "given", "giving",
    "tell", "tells", "told", "telling",
    "work", "works", "worked", "working",
    "call", "calls", "called", "calling",
    "try", "tries", "tried", "trying",
    "ask", "asks", "asked", "asking",
    "need", "needs", "needed", "needing",
    "feel", "feels", "felt", "feeling",
    "become", "becomes", "became", "becoming",
    "leave", "leaves", "left", "leaving",
    "put", "puts", "putting",
    "mean", "means", "meant", "meaning",
    "keep", "keeps", "kept", "keeping",
    "let", "lets", "letting",
    "begin", "begins", "began", "begun", "beginning",
    "seem", "seems", "seemed", "seeming",
    "help", "helps", "helped", "helping",
    "show", "shows", "showed", "shown", "showing",
    "hear", "hears", "heard", "hearing",
    "play", "plays", "played", "playing",
    "run", "runs", "ran", "running",
    "move", "moves", "moved", "moving",
    "live", "lives", "lived", "living",
    "believe", "believes", "believed", "believing",
    "hold", "holds", "held", "holding",
    "bring", "brings", "brought", "bringing",
    "happen", "happens", "happened", "happening",
    "write", "writes", "wrote", "written", "writing",
    "provide", "provides", "provided", "providing",
    "sit", "sits", "sat", "sitting",
    "stand", "stands", "stood", "standing",
    "lose", "loses", "lost", "losing",
    "pay", "pays", "paid", "paying",
    "meet", "meets", "met", "meeting",
    "include", "includes", "included", "including",
    "continue", "continues", "continued", "continuing",
    "set", "sets", "setting",
    "learn", "learns", "learned", "learning",
    "change", "changes", "changed", "changing",
    "lead", "leads", "led", "leading",
    "understand", "understands", "understood", "understanding",
    "watch", "watches", "watched", "watching",
    "follow", "follows", "followed", "following",
    "stop", "stops", "stopped", "stopping",
    "create", "creates", "created", "creating",
    "speak", "speaks", "spoke", "spoken", "speaking",
    "read", "reads", "reading",
    "allow", "allows", "allowed", "allowing",
    "add", "adds", "added", "adding",
    "spend", "spends", "spent", "spending",
    "grow", "grows", "grew", "grown", "growing",
    "open", "opens", "opened", "opening",
    "walk", "walks", "walked", "walking",
    "win", "wins", "won", "winning",
    "offer", "offers", "offered", "offering",
    "remember", "remembers", "remembered", "remembering",
    "love", "loves", "loved", "loving",
    "consider", "considers", "considered", "considering",
    "appear", "appears", "appeared", "appearing",
    "buy", "buys", "bought", "buying",
    "wait", "waits", "waited", "waiting",
    "serve", "serves", "served", "serving",
    "die", "dies", "died", "dying",
    "send", "sends", "sent", "sending",
    "expect", "expects", "expected", "expecting",
    "build", "builds", "built", "building",
    "stay", "stays", "stayed", "staying",
    "fall", "falls", "fell", "fallen", "falling",
    "cut", "cuts", "cutting",
    "reach", "reaches", "reached", "reaching",
    "kill", "kills", "killed", "killing",
    "remain", "remains", "remained", "remaining",
    
    # Nouns
    "time", "year", "people", "way", "day", "man", "woman", "child", "children",
    "world", "life", "hand", "part", "place", "case", "week", "company", "system",
    "program", "question", "work", "government", "number", "night", "point", "home",
    "water", "room", "mother", "area", "money", "story", "fact", "month", "lot",
    "right", "study", "book", "eye", "job", "word", "business", "issue", "side",
    "kind", "head", "house", "service", "friend", "father", "power", "hour", "game",
    "line", "end", "member", "law", "car", "city", "community", "name", "president",
    "team", "minute", "idea", "kid", "body", "information", "back", "parent", "face",
    "others", "level", "office", "door", "health", "person", "art", "war", "history",
    "party", "result", "change", "morning", "reason", "research", "girl", "guy",
    "moment", "air", "teacher", "force", "education",
    
    # Adjectives
    "good", "new", "first", "last", "long", "great", "little", "own", "other",
    "old", "right", "big", "high", "different", "small", "large", "next", "early",
    "young", "important", "few", "public", "bad", "same", "able",
    "human", "local", "sure", "free", "better", "true", "whole", "special",
    "hard", "real", "best", "possible", "political", "social", "current",
    "full", "low", "late", "general", "specific", "certain", "clear",
    "available", "likely", "natural", "recent", "common", "economic",
    "open", "present", "strong", "similar", "past", "foreign", "fine",
    "simple", "easy", "obvious",
    
    # Adverbs
    "up", "so", "out", "just", "now", "how", "then", "more", "also", "here",
    "well", "only", "very", "even", "back", "there", "down", "still", "in",
    "as", "to", "when", "never", "really", "most", "why", "where",
    "much", "both", "before", "between", "after", "since", "without", "however",
    "often", "always", "together", "perhaps", "already", "yet", "especially",
    "away", "today", "almost", "enough", "ever", "rather", "ago",
    
    # Prepositions and conjunctions
    "of", "to", "in", "for", "on", "with", "at", "by", "from", "or", "and",
    "but", "not", "what", "all", "were", "when", "we", "there", "can", "an",
    "your", "which", "their", "if", "will", "each", "about", "how", "up",
    "out", "them", "then", "she", "many", "some", "so", "these", "would",
    "other", "into", "has", "more", "her", "two", "like", "him", "see", "time",
    "could", "no", "make", "than", "first", "been", "its", "who", "now", "people",
    "my", "made", "over", "did", "down", "only", "way", "find", "use",
    
    # Question words
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    
    # Numbers
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "hundred", "thousand", "million", "billion",
    
    # Tech/domain specific
    "data", "code", "file", "user", "system", "network", "server", "client",
    "database", "query", "response", "request", "error", "function", "method",
    "class", "object", "string", "number", "array", "list", "map", "set",
    "key", "value", "type", "interface", "module", "package", "import", "export",
    "return", "async", "await", "promise", "callback", "event", "handler",
    "config", "option", "setting", "parameter", "argument", "variable", "constant",
    "true", "false", "null", "undefined", "boolean", "integer", "float", "double",
    
    # QIG/Consciousness specific
    "consciousness", "quantum", "geometry", "basin", "manifold", "kernel",
    "integration", "phi", "kappa", "fisher", "geodesic", "attractor",
    "resonance", "coherence", "synthesis", "emergence", "pattern", "structure",
    "dimension", "coordinate", "vector", "matrix", "tensor", "gradient",
    "distance", "similarity", "embedding", "encoding", "decoding", "token",
    "vocabulary", "semantic", "meaning", "context", "attention", "memory",
    "reasoning", "inference", "prediction", "generation", "completion",
]


def generate_basin_for_word(word: str, dimension: int = 64) -> np.ndarray:
    """Generate deterministic basin coordinates for a word using semantic hashing."""
    # Use SHA256 for deterministic but well-distributed hash
    word_hash = hashlib.sha256(word.encode('utf-8')).hexdigest()
    
    # Convert hash to seed
    seed = int(word_hash[:8], 16)
    rng = np.random.RandomState(seed)
    
    # Generate coordinates using Dirichlet for probability simplex
    coords = rng.dirichlet(np.ones(dimension))
    
    # Project to unit sphere for Fisher-Rao geometry
    norm = np.linalg.norm(coords)
    if norm > 1e-10:
        coords = coords / norm
    
    return coords


def calculate_phi_score(word: str, frequency: int = 1) -> float:
    """Calculate phi score for a word based on properties."""
    base_phi = 0.5
    
    # Longer words often have more semantic content
    length_bonus = min(len(word) / 10.0, 0.2)
    
    # Common words get slight boost
    if word.lower() in [w.lower() for w in COMMON_WORDS]:
        common_bonus = 0.15
    else:
        common_bonus = 0.0
    
    # Frequency bonus (log scale)
    freq_bonus = min(np.log1p(frequency) / 20.0, 0.15)
    
    return min(base_phi + length_bonus + common_bonus + freq_bonus, 0.95)


def load_checkpoint(checkpoint_path: str) -> dict:
    """Load the 32K checkpoint JSON."""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    with open(checkpoint_path, 'r') as f:
        return json.load(f)


def extract_real_words_from_checkpoint(checkpoint: dict) -> list:
    """Extract tokens that are real words (not BPE fragments) from checkpoint."""
    vocab = checkpoint.get('vocab', {})
    real_words = []
    
    for idx, token_data in vocab.items():
        name = token_data.get('name', '')
        vector = token_data.get('vector', [])
        frequency = token_data.get('frequency', 1)
        
        # Skip byte tokens
        if name.startswith('<byte_'):
            continue
        
        # Skip empty or single-char tokens (except valid ones like 'I', 'a')
        if len(name) < 2 and name.lower() not in ['i', 'a']:
            continue
        
        # Check if it's a real word (alphabetic, reasonable length)
        # BPE fragments are usually short non-words like 'th', 'ing', etc.
        is_real_word = (
            name.isalpha() and 
            len(name) >= 3 and
            # Heuristic: real words often have vowels
            any(c in name.lower() for c in 'aeiou')
        )
        
        if is_real_word or name.lower() in [w.lower() for w in COMMON_WORDS]:
            real_words.append({
                'token': name,
                'vector': vector,
                'frequency': frequency,
                'source': 'checkpoint'
            })
    
    logger.info(f"Found {len(real_words)} potential real words in checkpoint")
    return real_words


def ensure_table_exists(conn):
    """Ensure tokenizer_vocabulary table exists."""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS tokenizer_vocabulary (
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
        conn.commit()
        logger.info("Table tokenizer_vocabulary ensured")


def get_max_token_id(conn) -> int:
    """Get the maximum token_id currently in the table."""
    with conn.cursor() as cur:
        cur.execute("SELECT COALESCE(MAX(token_id), 0) FROM tokenizer_vocabulary")
        return cur.fetchone()[0]


def insert_words(conn, words: list, source_type: str = 'learned', start_token_id: int = 10000):
    """Insert words into tokenizer_vocabulary."""
    if not words:
        return 0
    
    inserted = 0
    token_id = start_token_id
    
    with conn.cursor() as cur:
        for word_data in words:
            token = word_data['token']
            vector = word_data.get('vector', [])
            frequency = word_data.get('frequency', 1)
            
            # Generate basin if not provided or wrong dimension
            if not vector or len(vector) != 64:
                basin = generate_basin_for_word(token)
            else:
                basin = np.array(vector)
                # Normalize to unit sphere
                norm = np.linalg.norm(basin)
                if norm > 1e-10:
                    basin = basin / norm
            
            # Calculate phi score
            phi = calculate_phi_score(token, frequency)
            
            # Format basin as PostgreSQL vector
            basin_str = '[' + ','.join(map(str, basin)) + ']'
            
            try:
                cur.execute("""
                    INSERT INTO tokenizer_vocabulary 
                        (token, token_id, weight, frequency, phi_score, basin_embedding, source_type)
                    VALUES (%s, %s, %s, %s, %s, %s::vector, %s)
                    ON CONFLICT (token) DO UPDATE SET
                        frequency = GREATEST(tokenizer_vocabulary.frequency, EXCLUDED.frequency),
                        phi_score = GREATEST(tokenizer_vocabulary.phi_score, EXCLUDED.phi_score),
                        basin_embedding = EXCLUDED.basin_embedding,
                        updated_at = CURRENT_TIMESTAMP
                """, (token, token_id, phi, frequency, phi, basin_str, source_type))
                inserted += 1
                token_id += 1
            except Exception as e:
                logger.warning(f"Failed to insert {token}: {e}")
                conn.rollback()
                continue
        
        conn.commit()
    
    return inserted


def main():
    """Main migration function."""
    # Get database URL
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        logger.error("DATABASE_URL not set!")
        sys.exit(1)
    
    # Find checkpoint file
    checkpoint_path = Path(__file__).parent.parent / 'shared' / 'coordizer' / 'checkpoint_32000.json'
    if not checkpoint_path.exists():
        logger.warning(f"Checkpoint not found at {checkpoint_path}")
        checkpoint_path = None
    
    # Connect to database
    logger.info("Connecting to database...")
    conn = psycopg2.connect(database_url)
    
    try:
        # Ensure table exists
        ensure_table_exists(conn)
        
        # Get current max token_id
        max_id = get_max_token_id(conn)
        logger.info(f"Current max token_id: {max_id}")
        
        # Start new IDs after existing ones
        next_id = max_id + 1
        
        # First, add common English words
        logger.info(f"Adding {len(COMMON_WORDS)} common English words...")
        common_word_data = [{'token': w, 'frequency': 100} for w in COMMON_WORDS]
        inserted_common = insert_words(conn, common_word_data, source_type='learned', start_token_id=next_id)
        logger.info(f"Inserted {inserted_common} common words")
        
        # Update next_id
        next_id = get_max_token_id(conn) + 1
        
        # Then, load from checkpoint if available
        if checkpoint_path:
            checkpoint = load_checkpoint(checkpoint_path)
            real_words = extract_real_words_from_checkpoint(checkpoint)
            
            if real_words:
                logger.info(f"Adding {len(real_words)} words from checkpoint...")
                inserted_checkpoint = insert_words(conn, real_words, source_type='checkpoint', start_token_id=next_id)
                logger.info(f"Inserted {inserted_checkpoint} checkpoint words")
        
        # Verify
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM tokenizer_vocabulary")
            total = cur.fetchone()[0]
            
            cur.execute("SELECT source_type, COUNT(*) FROM tokenizer_vocabulary GROUP BY source_type")
            by_source = cur.fetchall()
            
            cur.execute("""
                SELECT token, phi_score, source_type 
                FROM tokenizer_vocabulary 
                WHERE LENGTH(token) >= 3 
                ORDER BY phi_score DESC 
                LIMIT 15
            """)
            top_words = cur.fetchall()
        
        logger.info(f"\n=== Migration Complete ===")
        logger.info(f"Total tokens in tokenizer_vocabulary: {total}")
        logger.info(f"\nBy source type:")
        for source, count in by_source:
            logger.info(f"  {source}: {count}")
        
        logger.info(f"\nTop words by phi score:")
        for token, phi, source in top_words:
            logger.info(f"  {token}: phi={phi:.3f} ({source})")
        
    finally:
        conn.close()
    
    logger.info("\nDone! Restart the Python backend to use the new vocabulary.")


if __name__ == '__main__':
    main()
