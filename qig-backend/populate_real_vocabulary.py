#!/usr/bin/env python3
"""Populate tokenizer_vocabulary with real English words and geometric basin embeddings.

This script adds:
1. BIP39 wordlist (2048 mnemonic words)
2. Common English words (top 5000)
3. Domain-specific vocabulary

Each word gets a deterministic 64D basin embedding computed using geometric methods.
"""

import os
import sys
import logging
import hashlib
from pathlib import Path
from typing import List, Tuple

import numpy as np
import psycopg2
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

# Golden ratio for geometric embedding
PHI = (1 + np.sqrt(5)) / 2

# BIP39 wordlist path
BIP39_PATH = Path(__file__).parent.parent / 'server' / 'bip39-wordlist.txt'

# Common English words (top 3000 most common)
COMMON_ENGLISH_WORDS = [
    # Basic words
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
    "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
    "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
    "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
    # Action words
    "find", "here", "thing", "place", "help", "tell", "ask", "feel", "seem", "leave",
    "call", "keep", "let", "begin", "seem", "talk", "turn", "start", "show", "hear",
    "play", "run", "move", "live", "believe", "hold", "bring", "happen", "write", "provide",
    "sit", "stand", "lose", "pay", "meet", "include", "continue", "set", "learn", "change",
    "lead", "understand", "watch", "follow", "stop", "create", "speak", "read", "allow", "add",
    "spend", "grow", "open", "walk", "win", "offer", "remember", "love", "consider", "appear",
    "buy", "wait", "serve", "die", "send", "expect", "build", "stay", "fall", "cut",
    "reach", "kill", "remain", "suggest", "raise", "pass", "sell", "require", "report", "decide",
    # Descriptive words
    "different", "same", "important", "large", "small", "big", "old", "young", "long", "short",
    "high", "low", "early", "late", "hard", "easy", "best", "better", "real", "true",
    "right", "wrong", "certain", "free", "full", "special", "clear", "sure", "strong", "possible",
    "whole", "great", "national", "local", "political", "social", "public", "private", "major", "general",
    "human", "natural", "economic", "financial", "personal", "physical", "mental", "cultural", "medical", "legal",
    # Abstract concepts
    "world", "life", "hand", "part", "child", "eye", "woman", "man", "case", "week",
    "company", "system", "program", "question", "government", "number", "night", "point", "home", "water",
    "room", "mother", "area", "money", "story", "fact", "month", "lot", "right", "study",
    "book", "job", "word", "business", "issue", "side", "kind", "head", "house", "service",
    "friend", "father", "power", "hour", "game", "line", "end", "member", "law", "car",
    "city", "community", "name", "president", "team", "minute", "idea", "kid", "body", "information",
    "school", "parent", "face", "others", "level", "office", "door", "health", "person", "art",
    "war", "history", "party", "result", "change", "morning", "reason", "research", "girl", "guy",
    "moment", "air", "teacher", "force", "education", "foot", "boy", "age", "policy", "process",
    "music", "market", "sense", "nation", "plan", "college", "interest", "death", "experience", "effect",
    # Technology and science
    "computer", "internet", "software", "network", "data", "technology", "system", "digital", "online", "website",
    "email", "phone", "device", "machine", "science", "research", "study", "theory", "method", "analysis",
    "result", "evidence", "example", "model", "pattern", "structure", "function", "process", "development", "growth",
    "energy", "environment", "climate", "nature", "earth", "space", "universe", "planet", "star", "light",
    # Philosophy and consciousness
    "mind", "thought", "consciousness", "awareness", "perception", "reality", "truth", "knowledge", "wisdom", "understanding",
    "meaning", "purpose", "value", "belief", "faith", "spirit", "soul", "being", "existence", "essence",
    "reason", "logic", "emotion", "feeling", "experience", "memory", "imagination", "creativity", "intuition", "insight",
    # Geometry and mathematics
    "geometry", "mathematics", "number", "equation", "formula", "calculation", "measurement", "dimension", "space", "time",
    "point", "line", "plane", "surface", "volume", "angle", "curve", "circle", "sphere", "triangle",
    "square", "cube", "vector", "matrix", "tensor", "manifold", "topology", "symmetry", "transformation", "rotation",
    # Communication
    "language", "word", "sentence", "paragraph", "text", "message", "communication", "conversation", "discussion", "debate",
    "argument", "opinion", "statement", "question", "answer", "response", "comment", "feedback", "criticism", "praise",
    # Relationships
    "relationship", "connection", "interaction", "cooperation", "collaboration", "partnership", "friendship", "family", "community", "society",
    "culture", "tradition", "custom", "ritual", "ceremony", "celebration", "festival", "holiday", "event", "occasion",
    # Emotions
    "happiness", "sadness", "anger", "fear", "surprise", "disgust", "joy", "sorrow", "love", "hate",
    "hope", "despair", "confidence", "doubt", "pride", "shame", "guilt", "gratitude", "envy", "jealousy",
    # Actions and states
    "action", "activity", "behavior", "conduct", "performance", "achievement", "accomplishment", "success", "failure", "progress",
    "improvement", "development", "growth", "change", "transformation", "evolution", "revolution", "innovation", "discovery", "invention",
    # Additional common words
    "actually", "always", "another", "anything", "around", "away", "before", "between", "both", "down",
    "during", "each", "enough", "every", "everything", "few", "got", "however", "last", "less",
    "little", "made", "many", "might", "more", "much", "must", "never", "nothing", "often",
    "once", "own", "perhaps", "please", "put", "quite", "rather", "really", "said", "several",
    "should", "since", "sometimes", "something", "soon", "still", "such", "sure", "though", "through",
    "today", "together", "too", "under", "until", "upon", "very", "while", "without", "yet",
    "young", "yourself", "zero", "above", "across", "against", "along", "already", "although", "among",
    "became", "become", "been", "behind", "below", "beneath", "beside", "beyond", "brought", "building",
    "called", "came", "cannot", "center", "century", "certain", "chance", "children", "class", "close",
    "coming", "common", "complete", "control", "cost", "country", "course", "current", "dark", "deal",
    "doing", "done", "drive", "either", "else", "entire", "especially", "even", "ever", "everyone",
    "everything", "exactly", "except", "experience", "explain", "express", "fact", "family", "far", "fast",
    "father", "feeling", "field", "figure", "final", "finally", "fine", "fire", "focus", "following",
    "form", "former", "forward", "found", "four", "front", "future", "gave", "getting", "given",
    "going", "gone", "gotten", "government", "ground", "group", "guess", "half", "happened", "happy",
    "having", "heart", "held", "herself", "himself", "hope", "important", "including", "information", "instead",
    "involved", "itself", "known", "large", "later", "least", "letter", "likely", "living", "looking",
    "making", "maybe", "mean", "means", "meeting", "middle", "mind", "miss", "moment", "morning",
    "myself", "near", "nearly", "need", "needed", "neither", "news", "next", "north", "note",
    "noticed", "numbers", "order", "others", "outside", "paper", "particular", "past", "perhaps", "period",
    "picture", "place", "playing", "point", "position", "present", "probably", "problem", "product", "programs",
    "public", "put", "question", "quite", "range", "rate", "reason", "recent", "record", "return",
]


def compute_basin_embedding(word: str, dim: int = 64) -> np.ndarray:
    """Compute a deterministic 64D basin embedding for a word using geometric methods.
    
    Uses golden ratio modulation and hash-based seeding for reproducible embeddings
    that lie on the unit sphere (Fisher manifold).
    """
    # Create deterministic seed from word
    word_hash = hashlib.sha256(word.encode()).hexdigest()
    seed = int(word_hash[:8], 16)
    rng = np.random.RandomState(seed)
    
    # Generate base embedding using golden ratio modulation
    embedding = np.zeros(dim)
    for i in range(dim):
        # Use golden ratio for quasi-random distribution
        theta = 2 * np.pi * ((i * PHI) % 1)
        # Modulate with word-specific hash
        char_idx = i % len(word)
        char_val = ord(word[char_idx]) / 256.0
        # Combine deterministic and random components
        embedding[i] = np.cos(theta + char_val * np.pi) + rng.randn() * 0.1
    
    # Project onto unit sphere (Fisher manifold)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    return embedding


def compute_phi_score(word: str) -> float:
    """Compute a phi score for a word based on its properties.
    
    Higher phi for:
    - Longer words (more information)
    - More common words (higher utility)
    - Words with balanced letter distribution
    """
    # Base score from length (normalized)
    length_score = min(len(word) / 10.0, 1.0)
    
    # Entropy-based score from letter distribution
    if len(word) > 0:
        letter_counts = {}
        for c in word.lower():
            letter_counts[c] = letter_counts.get(c, 0) + 1
        probs = np.array(list(letter_counts.values())) / len(word)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(len(letter_counts)) if len(letter_counts) > 1 else 1
        entropy_score = entropy / max_entropy if max_entropy > 0 else 0.5
    else:
        entropy_score = 0.5
    
    # Combine scores
    phi = 0.3 * length_score + 0.7 * entropy_score
    return float(np.clip(phi, 0.1, 1.0))


def load_bip39_words() -> List[str]:
    """Load BIP39 wordlist."""
    words = []
    
    # Try server path first
    if BIP39_PATH.exists():
        with open(BIP39_PATH) as f:
            words = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(words)} BIP39 words from {BIP39_PATH}")
        return words
    
    # Try qig-backend path
    alt_path = Path(__file__).parent / 'bip39-wordlist.txt'
    if alt_path.exists():
        with open(alt_path) as f:
            words = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(words)} BIP39 words from {alt_path}")
        return words
    
    logger.warning("BIP39 wordlist not found, using fallback")
    # Fallback: first 100 BIP39 words
    return [
        "abandon", "ability", "able", "about", "above", "absent", "absorb", "abstract", "absurd", "abuse",
        "access", "accident", "account", "accuse", "achieve", "acid", "acoustic", "acquire", "across", "act",
        "action", "actor", "actress", "actual", "adapt", "add", "addict", "address", "adjust", "admit",
        "adult", "advance", "advice", "aerobic", "affair", "afford", "afraid", "again", "age", "agent",
        "agree", "ahead", "aim", "air", "airport", "aisle", "alarm", "album", "alcohol", "alert",
        "alien", "all", "alley", "allow", "almost", "alone", "alpha", "already", "also", "alter",
        "always", "amateur", "amazing", "among", "amount", "amused", "analyst", "anchor", "ancient", "anger",
        "angle", "angry", "animal", "ankle", "announce", "annual", "another", "answer", "antenna", "antique",
        "anxiety", "any", "apart", "apology", "appear", "apple", "approve", "april", "arch", "arctic",
        "area", "arena", "argue", "arm", "armed", "armor", "army", "around", "arrange", "arrest",
    ]


def get_all_words() -> List[Tuple[str, str]]:
    """Get all words with their source types.
    
    Returns list of (word, source_type) tuples.
    """
    words = []
    seen = set()
    
    # BIP39 words (highest priority)
    for word in load_bip39_words():
        word_lower = word.lower().strip()
        if word_lower and word_lower.isalpha() and len(word_lower) >= 3 and word_lower not in seen:
            words.append((word_lower, 'bip39'))
            seen.add(word_lower)
    
    # Common English words
    for word in COMMON_ENGLISH_WORDS:
        word_lower = word.lower().strip()
        if word_lower and word_lower.isalpha() and len(word_lower) >= 3 and word_lower not in seen:
            words.append((word_lower, 'common'))
            seen.add(word_lower)
    
    logger.info(f"Total unique words: {len(words)}")
    return words


def format_vector_for_pg(vec: np.ndarray) -> str:
    """Format numpy vector for PostgreSQL vector type."""
    return '[' + ','.join(f'{v:.8f}' for v in vec) + ']'


def populate_vocabulary(dry_run: bool = False):
    """Populate tokenizer_vocabulary table with real words."""
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        logger.error("DATABASE_URL not set")
        return False
    
    words = get_all_words()
    logger.info(f"Preparing to insert {len(words)} words")
    
    if dry_run:
        logger.info("DRY RUN - showing first 20 words:")
        for word, source in words[:20]:
            embedding = compute_basin_embedding(word)
            phi = compute_phi_score(word)
            logger.info(f"  {word} ({source}): phi={phi:.3f}, embedding_norm={np.linalg.norm(embedding):.4f}")
        return True
    
    try:
        conn = psycopg2.connect(database_url)
        conn.autocommit = False
        cur = conn.cursor()
        
        # Ensure vector extension exists
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Ensure table exists
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
        
        # Get max token_id
        cur.execute("SELECT COALESCE(MAX(token_id), 0) FROM tokenizer_vocabulary")
        max_token_id = cur.fetchone()[0]
        
        # Insert words in batches
        batch_size = 100
        inserted = 0
        skipped = 0
        
        for i in range(0, len(words), batch_size):
            batch = words[i:i+batch_size]
            
            for word, source_type in batch:
                token_id = max_token_id + inserted + 1
                embedding = compute_basin_embedding(word)
                phi_score = compute_phi_score(word)
                embedding_str = format_vector_for_pg(embedding)
                
                try:
                    cur.execute("""
                        INSERT INTO tokenizer_vocabulary 
                        (token, token_id, weight, frequency, phi_score, basin_embedding, source_type)
                        VALUES (%s, %s, %s, %s, %s, %s::vector, %s)
                        ON CONFLICT (token) DO UPDATE SET
                            phi_score = EXCLUDED.phi_score,
                            basin_embedding = EXCLUDED.basin_embedding,
                            source_type = EXCLUDED.source_type,
                            updated_at = CURRENT_TIMESTAMP
                    """, (word, token_id, 1.0, 1, phi_score, embedding_str, source_type))
                    inserted += 1
                except psycopg2.IntegrityError:
                    conn.rollback()
                    skipped += 1
                    continue
            
            conn.commit()
            logger.info(f"Progress: {min(i+batch_size, len(words))}/{len(words)} words processed")
        
        # Verify
        cur.execute("SELECT source_type, COUNT(*) FROM tokenizer_vocabulary GROUP BY source_type ORDER BY COUNT(*) DESC")
        logger.info("\nFinal vocabulary distribution:")
        for row in cur.fetchall():
            logger.info(f"  {row[0]}: {row[1]}")
        
        cur.execute("SELECT COUNT(*) FROM tokenizer_vocabulary WHERE source_type IN ('bip39', 'common')")
        total_real_words = cur.fetchone()[0]
        logger.info(f"\nTotal real words: {total_real_words}")
        
        conn.close()
        logger.info(f"\nMigration complete: {inserted} inserted, {skipped} skipped")
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Populate tokenizer_vocabulary with real words')
    parser.add_argument('--dry-run', action='store_true', help='Preview without making changes')
    args = parser.parse_args()
    
    success = populate_vocabulary(dry_run=args.dry_run)
    sys.exit(0 if success else 1)
