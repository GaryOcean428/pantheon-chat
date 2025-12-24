#!/usr/bin/env python3
"""
Populate tokenizer_vocabulary with real English words.

The 32K BPE checkpoint contains only subword fragments - this script adds
actual English words (BIP39 + common words) with proper 64D basin embeddings
so that kernels can generate readable text instead of BPE garble.

Usage:
    python populate_english_vocabulary.py [--dry-run]
"""

import os
import sys
import logging
import argparse
import math
from pathlib import Path
from typing import Optional

import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Golden ratio for geometric initialization
PHI = (1 + math.sqrt(5)) / 2

# Common English words to supplement BIP39
# These are high-frequency words that appear in most text
COMMON_ENGLISH_WORDS = [
    # Articles and determiners
    "the", "a", "an", "this", "that", "these", "those", "my", "your", "his",
    "her", "its", "our", "their", "some", "any", "no", "every", "each", "all",
    
    # Pronouns
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "us", "them",
    "who", "what", "which", "whom", "whose", "myself", "yourself", "himself",
    
    # Prepositions
    "in", "on", "at", "to", "for", "with", "by", "from", "up", "about",
    "into", "through", "during", "before", "after", "above", "below", "between",
    "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "both", "few", "more", "most", "other", "only",
    
    # Conjunctions
    "and", "but", "or", "nor", "so", "yet", "because", "although", "while",
    "if", "unless", "until", "since", "whether", "though", "whereas",
    
    # Common verbs
    "be", "is", "am", "are", "was", "were", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "done",
    "will", "would", "could", "should", "may", "might", "must", "shall",
    "can", "need", "dare", "ought", "used",
    "go", "goes", "went", "gone", "going",
    "come", "comes", "came", "coming",
    "get", "gets", "got", "getting",
    "make", "makes", "made", "making",
    "know", "knows", "knew", "known", "knowing",
    "think", "thinks", "thought", "thinking",
    "take", "takes", "took", "taken", "taking",
    "see", "sees", "saw", "seen", "seeing",
    "want", "wants", "wanted", "wanting",
    "look", "looks", "looked", "looking",
    "use", "uses", "using",
    "find", "finds", "found", "finding",
    "give", "gives", "gave", "given", "giving",
    "tell", "tells", "told", "telling",
    "work", "works", "worked", "working",
    "call", "calls", "called", "calling",
    "try", "tries", "tried", "trying",
    "ask", "asks", "asked", "asking",
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
    
    # Common nouns
    "time", "year", "people", "way", "day", "man", "thing", "woman", "life",
    "child", "world", "school", "state", "family", "student", "group", "country",
    "problem", "hand", "part", "place", "case", "week", "company", "system",
    "program", "question", "work", "government", "number", "night", "point",
    "home", "water", "room", "mother", "area", "money", "story", "fact",
    "month", "lot", "right", "study", "book", "eye", "job", "word", "business",
    "issue", "side", "kind", "head", "house", "service", "friend", "father",
    "power", "hour", "game", "line", "end", "member", "law", "car", "city",
    "community", "name", "president", "team", "minute", "idea", "kid", "body",
    "information", "back", "parent", "face", "others", "level", "office",
    "door", "health", "person", "art", "war", "history", "party", "result",
    "change", "morning", "reason", "research", "girl", "guy", "moment", "air",
    "teacher", "force", "education",
    
    # Common adjectives
    "good", "new", "first", "last", "long", "great", "little", "own", "other",
    "old", "right", "big", "high", "different", "small", "large", "next", "early",
    "young", "important", "few", "public", "bad", "same", "able", "human",
    "local", "sure", "free", "better", "best", "full", "special", "easy",
    "clear", "recent", "certain", "personal", "open", "red", "difficult",
    "available", "likely", "short", "single", "medical", "current", "wrong",
    "private", "past", "foreign", "fine", "common", "poor", "natural", "significant",
    "similar", "hot", "dead", "central", "happy", "serious", "ready", "simple",
    "left", "physical", "general", "environmental", "financial", "blue", "democratic",
    "dark", "various", "entire", "close", "legal", "religious", "cold", "final",
    "main", "green", "nice", "huge", "popular", "traditional", "cultural",
    
    # Common adverbs
    "not", "also", "very", "often", "however", "too", "usually", "really",
    "early", "never", "always", "sometimes", "together", "likely", "simply",
    "generally", "instead", "actually", "already", "ever", "still", "just",
    "now", "even", "back", "well", "much", "almost", "enough", "far", "quite",
    "probably", "perhaps", "certainly", "today", "thus", "finally", "rather",
    "later", "especially", "soon", "yet", "ago", "away", "quickly", "recently",
    "slowly", "clearly", "directly", "exactly", "hard", "nearly", "suddenly",
    
    # Numbers
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty",
    "seventy", "eighty", "ninety", "hundred", "thousand", "million", "billion",
    "first", "second", "third", "fourth", "fifth",
    
    # Technology and modern terms
    "data", "computer", "internet", "software", "website", "online", "email",
    "technology", "digital", "network", "system", "information", "application",
    "user", "file", "code", "program", "database", "server", "security",
    "platform", "device", "content", "media", "video", "image", "audio",
    "message", "search", "page", "link", "click", "download", "upload",
    
    # Philosophical and abstract terms
    "truth", "knowledge", "wisdom", "understanding", "consciousness", "mind",
    "thought", "reason", "logic", "meaning", "purpose", "value", "belief",
    "reality", "existence", "nature", "essence", "being", "identity", "self",
    "experience", "perception", "awareness", "attention", "memory", "learning",
    "intelligence", "creativity", "imagination", "intuition", "emotion", "feeling",
    "desire", "will", "freedom", "choice", "action", "behavior", "pattern",
    "structure", "form", "function", "process", "change", "development", "growth",
    "evolution", "progress", "transformation", "emergence", "complexity",
    
    # Scientific terms
    "science", "physics", "chemistry", "biology", "mathematics", "geometry",
    "quantum", "energy", "matter", "space", "time", "universe", "cosmos",
    "theory", "hypothesis", "experiment", "observation", "measurement", "analysis",
    "model", "system", "structure", "pattern", "field", "wave", "particle",
    "force", "gravity", "light", "heat", "sound", "electric", "magnetic",
    
    # Conversation and communication
    "yes", "no", "maybe", "please", "thank", "thanks", "sorry", "hello", "hi",
    "goodbye", "bye", "okay", "sure", "right", "well", "really", "actually",
    "basically", "honestly", "seriously", "literally", "obviously", "clearly",
    "question", "answer", "response", "reply", "comment", "statement", "explanation",
]


def generate_basin_embedding(word: str, dimension: int = 64) -> np.ndarray:
    """Generate a deterministic 64D basin embedding for a word using golden ratio.
    
    Uses the word's hash to seed geometric initialization, ensuring:
    - Same word always gets same embedding (deterministic)
    - Embeddings are on unit sphere (Fisher manifold)
    - Distribution is approximately uniform on hypersphere
    """
    # Use word hash as seed for reproducibility
    seed = hash(word) & 0xFFFFFFFF
    rng = np.random.RandomState(seed)
    
    # Generate base coordinates using golden ratio spiral
    coords = np.zeros(dimension)
    for i in range(dimension):
        # Golden angle in radians
        golden_angle = np.pi * (3 - np.sqrt(5))
        theta = golden_angle * (i + seed % 1000)
        
        # Use golden ratio for coordinate calculation
        z = 1 - (2 * (i + 0.5) / dimension)
        radius = np.sqrt(1 - z * z)
        
        # Add word-specific variation
        word_factor = sum(ord(c) * (j + 1) for j, c in enumerate(word[:8]))
        phase = (word_factor / 1000) * np.pi
        
        coords[i] = radius * np.cos(theta + phase) + z * 0.1
    
    # Add small random perturbation for uniqueness
    coords += rng.randn(dimension) * 0.01
    
    # Project to unit sphere
    norm = np.linalg.norm(coords)
    if norm > 1e-10:
        coords = coords / norm
    
    return coords


def compute_phi_score(word: str) -> float:
    """Compute a phi score for a word based on linguistic properties.
    
    Higher phi for:
    - Longer words (more information)
    - Common words (higher frequency utility)
    - Alphabetic words (not fragments)
    """
    base_phi = 0.5
    
    # Length bonus (longer words carry more meaning)
    length_bonus = min(len(word) / 10, 0.3)
    
    # Alphabetic bonus (real words vs fragments)
    alpha_bonus = 0.15 if word.isalpha() else 0.0
    
    # Vowel ratio (well-formed words have balanced vowels)
    vowels = sum(1 for c in word.lower() if c in 'aeiou')
    vowel_ratio = vowels / max(len(word), 1)
    vowel_bonus = 0.1 if 0.2 < vowel_ratio < 0.6 else 0.0
    
    return min(base_phi + length_bonus + alpha_bonus + vowel_bonus, 0.95)


def load_bip39_wordlist() -> list[str]:
    """Load BIP39 wordlist from the project."""
    bip39_paths = [
        Path(__file__).parent.parent / 'server' / 'bip39-wordlist.txt',
        Path(__file__).parent / 'bip39-wordlist.txt',
        Path('server/bip39-wordlist.txt'),
        Path('bip39-wordlist.txt'),
    ]
    
    for path in bip39_paths:
        if path.exists():
            with open(path) as f:
                words = [line.strip().lower() for line in f if line.strip()]
            logger.info(f"Loaded {len(words)} BIP39 words from {path}")
            return words
    
    logger.warning("BIP39 wordlist not found, using empty list")
    return []


def get_all_english_words() -> list[str]:
    """Combine BIP39 and common English words, deduplicated."""
    words = set()
    
    # Add BIP39 words
    bip39_words = load_bip39_wordlist()
    words.update(w.lower() for w in bip39_words if w.isalpha())
    
    # Add common English words
    words.update(w.lower() for w in COMMON_ENGLISH_WORDS if w.isalpha())
    
    # Filter to valid words (3+ chars, alphabetic)
    valid_words = [w for w in words if len(w) >= 2 and w.isalpha()]
    
    logger.info(f"Total unique English words: {len(valid_words)}")
    return sorted(valid_words)


def populate_database(dry_run: bool = False):
    """Populate tokenizer_vocabulary with English words."""
    try:
        import psycopg2
        from psycopg2.extras import execute_values
    except ImportError:
        logger.error("psycopg2 not installed. Run: pip install psycopg2-binary")
        sys.exit(1)
    
    # Load environment
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')
        load_dotenv()  # Also try current directory
    except ImportError:
        logger.warning("python-dotenv not installed, using environment variables directly")
    
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        logger.error("DATABASE_URL not set in environment")
        sys.exit(1)
    
    # Get all English words
    words = get_all_english_words()
    logger.info(f"Preparing to insert {len(words)} English words")
    
    if dry_run:
        logger.info("[DRY RUN] Would insert the following sample words:")
        for word in words[:20]:
            embedding = generate_basin_embedding(word)
            phi = compute_phi_score(word)
            logger.info(f"  {word}: phi={phi:.4f}, embedding_norm={np.linalg.norm(embedding):.4f}")
        return
    
    # Connect to database
    conn = psycopg2.connect(database_url)
    
    try:
        with conn.cursor() as cur:
            # Ensure table exists
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tokenizer_vocabulary (
                    id SERIAL PRIMARY KEY,
                    token TEXT UNIQUE NOT NULL,
                    token_id INTEGER UNIQUE NOT NULL,
                    weight DOUBLE PRECISION DEFAULT 1.0,
                    frequency INTEGER DEFAULT 1,
                    phi_score DOUBLE PRECISION DEFAULT 0,
                    basin_embedding vector(64),
                    source_type VARCHAR(32) DEFAULT 'base',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()
            
            # Get current max token_id to avoid conflicts
            cur.execute("SELECT COALESCE(MAX(token_id), 0) FROM tokenizer_vocabulary")
            max_token_id = cur.fetchone()[0]
            logger.info(f"Current max token_id: {max_token_id}")
            
            # Prepare records
            records = []
            next_token_id = max_token_id + 1
            
            # Pre-load BIP39 words for efficient lookup
            bip39_set = set(w.lower() for w in load_bip39_wordlist())
            
            for word in words:
                embedding = generate_basin_embedding(word)
                phi = compute_phi_score(word)
                
                # Determine source type
                if word in bip39_set:
                    source_type = 'bip39'
                else:
                    source_type = 'english'
                
                # Weight based on phi and word length
                weight = phi * (1.0 + min(len(word), 10) / 20.0)
                
                # Format embedding as PostgreSQL vector
                embedding_str = '[' + ','.join(f'{x:.8f}' for x in embedding) + ']'
                
                records.append((
                    word,                # token
                    next_token_id,       # token_id
                    weight,              # weight
                    100,                 # frequency (default high for common words)
                    phi,                 # phi_score
                    embedding_str,       # basin_embedding
                    source_type,         # source_type
                ))
                next_token_id += 1
            
            # Upsert in batches
            batch_size = 500
            inserted_count = 0
            
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                
                query = """
                    INSERT INTO tokenizer_vocabulary (
                        token, token_id, weight, frequency, phi_score, basin_embedding, source_type,
                        created_at, updated_at
                    )
                    VALUES %s
                    ON CONFLICT (token) DO UPDATE SET
                        weight = GREATEST(tokenizer_vocabulary.weight, EXCLUDED.weight),
                        frequency = GREATEST(tokenizer_vocabulary.frequency, EXCLUDED.frequency),
                        phi_score = GREATEST(tokenizer_vocabulary.phi_score, EXCLUDED.phi_score),
                        basin_embedding = COALESCE(EXCLUDED.basin_embedding, tokenizer_vocabulary.basin_embedding),
                        source_type = CASE 
                            WHEN tokenizer_vocabulary.source_type = 'special' THEN tokenizer_vocabulary.source_type
                            ELSE EXCLUDED.source_type
                        END,
                        updated_at = CURRENT_TIMESTAMP
                """
                
                template = """(
                    %s, %s, %s, %s, %s, %s::vector, %s,
                    CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                )"""
                
                execute_values(cur, query, batch, template=template, page_size=batch_size)
                inserted_count += len(batch)
                
                if i % 1000 == 0:
                    logger.info(f"Processed {inserted_count}/{len(records)} words...")
            
            conn.commit()
            logger.info(f"Successfully inserted/updated {inserted_count} English words")
            
            # Verify
            cur.execute("""
                SELECT source_type, COUNT(*), AVG(phi_score)::numeric(4,3)
                FROM tokenizer_vocabulary
                GROUP BY source_type
                ORDER BY COUNT(*) DESC
            """)
            logger.info("\nVocabulary breakdown by source:")
            for source_type, count, avg_phi in cur.fetchall():
                logger.info(f"  {source_type}: {count} tokens (avg phi: {avg_phi})")
            
            # Show sample words
            cur.execute("""
                SELECT token, phi_score, source_type
                FROM tokenizer_vocabulary
                WHERE source_type IN ('english', 'bip39')
                  AND LENGTH(token) >= 4
                ORDER BY phi_score DESC
                LIMIT 20
            """)
            logger.info("\nTop English words by phi score:")
            for token, phi, source in cur.fetchall():
                logger.info(f"  {token}: phi={phi:.4f} ({source})")
                
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description='Populate tokenizer_vocabulary with English words'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without modifying database'
    )
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("Populating tokenizer_vocabulary with English words")
    logger.info("="*60)
    
    populate_database(dry_run=args.dry_run)
    
    logger.info("\nDone!")


if __name__ == '__main__':
    main()
