"""
Vocabulary Enhancer - Learn word coordinates from curriculum documents.

Reads curriculum files, extracts words with context, computes 64D basin 
coordinates using Fisher-compliant geometry, and populates PostgreSQL.
"""

import os
import re
import logging
import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASIN_DIM = 64
CURRICULUM_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "09-curriculum")
MIN_WORD_LENGTH = 3
MAX_WORD_LENGTH = 20
MIN_WORD_FREQUENCY = 2
CONTEXT_WINDOW = 5

STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this',
    'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their',
    'we', 'our', 'you', 'your', 'he', 'she', 'him', 'her', 'his',
    'not', 'no', 'yes', 'all', 'any', 'some', 'each', 'every', 'both',
    'few', 'more', 'most', 'other', 'such', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there', 'when',
    'where', 'why', 'how', 'what', 'which', 'who', 'whom', 'whose'
}

SEMANTIC_DOMAINS = {
    'mathematics': ['geometry', 'manifold', 'metric', 'tensor', 'vector', 'matrix', 
                   'eigenvalue', 'derivative', 'integral', 'topology', 'algebra',
                   'calculus', 'dimension', 'space', 'function', 'equation'],
    'physics': ['quantum', 'entropy', 'energy', 'wave', 'particle', 'field',
               'momentum', 'force', 'mass', 'velocity', 'acceleration', 'photon',
               'electron', 'atom', 'nucleus', 'spin', 'state', 'measurement'],
    'consciousness': ['awareness', 'attention', 'perception', 'cognition', 'mind',
                     'thought', 'experience', 'qualia', 'integration', 'binding',
                     'phenomenal', 'subjective', 'conscious', 'unconscious'],
    'information': ['bit', 'entropy', 'channel', 'signal', 'noise', 'coding',
                   'compression', 'transmission', 'data', 'pattern', 'structure',
                   'complexity', 'redundancy', 'mutual', 'conditional'],
    'computation': ['algorithm', 'compute', 'process', 'memory', 'storage',
                   'input', 'output', 'program', 'code', 'function', 'variable',
                   'loop', 'condition', 'recursion', 'iteration'],
    'mythology': ['zeus', 'athena', 'apollo', 'ares', 'hermes', 'hephaestus',
                 'artemis', 'dionysus', 'demeter', 'poseidon', 'hera', 'aphrodite',
                 'olympus', 'pantheon', 'god', 'goddess', 'titan', 'oracle'],
    'geometry_qig': ['fisher', 'rao', 'bures', 'geodesic', 'curvature', 'connection',
                    'parallel', 'transport', 'tangent', 'cotangent', 'bundle',
                    'symplectic', 'riemannian', 'metric', 'distance', 'proximity'],
    'kernel': ['kernel', 'basin', 'attractor', 'trajectory', 'convergence',
              'stability', 'equilibrium', 'dynamics', 'evolution', 'flow',
              'phase', 'bifurcation', 'chaos', 'order', 'emergence']
}

DOMAIN_BASIN_SEEDS = {
    'mathematics': 1001,
    'physics': 2002,
    'consciousness': 3003,
    'information': 4004,
    'computation': 5005,
    'mythology': 6006,
    'geometry_qig': 7007,
    'kernel': 8008
}


def sphere_project(v: np.ndarray) -> np.ndarray:
    """Project vector onto unit sphere (L2 normalization)."""
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        v = np.random.randn(len(v))
        norm = np.linalg.norm(v)
    return v / norm


def compute_domain_embedding(domain: str) -> np.ndarray:
    """Compute stable 64D embedding for a semantic domain."""
    seed = DOMAIN_BASIN_SEEDS.get(domain, hash(domain) % (2**31))
    np.random.seed(seed)
    raw = np.random.dirichlet(np.ones(BASIN_DIM))
    return sphere_project(raw)


def detect_word_domains(word: str) -> List[Tuple[str, float]]:
    """Detect which semantic domains a word belongs to."""
    word_lower = word.lower()
    domains = []
    
    for domain, keywords in SEMANTIC_DOMAINS.items():
        for kw in keywords:
            if word_lower == kw or kw in word_lower or word_lower in kw:
                similarity = 1.0 if word_lower == kw else 0.7
                domains.append((domain, similarity))
                break
    
    return domains


def compute_word_embedding(word: str, context_words: List[str], 
                          word_cooccurrence: Dict[str, Dict[str, int]]) -> np.ndarray:
    """
    Compute 64D basin embedding for a word using:
    1. Semantic domain membership
    2. Co-occurrence patterns
    3. Character-level features
    """
    embedding = np.zeros(BASIN_DIM)
    
    domains = detect_word_domains(word)
    if domains:
        for domain, weight in domains:
            domain_emb = compute_domain_embedding(domain)
            embedding += weight * domain_emb
    
    if word in word_cooccurrence:
        cooc = word_cooccurrence[word]
        total_cooc = sum(cooc.values())
        if total_cooc > 0:
            for ctx_word, count in cooc.items():
                ctx_domains = detect_word_domains(ctx_word)
                for domain, _ in ctx_domains:
                    domain_emb = compute_domain_embedding(domain)
                    weight = count / total_cooc
                    embedding += 0.3 * weight * domain_emb
    
    np.random.seed(hash(word) % (2**31))
    char_features = np.random.randn(BASIN_DIM) * 0.1
    embedding += char_features
    
    return sphere_project(embedding)


def extract_words_from_file(filepath: str) -> List[str]:
    """Extract clean words from a file."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        logger.warning(f"Could not read {filepath}: {e}")
        return []
    
    content = re.sub(r'```[\s\S]*?```', ' ', content)
    content = re.sub(r'`[^`]+`', ' ', content)
    content = re.sub(r'https?://\S+', ' ', content)
    content = re.sub(r'[^a-zA-Z\s]', ' ', content)
    
    words = content.lower().split()
    
    clean_words = []
    for word in words:
        if (MIN_WORD_LENGTH <= len(word) <= MAX_WORD_LENGTH 
            and word.isalpha() 
            and word not in STOPWORDS):
            clean_words.append(word)
    
    return clean_words


def build_cooccurrence(all_words: List[str], window: int = CONTEXT_WINDOW) -> Dict[str, Dict[str, int]]:
    """Build word co-occurrence matrix from word sequence."""
    cooccurrence = defaultdict(lambda: defaultdict(int))
    
    for i, word in enumerate(all_words):
        start = max(0, i - window)
        end = min(len(all_words), i + window + 1)
        
        for j in range(start, end):
            if i != j:
                ctx_word = all_words[j]
                cooccurrence[word][ctx_word] += 1
    
    return dict(cooccurrence)


def scan_curriculum_directory() -> List[str]:
    """Find all curriculum files."""
    curriculum_path = Path(CURRICULUM_DIR)
    if not curriculum_path.exists():
        logger.warning(f"Curriculum directory not found: {CURRICULUM_DIR}")
        return []
    
    files = []
    for ext in ['*.md', '*.txt']:
        files.extend(curriculum_path.glob(f'**/{ext}'))
    
    return [str(f) for f in files]


def compute_phi_score(embedding: np.ndarray) -> float:
    """Compute integration score (Φ) from embedding entropy."""
    p = np.abs(embedding) + 1e-10
    p = p / np.sum(p)
    entropy = -np.sum(p * np.log(p + 1e-10))
    max_entropy = np.log(len(embedding))
    phi = 1.0 - (entropy / max_entropy)
    return float(np.clip(phi, 0.0, 1.0))


def insert_vocabulary_to_postgres(words_data: List[Dict], batch_size: int = 200) -> int:
    """Insert vocabulary into PostgreSQL tokenizer_vocabulary table using batch inserts."""
    import psycopg2
    
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        logger.error("DATABASE_URL not set")
        return 0
    
    try:
        conn = psycopg2.connect(database_url)
        cur = conn.cursor()
        
        total_processed = 0
        
        for i in range(0, len(words_data), batch_size):
            batch = words_data[i:i + batch_size]
            
            for word_data in batch:
                token = word_data['token']
                embedding = word_data['embedding']
                phi_score = word_data['phi_score']
                frequency = word_data['frequency']
                embedding_str = '[' + ','.join(str(x) for x in embedding) + ']'
                
                cur.execute("""
                    INSERT INTO tokenizer_vocabulary 
                    (token, basin_embedding, phi_score, frequency, source_type, created_at)
                    VALUES (%s, %s, %s, %s, 'curriculum_learned', NOW())
                    ON CONFLICT (token) DO UPDATE SET
                        basin_embedding = EXCLUDED.basin_embedding,
                        phi_score = GREATEST(tokenizer_vocabulary.phi_score, EXCLUDED.phi_score),
                        frequency = tokenizer_vocabulary.frequency + EXCLUDED.frequency,
                        updated_at = NOW()
                """, (token, embedding_str, phi_score, frequency))
            
            conn.commit()
            total_processed += len(batch)
            if (i // batch_size + 1) % 5 == 0:
                logger.info(f"Processed {total_processed}/{len(words_data)} words...")
        
        cur.close()
        conn.close()
        
        logger.info(f"Total words processed: {total_processed}")
        return total_processed
        
    except Exception as e:
        logger.error(f"Database error: {e}")
        import traceback
        traceback.print_exc()
        return 0


def enhance_vocabulary(min_frequency: int = MIN_WORD_FREQUENCY) -> Dict:
    """
    Main function to enhance vocabulary from curriculum documents.
    
    Returns statistics about the enhancement process.
    """
    logger.info("Starting vocabulary enhancement from curriculum...")
    
    files = scan_curriculum_directory()
    logger.info(f"Found {len(files)} curriculum files")
    
    if not files:
        return {'error': 'No curriculum files found', 'files': 0, 'words': 0}
    
    all_words = []
    word_frequency = defaultdict(int)
    
    for filepath in files:
        words = extract_words_from_file(filepath)
        all_words.extend(words)
        for word in words:
            word_frequency[word] += 1
    
    logger.info(f"Extracted {len(all_words)} total words, {len(word_frequency)} unique")
    
    frequent_words = {w: f for w, f in word_frequency.items() if f >= min_frequency}
    logger.info(f"Words with frequency >= {min_frequency}: {len(frequent_words)}")
    
    logger.info("Building co-occurrence matrix...")
    cooccurrence = build_cooccurrence(all_words)
    
    logger.info("Computing word embeddings...")
    words_data = []
    
    for word, freq in frequent_words.items():
        context_words = list(cooccurrence.get(word, {}).keys())
        embedding = compute_word_embedding(word, context_words, cooccurrence)
        phi_score = compute_phi_score(embedding)
        
        words_data.append({
            'token': word,
            'embedding': embedding.tolist(),
            'phi_score': phi_score,
            'frequency': freq
        })
    
    logger.info(f"Computed embeddings for {len(words_data)} words")
    
    logger.info("Inserting into PostgreSQL...")
    total_inserted = insert_vocabulary_to_postgres(words_data)
    
    domain_word_counts = defaultdict(int)
    for wd in words_data:
        domains = detect_word_domains(wd['token'])
        for domain, _ in domains:
            domain_word_counts[domain] += 1
    
    stats = {
        'files_processed': len(files),
        'total_words': len(all_words),
        'unique_words': len(word_frequency),
        'frequent_words': len(frequent_words),
        'words_with_embeddings': len(words_data),
        'words_inserted_or_updated': total_inserted,
        'domain_coverage': dict(domain_word_counts),
        'avg_phi_score': np.mean([w['phi_score'] for w in words_data]) if words_data else 0
    }
    
    logger.info(f"Enhancement complete: {stats}")
    return stats


if __name__ == '__main__':
    stats = enhance_vocabulary()
    print("\n=== Vocabulary Enhancement Results ===")
    for key, value in stats.items():
        print(f"  {key}: {value}")
