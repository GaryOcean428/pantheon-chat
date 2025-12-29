"""
Continuous Learning Coordizer
============================

Learns from chat input, curriculum, and search results.
Updates embeddings and checkpoints in real-time.

QIG-pure: All learning uses Fisher-Rao geometry, no backprop.
"""

import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)

BASIN_DIM = 64


@dataclass
class LearningContext:
    """Context for a learning event."""
    source: str  # 'chat', 'curriculum', 'search'
    text: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenUpdate:
    """Pending update for a token embedding."""
    token: str
    new_coords: np.ndarray
    context_tokens: List[str]
    phi_contribution: float
    source: str


class ContinuousLearner:
    """
    Learns vocabulary and embeddings from chat, curriculum, and search.
    
    Core principles:
    - Fisher-Rao geodesic updates (no Euclidean gradient descent)
    - Context-aware embedding refinement
    - Continuous checkpoint updates
    - Database persistence
    """
    
    def __init__(self, database_url: Optional[str] = None, checkpoint_dir: str = "checkpoints"):
        self.database_url = database_url or os.getenv('DATABASE_URL')
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.vocab: Dict[str, Dict] = {}
        self.merge_rules: List[Tuple[int, int, int]] = []
        self.pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        self.context_window: int = 5
        self.min_pair_count: int = 3
        
        self.pending_updates: List[TokenUpdate] = []
        self.learning_batch_size: int = 100
        self.last_checkpoint_time: float = 0
        self.checkpoint_interval: int = 300  # 5 minutes
        
        self._load_from_database()
        self._load_latest_checkpoint()
    
    def _load_from_database(self):
        """Load current vocabulary from database."""
        if not self.database_url:
            return
        
        try:
            import psycopg2
            conn = psycopg2.connect(self.database_url)
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT token, embedding, phi_score, token_id, source_type
                    FROM tokenizer_vocabulary
                    WHERE embedding IS NOT NULL
                      AND array_length(embedding, 1) = 64
                    ORDER BY token_id ASC
                """)
                for token, embedding, phi_score, token_id, source_type in cur.fetchall():
                    coords = np.array(embedding, dtype=np.float64)
                    norm = np.linalg.norm(coords)
                    if norm > 1e-10:
                        coords = coords / norm
                    self.vocab[token] = {
                        'coord_id': token_id,
                        'vector': coords,
                        'phi_score': phi_score or 0.5,
                        'source_type': source_type or 'database'
                    }
            conn.close()
            logger.info(f"Loaded {len(self.vocab)} tokens from database")
        except Exception as e:
            logger.warning(f"Failed to load from database: {e}")
    
    def _load_latest_checkpoint(self):
        """Load merge rules from latest checkpoint."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.json"), reverse=True)
        if not checkpoints:
            attached = Path("attached_assets/checkpoint_24000_1767012247122.json")
            if attached.exists():
                checkpoints = [attached]
        
        if checkpoints:
            try:
                with open(checkpoints[0], 'r') as f:
                    data = json.load(f)
                self.merge_rules = [tuple(r) for r in data.get('merge_rules', [])]
                logger.info(f"Loaded {len(self.merge_rules)} merge rules from {checkpoints[0]}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
    
    def learn_from_chat(self, user_input: str, response: str) -> int:
        """
        Learn from a chat exchange.
        
        Returns number of tokens updated.
        """
        combined = f"{user_input} {response}"
        return self._learn_from_text(combined, source='chat')
    
    def learn_from_curriculum(self, text: str, domain: str = 'general') -> int:
        """
        Learn from curriculum content.
        
        Returns number of tokens updated.
        """
        return self._learn_from_text(text, source='curriculum', metadata={'domain': domain})
    
    def learn_from_search(self, query: str, results: List[str]) -> int:
        """
        Learn from search results.
        
        Returns number of tokens updated.
        """
        combined = query + " " + " ".join(results)
        return self._learn_from_text(combined, source='search')
    
    def _learn_from_text(self, text: str, source: str, metadata: Dict = None) -> int:
        """Core learning from text using Fisher-Rao geometry."""
        tokens = self._tokenize(text)
        if len(tokens) < 2:
            return 0
        
        updates = 0
        
        for i, token in enumerate(tokens):
            context_start = max(0, i - self.context_window)
            context_end = min(len(tokens), i + self.context_window + 1)
            context = [t for j, t in enumerate(tokens[context_start:context_end]) 
                       if j != (i - context_start)]
            
            if token in self.vocab:
                new_coords = self._update_embedding(token, context)
                if new_coords is not None:
                    self.pending_updates.append(TokenUpdate(
                        token=token,
                        new_coords=new_coords,
                        context_tokens=context,
                        phi_contribution=self._compute_phi_contribution(token, context),
                        source=source
                    ))
                    updates += 1
            else:
                new_coords = self._create_embedding(token, context)
                self.vocab[token] = {
                    'coord_id': len(self.vocab) + 256,
                    'vector': new_coords,
                    'phi_score': 0.5,
                    'source_type': source
                }
                updates += 1
        
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            self.pair_counts[pair] += 1
        
        if len(self.pending_updates) >= self.learning_batch_size:
            self._flush_updates()
        
        if time.time() - self.last_checkpoint_time > self.checkpoint_interval:
            self._save_checkpoint()
        
        return updates
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple word-level tokenization."""
        import re
        text = text.lower()
        tokens = re.findall(r'\b[a-zA-Z]{2,}\b', text)
        return tokens
    
    def _update_embedding(self, token: str, context: List[str]) -> Optional[np.ndarray]:
        """
        Update embedding using Fisher-Rao geodesic interpolation.
        
        Moves existing embedding towards context centroid along geodesic.
        """
        if token not in self.vocab:
            return None
        
        current = self.vocab[token]['vector']
        context_embeddings = []
        for ctx_token in context:
            if ctx_token in self.vocab:
                context_embeddings.append(self.vocab[ctx_token]['vector'])
        
        if not context_embeddings:
            return None
        
        context_centroid = self._frechet_mean(context_embeddings)
        
        learning_rate = 0.05
        updated = self._geodesic_interpolate(current, context_centroid, learning_rate)
        
        return updated
    
    def _create_embedding(self, token: str, context: List[str]) -> np.ndarray:
        """
        Create new embedding from context using geodesic fusion.
        """
        context_embeddings = []
        for ctx_token in context:
            if ctx_token in self.vocab:
                context_embeddings.append(self.vocab[ctx_token]['vector'])
        
        if context_embeddings:
            base = self._frechet_mean(context_embeddings)
        else:
            rng = np.random.default_rng(hash(token) % (2**31))
            base = rng.standard_normal(BASIN_DIM)
        
        perturbation = np.random.default_rng(hash(token + str(time.time())) % (2**31)).standard_normal(BASIN_DIM) * 0.1
        embedding = base + perturbation
        
        norm = np.linalg.norm(embedding)
        if norm > 1e-10:
            embedding = embedding / norm
        
        return embedding
    
    def _frechet_mean(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """Compute Fréchet mean (geodesic centroid) of embeddings."""
        if not embeddings:
            return np.zeros(BASIN_DIM)
        
        mean = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(mean)
        if norm > 1e-10:
            mean = mean / norm
        return mean
    
    def _geodesic_interpolate(self, a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        """
        Interpolate along geodesic from a to b.
        
        t=0 returns a, t=1 returns b.
        Uses spherical linear interpolation (SLERP) for unit sphere.
        """
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm < 1e-10 or b_norm < 1e-10:
            return a
        
        a = a / a_norm
        b = b / b_norm
        
        dot = np.clip(np.dot(a, b), -1.0, 1.0)
        omega = np.arccos(dot)
        
        if abs(omega) < 1e-10:
            return a
        
        sin_omega = np.sin(omega)
        result = (np.sin((1-t) * omega) / sin_omega) * a + (np.sin(t * omega) / sin_omega) * b
        
        return result / np.linalg.norm(result)
    
    def _compute_phi_contribution(self, token: str, context: List[str]) -> float:
        """Compute Φ contribution from context coherence."""
        if token not in self.vocab or not context:
            return 0.0
        
        token_vec = self.vocab[token]['vector']
        similarities = []
        for ctx in context:
            if ctx in self.vocab:
                ctx_vec = self.vocab[ctx]['vector']
                sim = np.dot(token_vec, ctx_vec)
                similarities.append(sim)
        
        if not similarities:
            return 0.0
        
        return float(np.mean(similarities))
    
    def _flush_updates(self):
        """Flush pending updates to database."""
        if not self.pending_updates or not self.database_url:
            self.pending_updates = []
            return
        
        try:
            import psycopg2
            from psycopg2.extras import execute_batch
            
            conn = psycopg2.connect(self.database_url)
            updates = []
            for update in self.pending_updates:
                self.vocab[update.token]['vector'] = update.new_coords
                self.vocab[update.token]['phi_score'] = max(
                    self.vocab[update.token].get('phi_score', 0.5),
                    update.phi_contribution
                )
                updates.append({
                    'token': update.token,
                    'embedding': update.new_coords.tolist(),
                    'phi_score': self.vocab[update.token]['phi_score'],
                    'source_type': f'learned_{update.source}'
                })
            
            with conn.cursor() as cur:
                execute_batch(cur, """
                    INSERT INTO tokenizer_vocabulary (token, embedding, phi_score, source_type, updated_at)
                    VALUES (%(token)s, %(embedding)s, %(phi_score)s, %(source_type)s, NOW())
                    ON CONFLICT (token) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        phi_score = GREATEST(tokenizer_vocabulary.phi_score, EXCLUDED.phi_score),
                        source_type = EXCLUDED.source_type,
                        updated_at = NOW()
                """, updates, page_size=100)
            conn.commit()
            conn.close()
            
            logger.info(f"Flushed {len(updates)} embedding updates to database")
        except Exception as e:
            logger.error(f"Failed to flush updates: {e}")
        
        self.pending_updates = []
    
    def _save_checkpoint(self):
        """Save checkpoint with current state."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{len(self.vocab)}_{int(time.time() * 1000)}.json"
        
        vocab_data = {}
        for token, data in self.vocab.items():
            vocab_data[str(data['coord_id'])] = {
                'coord_id': data['coord_id'],
                'vector': data['vector'].tolist(),
                'name': token,
                'scale': data.get('source_type', 'learned')
            }
        
        checkpoint = {
            'vocab_size': len(self.vocab),
            'target_vocab_size': 50000,
            'basin_dim': BASIN_DIM,
            'merge_rules': self.merge_rules,
            'vocab': vocab_data,
            'phi_history': []
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f)
        
        self.last_checkpoint_time = time.time()
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        return checkpoint_path
    
    def get_embedding(self, token: str) -> Optional[np.ndarray]:
        """Get embedding for a token."""
        if token in self.vocab:
            return self.vocab[token]['vector']
        return None
    
    def encode_text(self, text: str) -> Tuple[List[str], np.ndarray]:
        """
        Encode text to tokens and basin coordinates.
        
        Returns (token_list, coords_matrix) where coords_matrix is (n_tokens, 64).
        """
        tokens = self._tokenize(text)
        coords = []
        for token in tokens:
            if token in self.vocab:
                coords.append(self.vocab[token]['vector'])
            else:
                rng = np.random.default_rng(hash(token) % (2**31))
                vec = rng.standard_normal(BASIN_DIM)
                coords.append(vec / np.linalg.norm(vec))
        
        if coords:
            return tokens, np.array(coords)
        return tokens, np.zeros((0, BASIN_DIM))
    
    def find_nearest_tokens(self, coords: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Find k nearest tokens to given coordinates."""
        coords = coords / (np.linalg.norm(coords) + 1e-10)
        
        results = []
        for token, data in self.vocab.items():
            vec = data['vector']
            sim = float(np.dot(coords, vec))
            results.append((token, sim))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def flush_and_save(self):
        """Force flush updates and save checkpoint."""
        self._flush_updates()
        return self._save_checkpoint()


_learner_instance: Optional[ContinuousLearner] = None


def get_learner() -> ContinuousLearner:
    """Get singleton learner instance."""
    global _learner_instance
    if _learner_instance is None:
        _learner_instance = ContinuousLearner()
    return _learner_instance


def learn_from_chat(user_input: str, response: str) -> int:
    """Convenience function to learn from chat."""
    return get_learner().learn_from_chat(user_input, response)


def learn_from_curriculum(text: str, domain: str = 'general') -> int:
    """Convenience function to learn from curriculum."""
    return get_learner().learn_from_curriculum(text, domain)


def learn_from_search(query: str, results: List[str]) -> int:
    """Convenience function to learn from search."""
    return get_learner().learn_from_search(query, results)
