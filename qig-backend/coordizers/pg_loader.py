"""PostgreSQL-backed Coordizer Loader.

Loads vocabulary from PostgreSQL database into QIGCoordizer format,
enabling seamless integration of the migrated 32K checkpoint.
"""

import os
import logging
from typing import Optional, Dict, Any

import numpy as np
import psycopg2
from dotenv import load_dotenv

from .base import FisherCoordizer

load_dotenv()
logger = logging.getLogger(__name__)


class PostgresCoordizer(FisherCoordizer):
    """Fisher-compliant coordizer backed by PostgreSQL.
    
    Loads vocabulary and basin coordinates from the qig_vocabulary table,
    enabling QIG-pure generation with the migrated 32K checkpoint.
    """
    
    def __init__(self, database_url: Optional[str] = None, min_phi: float = 0.0):
        """Initialize coordizer from PostgreSQL.
        
        Args:
            database_url: PostgreSQL connection string (defaults to DATABASE_URL env var)
            min_phi: Minimum phi score for tokens to load (0.0 = all tokens)
        """
        super().__init__()
        
        self.database_url = database_url or os.getenv('DATABASE_URL')
        self.min_phi = min_phi
        self._conn = None
        
        # Load vocabulary from database
        self._load_from_database()
    
    def _get_connection(self):
        """Get or create database connection."""
        if self._conn is None or self._conn.closed:
            if not self.database_url:
                raise ValueError("DATABASE_URL not set")
            self._conn = psycopg2.connect(self.database_url)
        return self._conn
    
    def _load_from_database(self):
        """Load vocabulary and basin coordinates from PostgreSQL."""
        try:
            conn = self._get_connection()
            
            with conn.cursor() as cur:
                # Query vocabulary with optional phi filter
                query = """
                    SELECT token, basin_coords, phi_score, frequency, coherence_rank
                    FROM qig_vocabulary
                    WHERE phi_score >= %s
                    ORDER BY coherence_rank ASC
                """
                cur.execute(query, (self.min_phi,))
                rows = cur.fetchall()
            
            if not rows:
                logger.warning("No tokens found in database")
                return
            
            # Build vocabulary and coordinate mappings
            self.vocab = {}
            self.basin_coords = {}
            self.token_phi = {}
            self.token_frequencies = {}
            self.id_to_token = {}
            self.token_to_id = {}
            
            for idx, (token, basin_coords, phi_score, frequency, rank) in enumerate(rows):
                # Parse vector
                if isinstance(basin_coords, str):
                    coords = np.array([float(x) for x in basin_coords.strip('[]').split(',')])
                elif basin_coords is not None:
                    coords = np.array(basin_coords)
                else:
                    coords = np.random.randn(64)
                    coords = coords / (np.linalg.norm(coords) + 1e-10)
                
                # Ensure unit sphere normalization
                norm = np.linalg.norm(coords)
                if norm > 1e-10:
                    coords = coords / norm
                
                # Store mappings
                self.vocab[token] = idx
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token
                self.basin_coords[token] = coords
                self.token_phi[token] = float(phi_score) if phi_score else 0.5
                self.token_frequencies[token] = frequency or 1
            
            self.vocab_size = len(self.vocab)
            logger.info(f"Loaded {self.vocab_size} tokens from PostgreSQL")
            logger.info(f"Phi range: {min(self.token_phi.values()):.4f} - {max(self.token_phi.values()):.4f}")
            
        except Exception as e:
            logger.error(f"Failed to load from database: {e}")
            raise
    
    def encode(self, text: str) -> np.ndarray:
        """Encode text to basin coordinates.
        
        Uses Fisher-Rao compliant token matching.
        """
        # Simple character/token matching for now
        # TODO: Implement proper BPE-style tokenization with merge rules
        
        tokens = []
        i = 0
        text_lower = text.lower()
        
        while i < len(text):
            # Try to find longest matching token
            best_match = None
            best_len = 0
            
            for token in self.vocab:
                if text_lower[i:].startswith(token.lower()) and len(token) > best_len:
                    best_match = token
                    best_len = len(token)
            
            if best_match:
                tokens.append(best_match)
                i += best_len
            else:
                # Fall back to byte-level token
                byte_token = f"<byte_{ord(text[i]):02x}>"
                if byte_token in self.vocab:
                    tokens.append(byte_token)
                i += 1
        
        if not tokens:
            return np.zeros(64)
        
        # Combine token basin coordinates using phi-weighted average
        total_weight = 0.0
        combined = np.zeros(64)
        
        for token in tokens:
            if token in self.basin_coords:
                phi = self.token_phi.get(token, 0.5)
                weight = phi * self.token_frequencies.get(token, 1)
                combined += weight * self.basin_coords[token]
                total_weight += weight
        
        if total_weight > 1e-10:
            combined = combined / total_weight
        
        # Normalize to unit sphere
        norm = np.linalg.norm(combined)
        if norm > 1e-10:
            combined = combined / norm
        
        return combined
    
    def decode(self, basin: np.ndarray, top_k: int = 5, prefer_words: bool = True) -> list[tuple[str, float]]:
        """Decode basin coordinates to most likely tokens.
        
        Uses Fisher-Rao distance for similarity.
        
        Args:
            basin: Query basin coordinates
            top_k: Number of top tokens to return
            prefer_words: If True, prefer real words over byte tokens
        """
        # Normalize input basin
        norm = np.linalg.norm(basin)
        if norm > 1e-10:
            basin = basin / norm
        
        # Compute Fisher-Rao distance to all tokens
        distances = []
        for token, coords in self.basin_coords.items():
            # Skip pure byte tokens for generation
            if prefer_words and token.startswith('<byte_'):
                continue
            # Skip tokens with too many byte components
            if prefer_words and token.count('<byte_') > 2:
                continue
                
            # Fisher-Rao distance: arccos(dot product) for unit vectors
            dot = np.clip(np.dot(basin, coords), -1.0, 1.0)
            dist = np.arccos(dot)
            distances.append((token, dist))
        
        # Sort by distance (ascending)
        distances.sort(key=lambda x: x[1])
        
        # Return top-k with similarity scores (1 - normalized_distance)
        results = []
        for token, dist in distances[:top_k]:
            similarity = 1.0 - (dist / np.pi)  # Normalize to [0, 1]
            results.append((token, similarity))
        
        return results
    
    def find_similar_tokens(self, query_token: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Find tokens similar to the query using Fisher-Rao distance."""
        if query_token not in self.basin_coords:
            logger.warning(f"Token '{query_token}' not in vocabulary")
            return []
        
        query_basin = self.basin_coords[query_token]
        return self.decode(query_basin, top_k=top_k + 1)[1:]  # Exclude self
    
    def get_high_phi_tokens(self, min_phi: float = 0.7, limit: int = 100) -> list[tuple[str, float]]:
        """Get tokens with high phi (integration) scores."""
        high_phi = [(token, phi) for token, phi in self.token_phi.items() if phi >= min_phi]
        high_phi.sort(key=lambda x: x[1], reverse=True)
        return high_phi[:limit]
    
    def refresh(self):
        """Reload vocabulary from database."""
        self._load_from_database()
    
    def close(self):
        """Close database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()
            self._conn = None


def create_coordizer_from_pg(min_phi: float = 0.0) -> PostgresCoordizer:
    """Factory function to create a PostgreSQL-backed coordizer.
    
    Args:
        min_phi: Minimum phi score for tokens (0.0 = all tokens)
    
    Returns:
        PostgresCoordizer instance loaded from database
    """
    return PostgresCoordizer(min_phi=min_phi)
