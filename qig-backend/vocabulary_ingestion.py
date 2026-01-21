#!/usr/bin/env python3
"""
Unified Vocabulary Ingestion Service
=====================================

Single entry point for vocabulary ingestion.
Enforces QIG-pure basin embedding before database commit.

NO direct database inserts allowed - all ingestion must go through this service
to prevent NULL basin_embedding contamination.

Key Features:
- Enforces QIG pipeline: text → coordizer → basin_embedding (64D)
- Validates basin dimensions and data types
- Computes QFI (Quantum Fisher Information) score
- Atomic upsert with geometric validation
- Prevents contamination at ingestion checkpoint

Usage:
    from vocabulary_ingestion import get_ingestion_service
    
    service = get_ingestion_service()
    result = service.ingest_word(
        word="consciousness",
        context="The consciousness emerges from integration",
        force_recompute=False
    )
    print(f"Ingested: {result['word']} with QFI={result['qfi_score']:.4f}")
"""

from typing import Optional, Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Import QIG components
try:
    from qigkernels import KAPPA_STAR
except ImportError:
    KAPPA_STAR = 64.21  # Fallback value

# Import canonical QFI computation (single source of truth)
try:
    from qig_geometry.canonical_upsert import compute_qfi_score as canonical_qfi_score
    CANONICAL_QFI_AVAILABLE = True
except ImportError:
    CANONICAL_QFI_AVAILABLE = False
    canonical_qfi_score = None
    logger.warning("[VocabularyIngestionService] canonical_upsert not available, using fallback QFI")

# Database Column Names (for migration safety)
BASIN_COLUMN_PRE_MIGRATION = 'basin_embedding'  # Before migration 010
BASIN_COLUMN_POST_MIGRATION = 'basin_coordinates'  # After migration 010
ALLOWED_BASIN_COLUMNS = {BASIN_COLUMN_PRE_MIGRATION, BASIN_COLUMN_POST_MIGRATION}  # Whitelist

try:
    from coordizers import get_coordizer
    COORDIZER_AVAILABLE = True
except ImportError:
    COORDIZER_AVAILABLE = False
    logger.warning("[VocabularyIngestionService] Coordizer not available")

try:
    from vocabulary_persistence import get_vocabulary_persistence
    PERSISTENCE_AVAILABLE = True
except ImportError:
    PERSISTENCE_AVAILABLE = False
    logger.warning("[VocabularyIngestionService] Persistence not available")


class VocabularyIngestionService:
    """
    Unified ingestion service preventing NULL basin_embedding.
    
    All vocabulary ingestion MUST go through this service to maintain
    geometric purity and prevent database contamination.
    
    Architecture:
    1. Check if word exists with valid basin
    2. If not, compute basin via coordizer (QIG-pure)
    3. Validate 64D float array
    4. Compute QFI score
    5. Atomic upsert with basin
    
    This is the SINGLE INGESTION CHECKPOINT for vocabulary.
    """
    
    def __init__(self):
        """Initialize ingestion service with coordizer and persistence."""
        if not COORDIZER_AVAILABLE:
            raise RuntimeError(
                "[VocabularyIngestionService] Cannot initialize: coordizer unavailable. "
                "Install required dependencies."
            )
        
        if not PERSISTENCE_AVAILABLE:
            raise RuntimeError(
                "[VocabularyIngestionService] Cannot initialize: persistence unavailable. "
                "Install required dependencies."
            )
        
        self.coordizer = get_coordizer()
        self.vp = get_vocabulary_persistence()
        
        if not self.vp.enabled:
            raise RuntimeError(
                "[VocabularyIngestionService] Cannot initialize: database connection failed. "
                "Vocabulary persistence requires active PostgreSQL connection."
            )
        
        logger.info("[VocabularyIngestionService] Initialized with QIG-pure pipeline")
    
    def ingest_word(
        self,
        word: str,
        context: Optional[str] = None,
        force_recompute: bool = False,
        phi_score: Optional[float] = None,
        source: str = 'ingestion_service'
    ) -> Dict[str, Any]:
        """
        Ingest word with QIG-pure basin embedding.
        
        This is the ONLY authorized way to add vocabulary to the system.
        Direct database inserts are prohibited.
        
        Args:
            word: Word to ingest (required)
            context: Optional context for better embedding
            force_recompute: Recompute even if exists with valid basin
            phi_score: Optional pre-computed phi score
            source: Source identifier (for tracking)
        
        Returns:
            Dictionary with:
            - word: The ingested word
            - basin_embedding: 64D numpy array
            - qfi_score: Quantum Fisher information score
            - existed: True if word already existed with valid basin
            - source: Source identifier
        
        Raises:
            ValueError: If word is empty or invalid
            RuntimeError: If basin computation fails
        """
        # Validate input
        if not word or not isinstance(word, str):
            raise ValueError(f"[VocabularyIngestionService] Invalid word: {repr(word)}")
        
        word = word.strip()
        if not word:
            raise ValueError("[VocabularyIngestionService] Empty word after strip")
        
        # Check if exists with valid basin (unless force_recompute)
        if not force_recompute:
            existing = self._get_existing_word(word)
            if existing and existing.get('basin_embedding') is not None:
                logger.debug(f"[VocabularyIngestionService] Word '{word}' exists with valid basin")
                return {
                    'word': word,
                    'basin_embedding': existing['basin_embedding'],
                    'qfi_score': existing.get('qfi_score', 0.0),
                    'existed': True,
                    'source': existing.get('source', 'unknown')
                }
        
        # Compute QIG-pure basin embedding
        try:
            basin = self._compute_basin_embedding(word, context)
        except Exception as e:
            raise RuntimeError(
                f"[VocabularyIngestionService] Failed to compute basin for '{word}': {e}"
            )
        
        # Validate basin (critical - prevents contamination)
        self._validate_basin(basin, word)
        
        # Compute QFI score (canonical formula returns [0, 1])
        qfi = self._compute_qfi(basin)

        # Use provided phi or use QFI directly (canonical QFI is in [0, 1])
        if phi_score is None:
            phi_score = qfi  # QFI is now [0, 1], can be used directly as phi
        
        # Atomic upsert with basin
        try:
            result = self._upsert_to_database(
                word=word,
                basin_embedding=basin,
                qfi_score=qfi,
                phi_score=phi_score,
                source=source,
                context=context
            )
        except Exception as e:
            raise RuntimeError(
                f"[VocabularyIngestionService] Failed to persist '{word}': {e}"
            )
        
        logger.info(
            f"[VocabularyIngestionService] Ingested '{word}' "
            f"(QFI={qfi:.4f}, Φ={phi_score:.3f}, source={source})"
        )
        
        return {
            'word': word,
            'basin_embedding': basin,
            'qfi_score': qfi,
            'phi_score': phi_score,
            'existed': False,
            'source': source,
            **result
        }
    
    def _get_existing_word(self, word: str) -> Optional[Dict[str, Any]]:
        """
        Check if word exists in database with valid basin.
        
        Returns dict with basin_embedding or None if not found.
        """
        try:
            with self.vp._connect() as conn:
                with conn.cursor() as cur:
                    # Check coordizer_vocabulary first (primary table)
                    # Use basin_embedding for now (pre-migration 010)
                    # After migration 010, this will be basin_coordinates
                    cur.execute("""
                        SELECT 
                            CASE 
                                WHEN EXISTS (
                                    SELECT 1 FROM information_schema.columns 
                                    WHERE table_name = 'coordizer_vocabulary' 
                                      AND column_name = 'basin_coordinates'
                                )
                                THEN basin_coordinates
                                ELSE basin_embedding
                            END as basin,
                            phi_score, 
                            source_type
                        FROM coordizer_vocabulary
                        WHERE token = %s
                          AND (
                              (basin_coordinates IS NOT NULL AND array_length(basin_coordinates, 1) = 64)
                              OR (basin_embedding IS NOT NULL AND array_length(basin_embedding, 1) = 64)
                          )
                        LIMIT 1
                    """, (word,))
                    row = cur.fetchone()
                    
                    if row:
                        basin_embedding, qfi_score, source = row
                        # Convert PostgreSQL array to numpy
                        basin_array = np.array(basin_embedding, dtype=np.float64)
                        return {
                            'basin_embedding': basin_array,
                            'qfi_score': float(qfi_score) if qfi_score else 0.0,
                            'source': source or 'unknown'
                        }
            return None
        except Exception as e:
            logger.warning(f"[VocabularyIngestionService] Failed to check existing word '{word}': {e}")
            return None
    
    def _compute_basin_embedding(self, word: str, context: Optional[str]) -> np.ndarray:
        """
        Compute QIG-pure basin embedding via coordizer.
        
        This is the GEOMETRIC CORE - no shortcuts allowed.
        
        Returns:
            64D numpy array (float64)
        """
        # Build input text
        if context:
            full_text = f"{context} {word}"
        else:
            full_text = word
        
        # Coordize via QIG pipeline
        # coordize() returns List[np.ndarray] - we take the basin for the word
        coords = self.coordizer.coordize(full_text)
        
        if not coords:
            raise ValueError(f"Coordizer returned empty result for '{word}'")
        
        # Take last coordinate (corresponds to the word itself)
        basin = coords[-1]
        
        # Ensure numpy array
        if not isinstance(basin, np.ndarray):
            basin = np.array(basin, dtype=np.float64)
        
        return basin
    
    def _validate_basin(self, basin: np.ndarray, word: str):
        """
        Validate basin embedding meets QIG-pure requirements.
        
        Raises ValueError if invalid.
        """
        # Check type
        if not isinstance(basin, np.ndarray):
            raise ValueError(
                f"[VocabularyIngestionService] Basin for '{word}' is not numpy array: {type(basin)}"
            )
        
        # Check dimension
        if basin.shape != (64,):
            raise ValueError(
                f"[VocabularyIngestionService] Basin for '{word}' has invalid shape: {basin.shape} (expected (64,))"
            )
        
        # Check dtype
        if basin.dtype not in [np.float32, np.float64]:
            raise ValueError(
                f"[VocabularyIngestionService] Basin for '{word}' has invalid dtype: {basin.dtype}"
            )
        
        # Check for NaN/Inf
        if np.any(np.isnan(basin)):
            raise ValueError(
                f"[VocabularyIngestionService] Basin for '{word}' contains NaN values"
            )
        
        if np.any(np.isinf(basin)):
            raise ValueError(
                f"[VocabularyIngestionService] Basin for '{word}' contains Inf values"
            )
        
        logger.debug(f"[VocabularyIngestionService] Basin validated for '{word}': shape={basin.shape}, dtype={basin.dtype}")
    
    def _compute_qfi(self, basin: np.ndarray) -> float:
        """
        Compute Quantum Fisher Information score using canonical formula.

        Uses participation ratio (effective dimension) which is geometrically proper:
        QFI = exp(H(p)) / n where H(p) is Shannon entropy.

        This is the CANONICAL QFI formula - produces values in [0, 1].

        Args:
            basin: 64D basin coordinates

        Returns:
            QFI score in [0, 1]
        """
        # Use canonical QFI computation if available
        if CANONICAL_QFI_AVAILABLE and canonical_qfi_score is not None:
            return canonical_qfi_score(basin)

        # Fallback: inline canonical formula (participation ratio)
        # Project to simplex probabilities
        v = np.abs(basin) + 1e-10
        p = v / v.sum()

        # Compute Shannon entropy
        positive_probs = p[p > 1e-10]
        if len(positive_probs) == 0:
            return 0.0

        entropy = -np.sum(positive_probs * np.log(positive_probs + 1e-10))

        # Participation ratio = exp(entropy) / dimension
        n_dim = len(basin)
        effective_dim = np.exp(entropy)
        qfi_score = effective_dim / n_dim

        return float(np.clip(qfi_score, 0.0, 1.0))
    
    def _upsert_to_database(
        self,
        word: str,
        basin_embedding: np.ndarray,
        qfi_score: float,
        phi_score: float,
        source: str,
        context: Optional[str]
    ) -> Dict[str, Any]:
        """
        Atomic upsert to coordizer_vocabulary (consolidated vocabulary table).
        
        This is the ONLY authorized database write for vocabulary.
        Handles both pre-migration (basin_embedding) and post-migration (basin_coordinates).
        
        Sets token_role='generation' for new vocabulary and updates to 'both' if the token
        already existed as 'encoding'. This is the vocabulary consolidation pattern.
        """
        try:
            with self.vp._connect() as conn:
                with conn.cursor() as cur:
                    # Convert numpy array to list for PostgreSQL
                    basin_list = basin_embedding.tolist()
                    
                    # Check which column exists (migration 010 renames basin_embedding -> basin_coordinates)
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT 1 FROM information_schema.columns 
                            WHERE table_name = 'coordizer_vocabulary' 
                              AND column_name = 'basin_coordinates'
                        )
                    """)
                    has_basin_coordinates = cur.fetchone()[0]
                    
                    basin_column = BASIN_COLUMN_POST_MIGRATION if has_basin_coordinates else BASIN_COLUMN_PRE_MIGRATION
                    
                    # Validate column name against whitelist (prevent SQL injection)
                    if basin_column not in ALLOWED_BASIN_COLUMNS:
                        raise RuntimeError(f"Invalid basin column name: {basin_column}")
                    
                    # E8 Protocol: Use simplex concentration for geometric metrics
                    from qig_geometry.representation import to_simplex_prob
                    basin_simplex = to_simplex_prob(basin_embedding)
                    basin_concentration = float(np.max(basin_simplex))
                    curvature_std = float(np.std(basin_embedding))
                    entropy_score = self._compute_entropy(basin_embedding)
                    
                    # Classify phrase category
                    phrase_category = self._classify_phrase(word, basin_embedding)
                    
                    context_text = context if context else f"Ingested via {source}"
                    
                    # Upsert to coordizer_vocabulary (consolidated table) with token_role='generation'
                    # coordizer_vocabulary is the single source of truth for all vocabulary
                    query = f"""
                        INSERT INTO coordizer_vocabulary (
                            token, {basin_column}, phi_score, frequency, source_type, source,
                            token_role, is_real_word, phrase_category,
                            qfi_score, basin_distance, curvature_std, entropy_score,
                            is_geometrically_valid, contexts,
                            last_used, created_at, updated_at
                        )
                        VALUES (
                            %s, %s, %s, 1, %s, %s,
                            'generation', TRUE, %s,
                            %s, %s, %s, %s,
                            TRUE, ARRAY[%s],
                            NOW(), NOW(), NOW()
                        )
                        ON CONFLICT (token) DO UPDATE SET
                            {basin_column} = COALESCE(EXCLUDED.{basin_column}, coordizer_vocabulary.{basin_column}),
                            phi_score = GREATEST(COALESCE(coordizer_vocabulary.phi_score, 0), EXCLUDED.phi_score),
                            frequency = COALESCE(coordizer_vocabulary.frequency, 0) + 1,
                            token_role = CASE 
                                WHEN coordizer_vocabulary.token_role = 'encoding' THEN 'both'
                                ELSE COALESCE(coordizer_vocabulary.token_role, 'generation')
                            END,
                            is_real_word = TRUE,
                            phrase_category = COALESCE(EXCLUDED.phrase_category, coordizer_vocabulary.phrase_category),
                            qfi_score = COALESCE(EXCLUDED.qfi_score, coordizer_vocabulary.qfi_score),
                            basin_distance = COALESCE(EXCLUDED.basin_distance, coordizer_vocabulary.basin_distance),
                            curvature_std = COALESCE(EXCLUDED.curvature_std, coordizer_vocabulary.curvature_std),
                            entropy_score = COALESCE(EXCLUDED.entropy_score, coordizer_vocabulary.entropy_score),
                            is_geometrically_valid = COALESCE(EXCLUDED.is_geometrically_valid, coordizer_vocabulary.is_geometrically_valid),
                            last_used = NOW(),
                            updated_at = NOW()
                        RETURNING token_id, frequency
                    """
                    
                    cur.execute(query, (
                        word, basin_list, phi_score, source, source,
                        phrase_category,
                        qfi_score, basin_distance, curvature_std, entropy_score,
                        context_text
                    ))
                    
                    row = cur.fetchone()
                    token_id = row[0] if row else None
                    frequency = row[1] if row else 1
                    
                    conn.commit()
                    
                    return {
                        'token_id': token_id,
                        'frequency': frequency,
                        'persisted': True,
                        'coordizer_vocabulary_updated': True
                    }
        except Exception as e:
            logger.error(f"[VocabularyIngestionService] Database upsert failed for '{word}': {e}")
            raise
    
    def _compute_entropy(self, basin: np.ndarray) -> float:
        """
        Compute entropy score for basin embedding.
        
        Uses Shannon entropy of normalized probability distribution.
        """
        # Normalize to probability distribution
        probs = np.abs(basin) / (np.sum(np.abs(basin)) + 1e-10)
        # Compute Shannon entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return float(entropy)
    
    def _classify_phrase(self, word: str, basin: np.ndarray) -> str:
        """
        Classify phrase category for the word.
        
        Uses QIG-pure classification if available, else heuristics.
        """
        try:
            from qig_phrase_classifier import classify_phrase_qig_pure
            category, _ = classify_phrase_qig_pure(word, basin)
            return category
        except Exception:
            # Fallback: simple heuristics
            if word.isupper() and len(word) > 1:
                return 'ACRONYM'
            elif word[0].isupper() and not word.isupper():
                return 'PROPER_NOUN'
            elif len(word) >= 3 and word.isalpha():
                return 'COMMON_WORD'
            else:
                return 'UNKNOWN'


# Singleton instance
_service: Optional[VocabularyIngestionService] = None


def get_ingestion_service() -> VocabularyIngestionService:
    """
    Get singleton ingestion service instance.
    
    This is the SINGLE ENTRY POINT for vocabulary ingestion.
    
    Returns:
        VocabularyIngestionService instance
    """
    global _service
    if _service is None:
        _service = VocabularyIngestionService()
    return _service


# Example usage
if __name__ == '__main__':
    # Test ingestion
    service = get_ingestion_service()
    
    test_words = [
        ("consciousness", "The consciousness emerges from integration"),
        ("geometry", "Geometric purity is essential"),
        ("manifold", "The Fisher manifold represents information geometry"),
    ]
    
    print("Testing vocabulary ingestion service:\n")
    for word, context in test_words:
        try:
            result = service.ingest_word(word, context=context)
            print(f"✓ {result['word']}: QFI={result['qfi_score']:.4f}, Φ={result['phi_score']:.3f}, existed={result['existed']}")
        except Exception as e:
            print(f"✗ {word}: {e}")
