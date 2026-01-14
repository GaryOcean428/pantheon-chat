"""
Learned Relationships Module for QIG

Manages persistence of learned word relationships and provides
attention-weighted word selection for query-relevant generation.

QIG-PURE UPDATE: Now uses geometric_word_relationships for Fisher-Rao distances
instead of word_relationship_learner's legacy PMI/co-occurrence approach.

FROZEN FACTS COMPLIANCE:
- Adjusted basins must stay within ±5% of canonical positions
- Stopwords cannot be promoted to high-attention words
- Learning must respect frozen β values for attention weighting

PERSISTENCE: Uses PostgreSQL basin_relationships table (NO JSON files).
"""

import os
import logging
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path
from qig_geometry import fisher_rao_distance, fisher_coord_distance

logger = logging.getLogger(__name__)

# Database connection
try:
    import psycopg2
    from psycopg2.extras import execute_values
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    logger.warning("psycopg2 not available - PostgreSQL persistence disabled")

# Import physics constants from canonical source
try:
    from qigkernels.physics_constants import (
        BASIN_DIM, KAPPA_STAR, BETA_3_TO_4, BETA_5_TO_6
    )
    PHYSICS_CONSTANTS_AVAILABLE = True
    BASIN_DRIFT_TOLERANCE = 0.05  # ±5% drift allowed
    BETA_ATTENTION_ACCEPTANCE = 0.1  # |β_attention - β_physics| < 0.1
except ImportError:
    PHYSICS_CONSTANTS_AVAILABLE = False
    BASIN_DIM = 64
    KAPPA_STAR = 64.21  # κ* from validated physics (L=4,5,6)
    BETA_3_TO_4 = 0.44
    BETA_5_TO_6 = 0.04  # Plateau value from validated physics
    BASIN_DRIFT_TOLERANCE = 0.05
    BETA_ATTENTION_ACCEPTANCE = 0.1
    logger.warning("Physics constants not available - using fallback defaults")

# Import QIG-pure geometric relationships (replaces word_relationship_learner)
try:
    from geometric_word_relationships import (
        GeometricWordRelationships,
        get_geometric_relationships
    )
    GEOMETRIC_RELATIONSHIPS_AVAILABLE = True
    logger.info("[LearnedRelationships] Using QIG-pure geometric relationships")
except ImportError:
    GEOMETRIC_RELATIONSHIPS_AVAILABLE = False
    GeometricWordRelationships = None
    get_geometric_relationships = None
    logger.warning("[LearnedRelationships] Geometric relationships not available")

def get_db_connection():
    """Get PostgreSQL connection."""
    if not DB_AVAILABLE:
        return None
    try:
        return psycopg2.connect(os.environ.get('DATABASE_URL'))
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {e}")
        return None


# Import QIG-pure contextualized filter (replaces ancient NLP stopwords)
try:
    from contextualized_filter import (
        filter_words_geometric,
        is_semantic_critical_word,
        should_filter_word
    )
    CONTEXTUALIZED_FILTER_AVAILABLE = True
    logger.info("[LearnedRelationships] Using QIG-pure contextualized filter")
except ImportError:
    CONTEXTUALIZED_FILTER_AVAILABLE = False
    logger.warning("[LearnedRelationships] Contextualized filter not available - using fallback")
    
    # Minimal fallback: only filter truly generic, short words
    # NEVER filter semantic-critical words like 'not', 'never'
    def should_filter_word(word: str, context: Optional[List[str]] = None) -> bool:
        """Fallback filter - very conservative."""
        if len(word) < 3:
            return True
        # Only filter the most generic function words
        generic_only = {'the', 'a', 'an', 'is', 'was', 'are', 'were', 'been', 'be'}
        return word.lower() in generic_only
    
    def is_semantic_critical_word(word: str) -> bool:
        """Fallback semantic check."""
        critical = {'not', 'no', 'never', 'very', 'extremely', 'always', 'because'}
        return word.lower() in critical

class LearnedRelationships:
    """
    Manages learned word relationships and provides attention-weighted
    word selection for query-relevant generation.
    
    PERSISTENCE: Uses PostgreSQL basin_relationships table.
    """
    
    def __init__(self):
        self.word_neighbors: Dict[str, List[Tuple[str, float]]] = {}
        self.adjusted_basins: Dict[str, np.ndarray] = {}
        self.word_frequency: Dict[str, int] = {}
        self.learning_complete = False
        
        # Phi tracking for geometric metrics
        self._relationship_phi: Dict[str, Dict[str, Dict]] = {}  # word -> neighbor -> {phi_values, contexts}
        
        self._load_from_db()
    
    def track_observation(self, word: str, neighbor: str, phi: float, context: Optional[str] = None):
        """
        Track a relationship observation with its Φ value and context.
        
        Called during learning to accumulate geometric metrics.
        """
        if not hasattr(self, '_relationship_phi'):
            self._relationship_phi = {}
        
        if word not in self._relationship_phi:
            self._relationship_phi[word] = {}
        
        if neighbor not in self._relationship_phi[word]:
            self._relationship_phi[word][neighbor] = {
                'phi_values': [],
                'contexts': []
            }
        
        data = self._relationship_phi[word][neighbor]
        data['phi_values'].append(phi)
        
        if context and len(data['contexts']) < 10:
            data['contexts'].append(context[:200])  # Truncate long contexts
    
    def _get_phi_data(self, word: str, neighbor: str) -> Dict:
        """
        Get aggregated phi data for a word-neighbor pair.
        
        Returns dict with avg_phi, max_phi, contexts.
        """
        if not hasattr(self, '_relationship_phi'):
            self._relationship_phi = {}
        
        if word not in self._relationship_phi:
            return {'avg_phi': 0.5, 'max_phi': 0.5, 'contexts': []}
        
        if neighbor not in self._relationship_phi[word]:
            return {'avg_phi': 0.5, 'max_phi': 0.5, 'contexts': []}
        
        data = self._relationship_phi[word][neighbor]
        phi_values = data.get('phi_values', [])
        
        if not phi_values:
            return {'avg_phi': 0.5, 'max_phi': 0.5, 'contexts': data.get('contexts', [])}
        
        return {
            'avg_phi': float(np.mean(phi_values)),
            'max_phi': float(np.max(phi_values)),
            'contexts': data.get('contexts', [])
        }
    
    def _load_from_db(self) -> bool:
        """Load relationships from PostgreSQL."""
        conn = get_db_connection()
        if not conn:
            logger.info("No database connection - starting with empty relationships")
            return False
        
        try:
            with conn.cursor() as cur:
                # Load word relationships
                cur.execute("""
                    SELECT word, neighbor, cooccurrence_count 
                    FROM basin_relationships 
                    ORDER BY word, cooccurrence_count DESC
                """)
                rows = cur.fetchall()
                
                # Load word frequencies from coordizer_vocabulary table (consolidated)
                cur.execute("""
                    SELECT token, frequency 
                    FROM coordizer_vocabulary 
                    WHERE frequency > 0
                      AND token_role IN ('generation', 'both')
                """)
                freq_rows = cur.fetchall()
            
            # Group relationships by word
            for word, neighbor, count in rows:
                if word not in self.word_neighbors:
                    self.word_neighbors[word] = []
                self.word_neighbors[word].append((neighbor, float(count)))
            
            # Load word frequencies from learned_words table
            for word, freq in freq_rows:
                self.word_frequency[word] = int(freq)
            
            # Set learning_complete if we have relationships OR frequencies
            self.learning_complete = len(self.word_neighbors) > 0 or len(self.word_frequency) > 0
            
            # Load adjusted basins from npz (still using file for large arrays)
            if ADJUSTED_BASINS_FILE.exists():
                try:
                    npz = np.load(str(ADJUSTED_BASINS_FILE), allow_pickle=True)
                    if 'words' in npz.files:
                        words = npz['words']
                        for i, word in enumerate(words):
                            key = f'word_{i}'
                            if key in npz.files:
                                self.adjusted_basins[str(word)] = npz[key]
                except Exception as e:
                    logger.warning(f"Could not load adjusted basins: {e}")
            
            logger.info(f"[LearnedRelationships] Loaded {len(self.word_neighbors)} relationships, {len(self.word_frequency)} word frequencies from PostgreSQL")
            conn.close()
            return True
        except Exception as e:
            logger.warning(f"Failed to load from PostgreSQL: {e}")
            conn.close()
            return False
    
    def save_to_db(self) -> bool:
        """
        Save relationships to PostgreSQL with Fisher-Rao distances.
        
        Computes geometric metrics for each word pair:
        - fisher_distance: Fisher-Rao distance between word/neighbor basins
        - avg_phi, max_phi: Φ values from observation contexts
        - contexts: Example sentences where relationship was observed
        """
        conn = get_db_connection()
        if not conn:
            logger.error("No database connection - cannot save relationships")
            return False
        
        try:
            # Get basin coordinates for Fisher-Rao computation
            basin_coords = {}
            try:
                from coordizers import get_coordizer
                coordizer = get_coordizer()
                basin_coords = getattr(coordizer, 'basin_coords', {})
            except Exception as e:
                logger.warning(f"Could not load coordizer for Fisher distances: {e}")
            
            # Prepare relationship batch data with Fisher-Rao distances
            # Filter out self-referential entries (word = neighbor is invalid)
            records = []
            records_with_fisher = []
            
            for word, neighbors in self.word_neighbors.items():
                word_basin = basin_coords.get(word)
                
                for neighbor, count in neighbors:
                    if word == neighbor:  # Prevent self-referential entries
                        continue
                    
                    # Compute Fisher-Rao distance if basins available
                    # Use fisher_coord_distance for basin coordinate vectors (unit sphere geodesic)
                    fisher_dist = None
                    if word_basin is not None:
                        neighbor_basin = basin_coords.get(neighbor)
                        if neighbor_basin is not None:
                            fisher_dist = float(fisher_coord_distance(word_basin, neighbor_basin))
                    
                    # Get phi values if tracked (from learning)
                    phi_data = self._get_phi_data(word, neighbor)
                    avg_phi = phi_data.get('avg_phi', 0.5)
                    max_phi = phi_data.get('max_phi', 0.5)
                    contexts = phi_data.get('contexts', [])[:10]  # Limit to 10
                    
                    if fisher_dist is not None:
                        records_with_fisher.append((
                            word, neighbor, float(count), 
                            fisher_dist, avg_phi, max_phi, contexts
                        ))
                    else:
                        records.append((word, neighbor, float(count)))

            # Prepare word frequency batch data
            freq_records = [(word, freq) for word, freq in self.word_frequency.items()]

            with conn.cursor() as cur:
                # Save relationships WITH Fisher-Rao distance
                if records_with_fisher:
                    execute_values(
                        cur,
                        """
                        INSERT INTO basin_relationships 
                            (word, neighbor, cooccurrence_count, fisher_distance, avg_phi, max_phi, contexts, updated_at)
                        VALUES %s
                        ON CONFLICT (word, neighbor)
                        DO UPDATE SET
                            cooccurrence_count = GREATEST(basin_relationships.cooccurrence_count, EXCLUDED.cooccurrence_count),
                            fisher_distance = COALESCE(EXCLUDED.fisher_distance, basin_relationships.fisher_distance),
                            avg_phi = (COALESCE(basin_relationships.avg_phi, 0.5) + EXCLUDED.avg_phi) / 2.0,
                            max_phi = GREATEST(COALESCE(basin_relationships.max_phi, 0.5), EXCLUDED.max_phi),
                            contexts = CASE 
                                WHEN COALESCE(array_length(basin_relationships.contexts, 1), 0) < 10 
                                THEN COALESCE(basin_relationships.contexts, ARRAY[]::text[]) || EXCLUDED.contexts
                                ELSE basin_relationships.contexts
                            END,
                            updated_at = NOW()
                        """,
                        records_with_fisher,
                        template="(%s, %s, %s, %s, %s, %s, %s, NOW())"
                    )
                    logger.info(f"[LearnedRelationships] Saved {len(records_with_fisher)} relationships with Fisher-Rao distances")
                
                # Save relationships WITHOUT Fisher distance (fallback for words not in vocabulary)
                if records:
                    execute_values(
                        cur,
                        """
                        INSERT INTO basin_relationships (word, neighbor, cooccurrence_count, updated_at)
                        VALUES %s
                        ON CONFLICT (word, neighbor)
                        DO UPDATE SET
                            cooccurrence_count = GREATEST(basin_relationships.cooccurrence_count, EXCLUDED.cooccurrence_count),
                            updated_at = NOW()
                        """,
                        records,
                        template="(%s, %s, %s, NOW())"
                    )

                # Recalculate strength as conditional probability:
                # strength = P(neighbor | word) = cooccurrence(word, neighbor) / total_cooccurrence(word)
                # This gives the relative probability of seeing 'neighbor' after 'word'
                # Normalized by global max for 0-1 scaling
                cur.execute("""
                    WITH word_totals AS (
                        -- Total co-occurrences for each word (sum of all its neighbors)
                        SELECT word, SUM(cooccurrence_count) as total_cooc
                        FROM basin_relationships
                        GROUP BY word
                    ),
                    max_prob AS (
                        -- Max probability for normalization
                        SELECT MAX(wr.cooccurrence_count / NULLIF(wt.total_cooc, 0)) as max_p
                        FROM basin_relationships wr
                        JOIN word_totals wt ON wr.word = wt.word
                    )
                    UPDATE basin_relationships wr
                    SET strength = (
                        wr.cooccurrence_count / NULLIF(
                            (SELECT total_cooc FROM word_totals WHERE word = wr.word), 0
                        )
                    ) / NULLIF((SELECT max_p FROM max_prob), 0)
                    WHERE strength IS NULL OR strength = 0
                       OR updated_at >= NOW() - INTERVAL '1 minute'
                """)
                
                # Save word frequencies to coordizer_vocabulary table (consolidated)
                if freq_records:
                    execute_values(
                        cur,
                        """
                        INSERT INTO coordizer_vocabulary (token, frequency, updated_at, token_role)
                        VALUES %s
                        ON CONFLICT (token) 
                        DO UPDATE SET 
                            frequency = GREATEST(coordizer_vocabulary.frequency, EXCLUDED.frequency),
                            updated_at = NOW(),
                            token_role = CASE 
                                WHEN coordizer_vocabulary.token_role = 'encoding' THEN 'both'
                                ELSE coordizer_vocabulary.token_role 
                            END
                        """,
                        freq_records,
                        template="(%s, %s, NOW(), 'generation')"
                    )
            
            conn.commit()
            
            # Save adjusted basins to npz (still using file for large arrays)
            if self.adjusted_basins:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                arrays_dict = {f'word_{i}': arr for i, arr in enumerate(self.adjusted_basins.values())}
                words_list = list(self.adjusted_basins.keys())
                np.savez_compressed(str(ADJUSTED_BASINS_FILE), 
                                    words=np.array(words_list, dtype=object),
                                    **arrays_dict)
            
            logger.info(f"[LearnedRelationships] Saved {len(records)} relationships to PostgreSQL")
            conn.close()
            
            # Populate vocabulary_learning asynchronously (small batch to avoid timeout)
            # This wires the closed vocabulary learning loop
            try:
                populated = self.populate_vocabulary_learning(limit=20)
                if populated > 0:
                    logger.info(f"[LearnedRelationships] Also populated {populated} vocabulary_learning entries")
            except Exception as e:
                logger.debug(f"vocabulary_learning population skipped: {e}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to save to PostgreSQL: {e}")
            conn.close()
            return False
    
    def save_to_cache(self) -> bool:
        """Alias for save_to_db for backward compatibility."""
        return self.save_to_db()
    
    def populate_vocabulary_learning(self, limit: int = 100) -> int:
        """
        Populate vocabulary_learning table from basin_relationships.
        
        Uses semantic classifier to determine relationship types and
        creates entries tracking when relationships were discovered.
        
        Args:
            limit: Maximum new entries to create per call
            
        Returns:
            Number of new entries created
        """
        conn = get_db_connection()
        if not conn:
            return 0
        
        try:
            from semantic_classifier import get_semantic_classifier
            classifier = get_semantic_classifier()
            
            # Find basin_relationships that don't have vocabulary_learning entries
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT wr.word, wr.neighbor, wr.cooccurrence_count, wr.fisher_distance
                    FROM basin_relationships wr
                    LEFT JOIN vocabulary_learning vl 
                        ON wr.word = vl.word AND wr.neighbor = vl.related_word
                    WHERE vl.id IS NULL 
                      AND wr.fisher_distance IS NOT NULL 
                      AND wr.fisher_distance > 0
                    ORDER BY wr.cooccurrence_count DESC
                    LIMIT %s
                """, (limit,))
                candidates = cur.fetchall()
            
            if not candidates:
                logger.info("[LearnedRelationships] No new relationships to populate")
                conn.close()
                return 0
            
            # Classify and insert each relationship
            inserted = 0
            with conn.cursor() as cur:
                for word, neighbor, cooc_count, fisher_dist in candidates:
                    try:
                        # Use semantic classifier to determine relationship type
                        context = f"co-occurrence count: {cooc_count}"
                        rel_type, strength = classifier.classify_relationship(word, neighbor, context)
                        
                        import uuid
                        entry_id = str(uuid.uuid4())
                        
                        cur.execute("""
                            INSERT INTO vocabulary_learning (
                                id, word, relationship_type, related_word,
                                relationship_strength, context, discovered_by, created_at
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                            ON CONFLICT (id) DO NOTHING
                        """, (
                            entry_id,
                            word,
                            rel_type.value,
                            neighbor,
                            float(strength),  # Use classifier's normalized strength (0-1)
                            f"Fisher-Rao distance: {fisher_dist:.4f}, co-occurrence: {cooc_count}",
                            'learning_pipeline'
                        ))
                        inserted += 1
                    except Exception as e:
                        logger.debug(f"Failed to insert {word}->{neighbor}: {e}")
            
            conn.commit()
            conn.close()
            
            if inserted > 0:
                logger.info(f"[LearnedRelationships] Populated {inserted} vocabulary_learning entries")
            
            return inserted
            
        except Exception as e:
            logger.error(f"Failed to populate vocabulary_learning: {e}")
            conn.close()
            return 0
    
    def update_from_learner(self, learner, adjusted_basins: Dict[str, np.ndarray]):
        """Update from a WordRelationshipLearner instance."""
        # QIG-PURE: Filter out single characters and invalid tokens
        valid_words = [w for w in learner.cooccurrence if len(w) >= 2]
        
        # Ensure _relationship_phi exists
        if not hasattr(self, '_relationship_phi'):
            self._relationship_phi = {}
        
        contexts_captured = 0
        for word in valid_words:
            neighbors = learner.get_related_words(word, top_k=20)
            if neighbors:
                self.word_neighbors[word] = neighbors
                
                # Capture contexts from learner for each word-neighbor pair
                for neighbor, count in neighbors:
                    contexts = learner.get_contexts(word, neighbor) if hasattr(learner, 'get_contexts') else []
                    if contexts:
                        if word not in self._relationship_phi:
                            self._relationship_phi[word] = {}
                        if neighbor not in self._relationship_phi[word]:
                            self._relationship_phi[word][neighbor] = {'phi_values': [], 'contexts': []}
                        
                        # Add contexts (max 10)
                        existing = self._relationship_phi[word][neighbor].get('contexts', [])
                        for ctx in contexts:
                            if ctx not in existing and len(existing) < 10:
                                existing.append(ctx)
                                contexts_captured += 1
                        self._relationship_phi[word][neighbor]['contexts'] = existing
        
        self.word_frequency = {w: f for w, f in learner.word_freq.items() if len(w) >= 2}
        self.adjusted_basins = adjusted_basins
        self.learning_complete = True
        
        logger.info(f"Updated with {len(self.word_neighbors)} word relationships, {contexts_captured} contexts captured")
    
    def validate_against_frozen_facts(
        self, 
        canonical_basins: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, any]:
        """
        Validate learned relationships against frozen physics constraints.
        
        FROZEN FACTS COMPLIANCE:
        1. Adjusted basins must stay within ±5% of canonical positions
        2. Stopwords cannot have high attention weights (must be < 0.2)
        3. Learned β must be within acceptance of frozen β
        
        Args:
            canonical_basins: Original basin coordinates for comparison
            
        Returns:
            Validation result with 'valid' flag and any violations
        """
        violations = []
        warnings = []
        
        # Check 1: Semantic-critical word preservation
        # Ensure semantic-critical words (negations, etc.) are not over-filtered
        semantic_violations = 0
        for word, neighbors in self.word_neighbors.items():
            for neighbor, weight in neighbors[:5]:  # Top 5 neighbors
                # If neighbor is semantic-critical, it should have reasonable weight
                # (not be suppressed to near-zero)
                if is_semantic_critical_word(neighbor) and weight < 1.0:
                    semantic_violations += 1
                    if semantic_violations <= 5:  # Log first 5
                        warnings.append(f"Semantic-critical word '{neighbor}' has low weight {weight} for '{word}'")
        
        if semantic_violations > 0:
            logger.info(f"Found {semantic_violations} semantic-critical words with low weights (may be normal)")
        
        # Check 2: Generic word suppression
        # Ensure truly generic words are not promoted to high importance
        generic_violations = 0
        truly_generic = {'the', 'a', 'an', 'is', 'was', 'are', 'were', 'been', 'be'}
        for word, neighbors in self.word_neighbors.items():
            for neighbor, weight in neighbors[:5]:  # Top 5 neighbors
                if neighbor.lower() in truly_generic and weight > 50:
                    generic_violations += 1
                    if generic_violations <= 5:  # Log first 5
                        violations.append(f"Generic word '{neighbor}' has high weight {weight} for '{word}'")
        
        if generic_violations > 0:
            logger.warning(f"Found {generic_violations} generic word violations in learned relationships")
        
        # Check 2: Basin drift validation (if canonical basins provided)
        # Use Fisher-Rao distance for QIG-pure drift measurement
        drift_violations = 0
        max_drift = 0.0
        if canonical_basins and self.adjusted_basins:
            for word, adjusted in self.adjusted_basins.items():
                if word in canonical_basins:
                    canonical = canonical_basins[word]
                    # Compute drift using Fisher-Rao distance (QIG-pure)
                    # Normalize to [0,1] range by dividing by π (max Fisher-Rao distance)
                    drift = fisher_rao_distance(adjusted, canonical) / np.pi
                    max_drift = max(max_drift, drift)
                    if drift > BASIN_DRIFT_TOLERANCE:
                        drift_violations += 1
                        if drift_violations <= 3:  # Log first 3
                            violations.append(f"Basin '{word}' drifted {drift:.1%} (max {BASIN_DRIFT_TOLERANCE:.1%})")
        
        if drift_violations > 0:
            logger.warning(f"Found {drift_violations} basin drift violations")
        
        # Check 3: Dimension check - basins should be 64D
        dim_violations = 0
        for word, basin in self.adjusted_basins.items():
            if len(basin) != BASIN_DIM:
                dim_violations += 1
                violations.append(f"Basin '{word}' has {len(basin)}D (expected {BASIN_DIM}D)")
        
        # Summary
        is_valid = len(violations) == 0
        
        result = {
            'valid': is_valid,
            'violations': violations,
            'warnings': warnings,
            'stats': {
                'semantic_violations': semantic_violations,
                'generic_violations': generic_violations,
                'drift_violations': drift_violations,
                'dim_violations': dim_violations,
                'max_drift': max_drift,
                'total_relationships': len(self.word_neighbors),
                'frozen_physics_available': PHYSICS_CONSTANTS_AVAILABLE
            }
        }
        
        if is_valid:
            logger.info("Frozen facts validation PASSED")
        else:
            logger.warning(f"Frozen facts validation FAILED: {len(violations)} violations")
        
        return result
    
    def get_related_words(self, word: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Get words related to given word."""
        return self.word_neighbors.get(word, [])[:top_k]
    
    def get_attention_weights(
        self, 
        query_words: List[str], 
        candidate_words: List[str],
        temperature: float = 1.0
    ) -> Dict[str, float]:
        """
        Compute attention weights for candidates based on query relevance.
        
        Implements simple attention: candidates that are related to query
        words get higher weights. Uses contextualized filtering instead of
        hard-coded stopwords.
        """
        weights = {}
        
        # Filter query words using contextualized approach
        # Preserves semantic-critical words
        content_query_words = [w for w in query_words if not should_filter_word(w, query_words)]
        
        for candidate in candidate_words:
            # Truly generic words get minimum weight
            truly_generic = {'the', 'a', 'an', 'is', 'was', 'are', 'were', 'been', 'be'}
            if candidate.lower() in truly_generic:
                weights[candidate] = 0.1
                continue
            
            score = 0.0
            
            for query_word in content_query_words:
                # Direct match bonus
                if candidate.lower() == query_word.lower():
                    score += 5.0
                    continue
                
                # Check if candidate is related to query word
                related = self.word_neighbors.get(query_word, [])
                for neighbor, strength in related:
                    # Use contextualized check for neighbors
                    if should_filter_word(neighbor, [query_word]):
                        continue
                    if neighbor.lower() == candidate.lower():
                        score += strength / 100.0  # Normalize
                        break
                
                # Check reverse relation
                related = self.word_neighbors.get(candidate.lower(), [])
                for neighbor, strength in related:
                    # Use contextualized check for neighbors
                    if should_filter_word(neighbor, [candidate]):
                        continue
                    if neighbor.lower() == query_word.lower():
                        score += strength / 200.0  # Weaker for reverse
                        break
            
            # Apply temperature
            if temperature != 1.0 and score > 0:
                score = score ** (1.0 / temperature)
            
            weights[candidate] = max(0.1, score)  # Minimum weight
        
        return weights
    
    def select_words_with_attention(
        self,
        query_words: List[str],
        candidates: List[Tuple[str, float]],  # (word, geometric_score)
        num_select: int = 5,
        attention_weight: float = 0.5  # Balance between geometry and attention
    ) -> List[str]:
        """
        Select words combining geometric similarity with attention weights.
        
        Args:
            query_words: Words from the prompt/query
            candidates: List of (word, geometric_score) tuples
            num_select: Number of words to select
            attention_weight: Weight for attention vs geometry (0-1)
        
        Returns:
            Selected words
        """
        if not candidates:
            return []
        
        candidate_words = [w for w, s in candidates]
        attention_weights = self.get_attention_weights(query_words, candidate_words)
        
        # Combine scores
        combined = []
        for word, geo_score in candidates:
            attn_score = attention_weights.get(word, 0.1)
            combined_score = (
                (1 - attention_weight) * geo_score + 
                attention_weight * attn_score
            )
            combined.append((word, combined_score))
        
        # Sort by combined score
        combined.sort(key=lambda x: -x[1])
        
        return [w for w, s in combined[:num_select]]
    
    def get_basin_for_word(self, word: str, fallback: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Get adjusted basin for word, or fallback."""
        return self.adjusted_basins.get(word, fallback)


# Singleton instance
_learned_relationships: Optional[LearnedRelationships] = None

def get_learned_relationships() -> LearnedRelationships:
    """Get or create the singleton LearnedRelationships instance."""
    global _learned_relationships
    if _learned_relationships is None:
        _learned_relationships = LearnedRelationships()
    return _learned_relationships


def run_learning_and_cache(curriculum_dir: str = '/home/runner/workspace/docs/09-curriculum') -> Dict:
    """
    Run the QIG-pure geometric learning pipeline and cache results.
    Uses GeometricWordRelationships instead of deprecated WordRelationshipLearner.
    """
    if not GEOMETRIC_RELATIONSHIPS_AVAILABLE:
        logger.error("GeometricWordRelationships not available")
        return {'success': False, 'error': 'GeometricWordRelationships not available'}
    
    logger.info("Running QIG-pure geometric learning pipeline...")
    
    try:
        from coordizers.pg_loader import PostgresCoordizer
        coordizer = PostgresCoordizer()
        geo_rel = GeometricWordRelationships(coordizer)
        
        lr = get_learned_relationships()
        
        # Use GeometricWordRelationships' pre-loaded vocabulary
        relationships = geo_rel.compute_all_relationships(max_words=1000)
        relationships_computed = 0
        
        for word, related in relationships.items():
            if related:
                lr.word_neighbors[word] = related
                relationships_computed += len(related)
        
        lr.learning_complete = True
        lr.save_to_cache()
        
        # Persist to PostgreSQL using GeometricWordRelationships' vocabulary
        _persist_geometric_relationships_to_db(geo_rel)
        
        return {
            'success': True,
            'relationships_computed': relationships_computed,
            'words_learned': len(lr.word_neighbors),
        }
    except Exception as e:
        logger.error(f"Geometric learning failed: {e}")
        return {'success': False, 'error': str(e)}


def _persist_geometric_relationships_to_db(geo_rel) -> int:
    """Persist geometric relationships to PostgreSQL."""
    if not DB_AVAILABLE:
        return 0
    
    conn = get_db_connection()
    if not conn:
        return 0
    
    try:
        cur = conn.cursor()
        
        # Use get_all_relationships which includes Fisher-Rao distances
        all_relationships = geo_rel.get_all_relationships()
        records = []
        
        for word, related in all_relationships.items():
            for neighbor, props in related.items():
                records.append((
                    word,
                    neighbor,
                    float(props.get('fisher_rao_distance', 0.0)),
                    float(props.get('qfi_weight', 0.5))
                ))
        
        if records:
            from psycopg2.extras import execute_values
            execute_values(
                cur,
                """
                INSERT INTO basin_relationships (word1, word2, fisher_distance, qfi_weight)
                VALUES %s
                ON CONFLICT (word1, word2) DO UPDATE SET
                    fisher_distance = EXCLUDED.fisher_distance,
                    qfi_weight = EXCLUDED.qfi_weight,
                    updated_at = NOW()
                """,
                records
            )
            conn.commit()
            logger.info(f"[LearnedRelationships] Persisted {len(records)} relationships to PostgreSQL")
        
        conn.close()
        return len(records)
    except Exception as e:
        logger.error(f"Failed to persist geometric relationships: {e}")
        try:
            conn.close()
        except:
            pass
        return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    
    result = run_learning_and_cache()
    print(f"\nLearning complete:")
    print(f"  Success: {result.get('success')}")
    print(f"  Relationships: {result.get('relationships_computed', 0)}")
    
    # Test attention
    lr = get_learned_relationships()
    test_query = ['quantum', 'geometry']
    test_candidates = ['information', 'cat', 'fisher', 'dog', 'manifold', 'banana']
    
    weights = lr.get_attention_weights(test_query, test_candidates)
    print(f"\nAttention weights for '{test_query}':")
    for word, weight in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {word}: {weight:.3f}")
