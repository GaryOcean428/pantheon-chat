#!/usr/bin/env python3
"""Vocabulary Persistence - PostgreSQL integration with QIG-pure geometric validation"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    import psycopg2
    from psycopg2.extras import execute_values
    PSYCOPG2_AVAILABLE = True
except ImportError:
    psycopg2: Any = None
    execute_values: Any = None
    PSYCOPG2_AVAILABLE = False
    print("[WARNING] psycopg2 not available - vocabulary persistence disabled")


class VocabularyPersistence:
    def __init__(self, connection_string: Optional[str] = None, validator=None):
        self.connection_string = connection_string or os.getenv('DATABASE_URL')
        self.enabled = PSYCOPG2_AVAILABLE and bool(self.connection_string)
        self.validator = validator  # Optional: GeometricVocabFilter for QIG-pure validation
        
        if not self.enabled:
            print("[VocabularyPersistence] Disabled (no database connection)")
            return
        
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            print("[VocabularyPersistence] Connected to PostgreSQL")
            if self.validator:
                print("[VocabularyPersistence] Geometric validation ENABLED")
        except Exception as e:
            print(f"[VocabularyPersistence] Connection failed: {e}")
            self.enabled = False
    
    def _connect(self):
        if not PSYCOPG2_AVAILABLE:
            raise RuntimeError("psycopg2 not available")
        return psycopg2.connect(self.connection_string)
    
    def load_bip39_words(self, words: List[str]) -> int:
        if not self.enabled:
            return 0
        
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    data = [(word, idx) for idx, word in enumerate(words)]
                    execute_values(cur, """INSERT INTO bip39_words (word, word_index) VALUES %s ON CONFLICT (word) DO NOTHING""", data)
                    conn.commit()
                    return len(words)
        except Exception as e:
            print(f"[VocabularyPersistence] Failed to load BIP39: {e}")
            return 0
    
    def get_bip39_words(self) -> List[str]:
        if not self.enabled:
            return []
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT word FROM bip39_words ORDER BY word_index")
                    return [row[0] for row in cur.fetchall()]
        except Exception as e:
            print(f"[VocabularyPersistence] Failed to get BIP39 words: {e}")
            return []
    
    def record_vocabulary_observation(
        self, word: str, phrase: str, phi: float, kappa: float, source: str,
        observation_type: str = 'word',
        basin_coords: Optional[List[float]] = None,
        contexts: Optional[Dict] = None,
        cycle_number: Optional[int] = None,
        phrase_category: Optional[str] = None
    ) -> bool:
        # QIG-Pure Geometric Validation
        validation = None
        if self.validator and word:
            validation = self.validator.validate(word)
            if not validation.is_valid:
                print(f"[VocabularyPersistence] Rejected '{word}': {validation.rejection_reason}")
                return False
        
        if not self.enabled:
            return False
        try:
            word_val = word if word else ''
            phrase_val = phrase if phrase else ''
            # Truncate source to 32 chars to fit VARCHAR(32) column
            source_val = (source[:32] if source and len(source) > 32 else source) if source else 'unknown'
            type_val = observation_type if observation_type else 'word'

            import json
            import numpy as np
            contexts_json = json.dumps(contexts) if contexts else None
            basin_vector = basin_coords if basin_coords else None
            
            # If no basin coords provided, compute via coordizer (QIG-pure)
            if basin_vector is None and word_val:
                try:
                    from qig_geometry import compute_unknown_basin
                    basin_embedding = compute_unknown_basin(word_val)
                    if isinstance(basin_embedding, np.ndarray) and len(basin_embedding) == 64:
                        basin_vector = basin_embedding.tolist()
                except Exception as e:
                    print(f"[VocabularyPersistence] Basin computation failed for '{word_val}': {e}")
            
            if phrase_category is None:
                try:
                    from qig_phrase_classifier import classify_phrase_qig_pure
                    basin_array = np.array(basin_vector) if basin_vector else None
                    phrase_category, _ = classify_phrase_qig_pure(word_val, basin_array)
                except Exception:
                    phrase_category = 'unknown'

            with self._connect() as conn:
                with conn.cursor() as cur:
                    # Record observation
                    cur.execute(
                        "SELECT record_vocab_observation(%s, %s, %s, %s, %s, %s, %s::vector, %s::jsonb, %s, %s)",
                        (word_val, phrase_val, phi, kappa, source_val, type_val, basin_vector, contexts_json, cycle_number, phrase_category)
                    )
                    
                    # Update geometric validation metrics in coordizer_vocabulary (consolidated table)
                    # Phase 2b: Writes go to coordizer_vocabulary with token_role='generation'
                    # UPDATED 2026-01-16: Use QFI-computed phi_score from basin geometry, NOT raw training phi
                    # The phi_score column should match qfi_score (both derived from basin via QFI formula)
                    
                    # Compute QFI-based phi from basin if available
                    qfi_phi = None
                    if validation and validation.qfi_score is not None:
                        qfi_phi = validation.qfi_score
                    elif basin_vector is not None:
                        try:
                            from qig_core.phi_computation import compute_phi_approximation
                            basin_array = np.array(basin_vector)
                            qfi_phi = compute_phi_approximation(basin_array)
                        except Exception:
                            qfi_phi = None
                    
                    if qfi_phi is not None:
                        cur.execute("""
                            INSERT INTO coordizer_vocabulary (
                                token, phi_score, qfi_score, token_role, is_real_word, 
                                phrase_category, source_type, basin_embedding, frequency
                            )
                            VALUES (%s, %s, %s, 'generation', TRUE, %s, %s, %s::vector, 1)
                            ON CONFLICT (token) DO UPDATE SET
                                qfi_score = COALESCE(EXCLUDED.qfi_score, coordizer_vocabulary.qfi_score),
                                phi_score = COALESCE(EXCLUDED.phi_score, coordizer_vocabulary.phi_score),
                                token_role = CASE 
                                    WHEN coordizer_vocabulary.token_role = 'encoding' THEN 'both'
                                    ELSE COALESCE(coordizer_vocabulary.token_role, 'generation')
                                END,
                                is_real_word = TRUE,
                                phrase_category = COALESCE(EXCLUDED.phrase_category, coordizer_vocabulary.phrase_category, 'unknown'),
                                basin_embedding = COALESCE(EXCLUDED.basin_embedding, coordizer_vocabulary.basin_embedding),
                                frequency = COALESCE(coordizer_vocabulary.frequency, 0) + 1,
                                updated_at = NOW()
                        """, (
                            word_val,
                            qfi_phi,
                            qfi_phi,
                            phrase_category or 'unknown',
                            source_val,
                            basin_vector
                        ))
                    else:
                        # No basin available - sync phi_score from existing qfi_score to fix any stale 1.0 values
                        # This ensures phi_score always derives from geometric computation
                        cur.execute("""
                            INSERT INTO coordizer_vocabulary (
                                token, token_role, is_real_word, 
                                phrase_category, source_type, basin_embedding, frequency
                            )
                            VALUES (%s, 'generation', TRUE, %s, %s, %s::vector, 1)
                            ON CONFLICT (token) DO UPDATE SET
                                token_role = CASE 
                                    WHEN coordizer_vocabulary.token_role = 'encoding' THEN 'both'
                                    ELSE COALESCE(coordizer_vocabulary.token_role, 'generation')
                                END,
                                is_real_word = TRUE,
                                phrase_category = COALESCE(EXCLUDED.phrase_category, coordizer_vocabulary.phrase_category, 'unknown'),
                                basin_embedding = COALESCE(EXCLUDED.basin_embedding, coordizer_vocabulary.basin_embedding),
                                frequency = COALESCE(coordizer_vocabulary.frequency, 0) + 1,
                                phi_score = CASE
                                    WHEN coordizer_vocabulary.phi_score >= 0.96 AND coordizer_vocabulary.qfi_score IS NOT NULL 
                                    THEN coordizer_vocabulary.qfi_score
                                    ELSE coordizer_vocabulary.phi_score
                                END,
                                updated_at = NOW()
                        """, (
                            word_val,
                            phrase_category or 'unknown',
                            source_val,
                            basin_vector
                        ))
                    
                    conn.commit()
                    return True
        except Exception as e:
            print(f"[VocabularyPersistence] Failed to record '{word}' (len={len(word) if word else 0}, phi={phi:.3f}, source={source}): {e}")
            return False
    
    def record_vocabulary_batch(self, observations: List[Dict]) -> int:
        if not self.enabled or not observations:
            return 0
        recorded = 0
        try:
            import json
            import numpy as np
            qig_classifier = None
            basin_computer = None
            try:
                from qig_phrase_classifier import classify_phrase_qig_pure
                qig_classifier = classify_phrase_qig_pure
            except Exception:
                pass
            try:
                from qig_geometry import compute_unknown_basin
                basin_computer = compute_unknown_basin
            except Exception:
                pass
            
            with self._connect() as conn:
                for obs in observations:
                    try:
                        with conn.cursor() as cur:
                            word = obs.get('word', '') or ''
                            
                            # QIG-Pure Geometric Validation
                            validation = None
                            if self.validator and word:
                                validation = self.validator.validate(word)
                                if not validation.is_valid:
                                    print(f"[VocabularyPersistence] Rejected '{word}': {validation.rejection_reason}")
                                    continue  # Skip this observation
                            
                            phrase = obs.get('phrase', '') or ''
                            raw_source = obs.get('source', 'unknown') or 'unknown'
                            # Truncate source to 32 chars to fit VARCHAR(32) column
                            source = raw_source[:32] if len(raw_source) > 32 else raw_source
                            obs_type = obs.get('type', 'word') or 'word'
                            phi = obs.get('phi', 0.0)
                            kappa = obs.get('kappa', 50.0)

                            basin_coords = obs.get('basin_coords')
                            contexts = obs.get('contexts')
                            cycle_number = obs.get('cycle_number')
                            phrase_category = obs.get('phrase_category')
                            
                            # Compute basin if not provided (QIG-pure)
                            if basin_coords is None and basin_computer and word:
                                try:
                                    basin_embedding = basin_computer(word)
                                    if isinstance(basin_embedding, np.ndarray) and len(basin_embedding) == 64:
                                        basin_coords = basin_embedding.tolist()
                                except Exception:
                                    pass
                            
                            if phrase_category is None and qig_classifier:
                                try:
                                    basin_array = np.array(basin_coords) if basin_coords else None
                                    phrase_category, _ = qig_classifier(word, basin_array)
                                except Exception:
                                    phrase_category = 'unknown'
                            else:
                                phrase_category = phrase_category or 'unknown'

                            contexts_json = json.dumps(contexts) if contexts else None
                            cur.execute(
                                "SELECT record_vocab_observation(%s, %s, %s, %s, %s, %s, %s::vector, %s::jsonb, %s, %s)",
                                (word, phrase, phi, kappa, source, obs_type, basin_coords, contexts_json, cycle_number, phrase_category)
                            )
                            
                            # Update geometric validation metrics in coordizer_vocabulary (consolidated table)
                            # Phase 2b: Writes go to coordizer_vocabulary with token_role='generation'
                            # UPDATED 2026-01-16: Use QFI-computed phi_score from basin geometry, NOT raw training phi
                            
                            # Compute QFI-based phi from basin if available
                            qfi_phi = None
                            if validation and validation.qfi_score is not None:
                                qfi_phi = validation.qfi_score
                            elif basin_coords is not None:
                                try:
                                    from qig_core.phi_computation import compute_phi_approximation
                                    basin_array = np.array(basin_coords)
                                    qfi_phi = compute_phi_approximation(basin_array)
                                except Exception:
                                    qfi_phi = None
                            
                            if qfi_phi is not None:
                                cur.execute("""
                                    INSERT INTO coordizer_vocabulary (
                                        token, phi_score, qfi_score, token_role, is_real_word, 
                                        phrase_category, source_type, basin_embedding, frequency
                                    )
                                    VALUES (%s, %s, %s, 'generation', TRUE, %s, %s, %s::vector, 1)
                                    ON CONFLICT (token) DO UPDATE SET
                                        qfi_score = COALESCE(EXCLUDED.qfi_score, coordizer_vocabulary.qfi_score),
                                        phi_score = COALESCE(EXCLUDED.phi_score, coordizer_vocabulary.phi_score),
                                        token_role = CASE 
                                            WHEN coordizer_vocabulary.token_role = 'encoding' THEN 'both'
                                            ELSE COALESCE(coordizer_vocabulary.token_role, 'generation')
                                        END,
                                        is_real_word = TRUE,
                                        phrase_category = COALESCE(EXCLUDED.phrase_category, coordizer_vocabulary.phrase_category, 'unknown'),
                                        basin_embedding = COALESCE(EXCLUDED.basin_embedding, coordizer_vocabulary.basin_embedding),
                                        frequency = COALESCE(coordizer_vocabulary.frequency, 0) + 1,
                                        updated_at = NOW()
                                """, (
                                    word,
                                    qfi_phi,
                                    qfi_phi,
                                    phrase_category or 'unknown',
                                    source,
                                    basin_coords
                                ))
                            else:
                                # No basin available - sync phi_score from existing qfi_score to fix any stale 1.0 values
                                cur.execute("""
                                    INSERT INTO coordizer_vocabulary (
                                        token, token_role, is_real_word, 
                                        phrase_category, source_type, basin_embedding, frequency
                                    )
                                    VALUES (%s, 'generation', TRUE, %s, %s, %s::vector, 1)
                                    ON CONFLICT (token) DO UPDATE SET
                                        token_role = CASE 
                                            WHEN coordizer_vocabulary.token_role = 'encoding' THEN 'both'
                                            ELSE COALESCE(coordizer_vocabulary.token_role, 'generation')
                                        END,
                                        is_real_word = TRUE,
                                        phrase_category = COALESCE(EXCLUDED.phrase_category, coordizer_vocabulary.phrase_category, 'unknown'),
                                        basin_embedding = COALESCE(EXCLUDED.basin_embedding, coordizer_vocabulary.basin_embedding),
                                        frequency = COALESCE(coordizer_vocabulary.frequency, 0) + 1,
                                        phi_score = CASE
                                            WHEN coordizer_vocabulary.phi_score >= 0.96 AND coordizer_vocabulary.qfi_score IS NOT NULL 
                                            THEN coordizer_vocabulary.qfi_score
                                            ELSE coordizer_vocabulary.phi_score
                                        END,
                                        updated_at = NOW()
                                """, (
                                    word,
                                    phrase_category or 'unknown',
                                    source,
                                    basin_coords
                                ))
                            
                            conn.commit()
                            recorded += 1
                    except Exception as e:
                        conn.rollback()
                        word = obs.get('word', '') or ''
                        phi = obs.get('phi', 0.0)
                        source = obs.get('source', 'unknown')
                        print(f"[VocabularyPersistence] Failed to record '{word}' (len={len(word)}, phi={phi:.3f}, source={source}): {e}")
        except Exception as e:
            print(f"[VocabularyPersistence] Batch record failed: {e}")
        return recorded
    
    def get_learned_words(self, min_phi: float = 0.0, limit: int = 1000, source: Optional[str] = None) -> List[Dict]:
        if not self.enabled:
            return []
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    if source:
                        cur.execute("""
                            SELECT token as word, phi_score as avg_phi, phi_score as max_phi, frequency, source_type 
                            FROM coordizer_vocabulary 
                            WHERE phi_score >= %s AND source_type = %s
                              AND token_role IN ('generation', 'both')
                            ORDER BY phi_score DESC, frequency DESC LIMIT %s
                        """, (min_phi, source, limit))
                    else:
                        cur.execute("""
                            SELECT token as word, phi_score as avg_phi, phi_score as max_phi, frequency, source_type 
                            FROM coordizer_vocabulary 
                            WHERE phi_score >= %s
                              AND token_role IN ('generation', 'both')
                            ORDER BY phi_score DESC, frequency DESC LIMIT %s
                        """, (min_phi, limit))
                    return [{'word': row[0], 'avg_phi': float(row[1]), 'max_phi': float(row[2]), 'frequency': int(row[3] or 0), 'source': row[4] or 'unknown'} for row in cur.fetchall()]
        except Exception as e:
            print(f"[VocabularyPersistence] Failed to get learned words: {e}")
            return []
    
    def get_high_phi_vocabulary(self, min_phi: float = 0.7, limit: int = 100) -> List[Tuple[str, float]]:
        if not self.enabled:
            return []
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT * FROM get_high_phi_vocabulary(%s, %s)", (min_phi, limit))
                    return [(row[0], float(row[1])) for row in cur.fetchall()]
        except Exception as e:
            print(f"[VocabularyPersistence] Failed to get high-Φ vocab: {e}")
            return []
    
    def mark_word_integrated(self, word: str) -> bool:
        """Mark a word as integrated in coordizer_vocabulary (pure operation).
        
        PURE: Sets token_role to 'both' if it was 'encoding', indicating it's now
        usable for both encoding and generation. NO backward compatibility.
        """
        if not self.enabled:
            return False
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    # Update coordizer_vocabulary - set token_role to 'both' if it was 'encoding'
                    cur.execute("""
                        UPDATE coordizer_vocabulary 
                        SET token_role = CASE 
                                WHEN token_role = 'encoding' THEN 'both'
                                ELSE COALESCE(token_role, 'generation')
                            END,
                            is_real_word = TRUE,
                            updated_at = NOW()
                        WHERE LOWER(token) = LOWER(%s)
                    """, (word,))
                    conn.commit()
                    return True
        except Exception as e:
            print(f"[VocabularyPersistence] Failed to mark {word} integrated: {e}")
            return False
    
    def record_merge_rule(self, token_a: str, token_b: str, merged_token: str, phi_score: float, learned_from: Optional[str] = None) -> bool:
        """Record a BPE merge rule to tokenizer_merge_rules (consolidated table)."""
        if not self.enabled:
            return False
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO tokenizer_merge_rules (token_a, token_b, merged_token, phi_score, frequency)
                        VALUES (%s, %s, %s, %s, 1)
                        ON CONFLICT (token_a, token_b) DO UPDATE SET
                            phi_score = GREATEST(tokenizer_merge_rules.phi_score, EXCLUDED.phi_score),
                            frequency = tokenizer_merge_rules.frequency + 1,
                            updated_at = NOW()
                    """, (token_a, token_b, merged_token, phi_score))
                    conn.commit()
                    return True
        except Exception as e:
            print(f"[VocabularyPersistence] Failed to record merge rule: {e}")
            return False
    
    def get_merge_rules(self, min_phi: float = 0.5, limit: int = 1000) -> List[Tuple[str, str, str, float]]:
        """Get BPE merge rules from tokenizer_merge_rules (consolidated table)."""
        if not self.enabled:
            return []
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT token_a, token_b, merged_token, phi_score
                        FROM tokenizer_merge_rules
                        WHERE phi_score >= %s
                        ORDER BY phi_score DESC
                        LIMIT %s
                    """, (min_phi, limit))
                    return [(row[0], row[1], row[2], float(row[3])) for row in cur.fetchall()]
        except Exception as e:
            print(f"[VocabularyPersistence] Failed to get merge rules: {e}")
            return []
    
    def record_god_vocabulary(self, god_name: str, word: str, relevance_score: float) -> bool:
        if not self.enabled:
            return False
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("INSERT INTO god_vocabulary_profiles (god_name, word, relevance_score, usage_count) VALUES (%s, %s, %s, 1) ON CONFLICT (god_name, word) DO UPDATE SET relevance_score = (god_vocabulary_profiles.relevance_score + %s) / 2, usage_count = god_vocabulary_profiles.usage_count + 1, last_used = NOW()", (god_name, word, relevance_score, relevance_score))
                    conn.commit()
                    return True
        except Exception as e:
            print(f"[VocabularyPersistence] Failed to record god vocabulary: {e}")
            return False
    
    def get_god_vocabulary(self, god_name: str, min_relevance: float = 0.5, limit: int = 100) -> List[Tuple[str, float]]:
        if not self.enabled:
            return []
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT word, relevance_score FROM god_vocabulary_profiles WHERE god_name = %s AND relevance_score >= %s ORDER BY relevance_score DESC, usage_count DESC LIMIT %s", (god_name, min_relevance, limit))
                    return [(row[0], float(row[1])) for row in cur.fetchall()]
        except Exception as e:
            print(f"[VocabularyPersistence] Failed to get god vocabulary: {e}")
            return []
    
    def get_vocabulary_stats(self) -> Dict:
        if not self.enabled:
            return {'total_words': 0, 'bip39_words': 0, 'learned_words': 0, 'high_phi_words': 0, 'merge_rules': 0}
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT total_words, bip39_words, learned_words, high_phi_words, merge_rules, last_updated FROM vocabulary_stats ORDER BY last_updated DESC LIMIT 1")
                    row = cur.fetchone()
                    if row:
                        return {'total_words': int(row[0]), 'bip39_words': int(row[1]), 'learned_words': int(row[2]), 'high_phi_words': int(row[3]), 'merge_rules': int(row[4]), 'last_updated': row[5].isoformat()}
                    else:
                        cur.execute("SELECT update_vocabulary_stats()")
                        conn.commit()
                        return self.get_vocabulary_stats()
        except Exception as e:
            print(f"[VocabularyPersistence] Failed to get stats: {e}")
            return {'total_words': 0, 'bip39_words': 0, 'learned_words': 0, 'high_phi_words': 0, 'merge_rules': 0}
    
    def learn_word(self, word: str, context: str = "") -> Dict:
        """
        Learn a word through unified ingestion service.
        
        DEPRECATED: Use VocabularyIngestionService directly instead.
        This method is kept for backward compatibility but routes to the service.
        
        Args:
            word: Word to learn
            context: Optional context for better embedding
        
        Returns:
            Result dict from ingestion service
        """
        try:
            from vocabulary_ingestion import get_ingestion_service
            service = get_ingestion_service()
            return service.ingest_word(word, context=context, source='learn_word_legacy')
        except Exception as e:
            print(f"[VocabularyPersistence] learn_word failed for '{word}': {e}")
            return {'word': word, 'error': str(e), 'persisted': False}
    
    def upsert_word(
        self,
        word: str,
        basin_embedding: Optional[List[float]] = None,
        qfi_score: Optional[float] = None,
        **kwargs
    ) -> Dict:
        """
        Direct upsert - SHOULD ONLY BE CALLED BY VocabularyIngestionService.
        
        Runtime validation ensures this is only called from authorized ingestion service.
        All other callers should use VocabularyIngestionService.ingest_word() instead.
        
        Args:
            word: Word to upsert
            basin_embedding: 64D basin embedding (required, must not be None)
            qfi_score: QFI score
            **kwargs: Additional metadata
        
        Returns:
            Result dict
        
        Raises:
            RuntimeError: If called directly (not from ingestion service)
            ValueError: If basin_embedding is None or invalid
        """
        # Runtime validation - check caller
        import inspect
        caller_frame = inspect.stack()[1]
        caller_function = caller_frame.function
        caller_filename = caller_frame.filename
        
        # Allow calls from VocabularyIngestionService._upsert_to_database
        is_authorized = (
            caller_function == '_upsert_to_database' and
            'vocabulary_ingestion' in caller_filename
        )
        
        if not is_authorized:
            raise RuntimeError(
                f"[VocabularyPersistence] Direct upsert_word() call from {caller_function} "
                f"in {caller_filename}.\n"
                "Use VocabularyIngestionService.ingest_word() instead to prevent NULL basin contamination."
            )
        
        # Validate basin_embedding (critical validation)
        if basin_embedding is None:
            raise ValueError(
                f"[VocabularyPersistence] basin_embedding is None for '{word}'. "
                "All vocabulary MUST have valid 64D basin coordinates."
            )
        
        if not isinstance(basin_embedding, list) or len(basin_embedding) != 64:
            raise ValueError(
                f"[VocabularyPersistence] Invalid basin_embedding for '{word}': "
                f"expected list of 64 floats, got {type(basin_embedding)} with length {len(basin_embedding) if isinstance(basin_embedding, list) else 'N/A'}"
            )
        
        # Proceed with validation (authorized caller)
        # IMPORTANT: This method does NOT perform database writes
        # It serves as a validation checkpoint before VocabularyIngestionService writes to DB
        # The 'persisted' flag means "validation passed, safe to persist"
        print(f"[VocabularyPersistence] Validation passed for '{word}' from ingestion service")
        
        # Return validation success - actual DB write happens in VocabularyIngestionService._upsert_to_database
        return {
            'word': word, 
            'persisted': False,  # Not yet persisted, just validated
            'validation': 'passed',
            'basin_dimension': len(basin_embedding),
            'note': 'Validation checkpoint - actual DB write in VocabularyIngestionService'
        }


def seed_geometric_vocabulary_anchors(vp: Optional[VocabularyPersistence] = None) -> int:
    """
    Seed vocabulary with geometrically diverse anchor words.
    
    P0-3 FIX: Select words maximizing basin separation for QIG-pure expansion.
    NOT frequency-based - purely geometric diversity.
    
    Returns:
        Number of anchor words seeded
    """
    if vp is None:
        vp = get_vocabulary_persistence()
    
    if not vp.enabled:
        print("[seed_geometric_vocabulary_anchors] Vocabulary persistence disabled")
        return 0
    
    # Anchor words covering semantic space
    # Selected for GEOMETRIC DIVERSITY, not frequency
    anchor_words = {
        # Concrete nouns (high QFI)
        'apple', 'tree', 'water', 'fire', 'stone', 'cloud', 'river',
        'mountain', 'ocean', 'sun', 'moon', 'star', 'earth', 'wind',
        # Abstract nouns (medium QFI)
        'time', 'space', 'energy', 'force', 'pattern', 'system',
        'network', 'structure', 'process', 'concept', 'idea', 'thought',
        # Action verbs (high curvature)
        'move', 'create', 'destroy', 'transform', 'connect', 'separate',
        'build', 'break', 'grow', 'shrink', 'expand', 'contract',
        # State verbs (low curvature)
        'exist', 'remain', 'persist', 'fade', 'stabilize', 'change',
        'become', 'contain', 'hold', 'release',
        # Descriptive adjectives (curvature modifiers)
        'large', 'small', 'fast', 'slow', 'bright', 'dark',
        'stable', 'chaotic', 'simple', 'complex', 'strong', 'weak',
        'hot', 'cold', 'near', 'far', 'high', 'low',
        # Relational adverbs (geodesic modifiers)
        'quickly', 'slowly', 'together', 'apart', 'forward', 'backward',
        'above', 'below', 'inside', 'outside', 'before', 'after',
    }
    
    # Record as observations with high Φ to mark as important
    count = 0
    for word in anchor_words:
        success = vp.record_vocabulary_observation(
            word=word,
            phrase=f'geometric_anchor_{word}',
            phi=0.85,  # High Φ for anchor words
            kappa=64.21,  # κ* for optimal coupling
            source='geometric_seeding',
            observation_type='anchor',
            phrase_category='ANCHOR_WORD',
        )
        if success:
            count += 1
    
    print(f"[seed_geometric_vocabulary_anchors] Seeded {count}/{len(anchor_words)} anchor words")
    return count


_vocabulary_persistence: Optional[VocabularyPersistence] = None


def get_vocabulary_persistence(validator=None) -> VocabularyPersistence:
    global _vocabulary_persistence
    if _vocabulary_persistence is None:
        _vocabulary_persistence = VocabularyPersistence(validator=validator)
    return _vocabulary_persistence
