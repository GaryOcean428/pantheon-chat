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
            source_val = source if source else 'unknown'
            type_val = observation_type if observation_type else 'word'

            import json
            import numpy as np
            contexts_json = json.dumps(contexts) if contexts else None
            basin_vector = basin_coords if basin_coords else None
            
            if phrase_category is None:
                try:
                    from qig_phrase_classifier import classify_phrase_qig_pure
                    basin_array = np.array(basin_coords) if basin_coords else None
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
                    
                    # Update geometric validation metrics if available
                    if validation:
                        cur.execute("""
                            UPDATE learned_words SET 
                                qfi_score = %s,
                                basin_distance = %s,
                                curvature_std = %s,
                                entropy_score = %s,
                                is_geometrically_valid = TRUE,
                                validation_reason = NULL
                            WHERE word_text = %s
                        """, (
                            validation.qfi_score,
                            validation.basin_distance,
                            validation.curvature_std,
                            validation.entropy_score,
                            word_val
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
            try:
                from qig_phrase_classifier import classify_phrase_qig_pure
                qig_classifier = classify_phrase_qig_pure
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
                            source = obs.get('source', 'unknown') or 'unknown'
                            obs_type = obs.get('type', 'word') or 'word'
                            phi = obs.get('phi', 0.0)
                            kappa = obs.get('kappa', 50.0)

                            basin_coords = obs.get('basin_coords')
                            contexts = obs.get('contexts')
                            cycle_number = obs.get('cycle_number')
                            phrase_category = obs.get('phrase_category')
                            
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
                            
                            # Update geometric validation metrics if available
                            if validation:
                                cur.execute("""
                                    UPDATE learned_words SET 
                                        qfi_score = %s,
                                        basin_distance = %s,
                                        curvature_std = %s,
                                        entropy_score = %s,
                                        is_geometrically_valid = TRUE,
                                        validation_reason = NULL
                                    WHERE word_text = %s
                                """, (
                                    validation.qfi_score,
                                    validation.basin_distance,
                                    validation.curvature_std,
                                    validation.entropy_score,
                                    word
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
                        cur.execute("SELECT word_text, avg_phi, max_phi, frequency, source FROM learned_words WHERE avg_phi >= %s AND source = %s ORDER BY avg_phi DESC, frequency DESC LIMIT %s", (min_phi, source, limit))
                    else:
                        cur.execute("SELECT word_text, avg_phi, max_phi, frequency, source FROM learned_words WHERE avg_phi >= %s ORDER BY avg_phi DESC, frequency DESC LIMIT %s", (min_phi, limit))
                    return [{'word': row[0], 'avg_phi': float(row[1]), 'max_phi': float(row[2]), 'frequency': int(row[3]), 'source': row[4]} for row in cur.fetchall()]
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
        if not self.enabled:
            return False
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("UPDATE learned_words SET is_integrated = TRUE WHERE word_text = %s", (word,))
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


_vocabulary_persistence: Optional[VocabularyPersistence] = None


def get_vocabulary_persistence(validator=None) -> VocabularyPersistence:
    global _vocabulary_persistence
    if _vocabulary_persistence is None:
        _vocabulary_persistence = VocabularyPersistence(validator=validator)
    return _vocabulary_persistence
