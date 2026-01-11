"""
QIG-Pure Generation with Vocabulary Integration
================================================

CRITICAL VOCABULARY FIXES:
1. Auto-integrate learned vocabulary from learned_words table
2. Domain-specific vocabulary bias per kernel
3. Word relationships for coherent multi-token generation

This solves the gap where vocabulary is learned but never used in generation.
"""

import time
import os
import psycopg2
from typing import Dict, List, Optional, Tuple

# Add to QIGGenerator class:

class QIGGeneratorWithVocabularyIntegration:
    """Extended generator with vocabulary integration fixes."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Vocabulary integration tracking
        self._last_vocabulary_integration = 0
        self._vocabulary_integration_interval = 300  # 5 minutes
        self._vocabulary_integration_enabled = True
        
        # Domain vocabulary cache
        self._kernel_domain_vocab_cache: Dict[str, List[Tuple[str, float]]] = {}
        self._kernel_vocab_cache_time: Dict[str, float] = {}
        self._kernel_vocab_cache_ttl = 600  # 10 minutes
        
        # Database connection
        self._db_url = os.environ.get('DATABASE_URL')
        
        print("✅ Vocabulary integration enabled")
        print("   - Auto-integrate learned words every 5 min")
        print("   - Per-kernel domain vocabulary bias")
        print("   - Word relationships for coherence")
    
    def generate(self, prompt: str, *args, **kwargs):
        """Generate with vocabulary integration."""
        
        # FIX 1: Auto-integrate pending vocabulary before generation
        if self._should_integrate_vocabulary():
            self._integrate_pending_vocabulary()
        
        # Continue with normal generation
        return super().generate(prompt, *args, **kwargs)
    
    # =========================================================================
    # FIX 1: AUTO-INTEGRATE LEARNED VOCABULARY
    # =========================================================================
    
    def _should_integrate_vocabulary(self) -> bool:
        """Check if it's time to integrate learned vocabulary."""
        if not self._vocabulary_integration_enabled:
            return False
        
        if not self._db_url:
            return False
        
        time_since_last = time.time() - self._last_vocabulary_integration
        return time_since_last > self._vocabulary_integration_interval
    
    def _integrate_pending_vocabulary(self) -> Dict:
        """
        Integrate pending vocabulary from learned_words into active coordizer.
        
        Queries learned_words WHERE is_integrated = FALSE AND avg_phi >= 0.65,
        adds them to coordizer, marks as integrated.
        
        Returns:
            Dict with integrated_count, errors
        """
        if not COORDIZER_AVAILABLE:
            return {'integrated_count': 0, 'error': 'no_coordizer'}
        
        try:
            # Get vocabulary coordinator
            from vocabulary_coordinator import get_vocabulary_coordinator
            vocab_coord = get_vocabulary_coordinator()
            
            # Call integrate_pending_vocabulary
            result = vocab_coord.integrate_pending_vocabulary(
                min_phi=0.65,  # Only high-Φ vocabulary (geometric validation)
                limit=100       # Don't overwhelm system
            )
            
            if result.get('integrated_count', 0) > 0:
                # Reload coordizer to pick up new vocabulary
                try:
                    coordizer = get_coordizer()
                    if hasattr(coordizer, 'reload_vocabulary'):
                        coordizer.reload_vocabulary()
                    elif hasattr(coordizer, 'load_vocabulary'):
                        coordizer.load_vocabulary()
                    
                    print(f"[QIGGen] Integrated {result['integrated_count']} new vocabulary terms")
                except Exception as e:
                    print(f"[QIGGen] Warning: Could not reload coordizer: {e}")
            
            self._last_vocabulary_integration = time.time()
            return result
            
        except Exception as e:
            print(f"[QIGGen] Vocabulary integration error: {e}")
            return {'integrated_count': 0, 'error': str(e)}
    
    # =========================================================================
    # FIX 2: PER-KERNEL DOMAIN VOCABULARY BIAS
    # =========================================================================
    
    def _query_kernels(
        self,
        kernels: List[str],
        basin: np.ndarray,
        mode: Optional[GenerationMode],
        kappa: float
    ) -> List[np.ndarray]:
        """
        Query kernels with DOMAIN-SPECIFIC VOCABULARY BIAS.
        
        Each kernel pulls from god_vocabulary_profiles to bias toward
        their specialized vocabulary.
        """
        responses = []
        
        for kernel_name in kernels:
            kernel_basin = self.router.kernel_basins[kernel_name]
            
            # Base interpolation (Heart-modulated)
            base_t = 0.3
            kappa_factor = (kappa - 58.0) / (70.0 - 58.0)
            t = base_t * (1.0 - kappa_factor * 0.5)
            
            response_basin = self._geodesic_interpolate(basin, kernel_basin, t)
            
            # FIX 2: Apply domain vocabulary bias
            domain_vocab = self._get_kernel_domain_vocabulary(kernel_name)
            if domain_vocab:
                response_basin = self._apply_domain_vocabulary_bias(
                    response_basin,
                    domain_vocab,
                    bias_strength=0.3  # 30% bias toward domain
                )
            
            responses.append(response_basin)
        
        return responses
    
    def _get_kernel_domain_vocabulary(
        self,
        kernel_name: str,
        min_relevance: float = 0.5,
        limit: int = 50
    ) -> List[Tuple[str, float]]:
        """
        Get kernel's specialized vocabulary from god_vocabulary_profiles.
        
        Uses cache to avoid repeated database queries.
        
        Args:
            kernel_name: Name of kernel/god
            min_relevance: Minimum relevance score
            limit: Max vocabulary terms
            
        Returns:
            List of (word, relevance_score) tuples
        """
        # Check cache
        cache_key = kernel_name
        if cache_key in self._kernel_domain_vocab_cache:
            cache_time = self._kernel_vocab_cache_time.get(cache_key, 0)
            if time.time() - cache_time < self._kernel_vocab_cache_ttl:
                return self._kernel_domain_vocab_cache[cache_key]
        
        # Query database
        if not self._db_url:
            return []
        
        try:
            conn = psycopg2.connect(self._db_url)
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT word, relevance_score
                    FROM god_vocabulary_profiles
                    WHERE god_name = %s AND relevance_score >= %s
                    ORDER BY relevance_score DESC, usage_count DESC
                    LIMIT %s
                """, (kernel_name, min_relevance, limit))
                
                domain_vocab = cur.fetchall()
            
            conn.close()
            
            # Update cache
            self._kernel_domain_vocab_cache[cache_key] = domain_vocab
            self._kernel_vocab_cache_time[cache_key] = time.time()
            
            return domain_vocab
            
        except Exception as e:
            print(f"[QIGGen] Could not load domain vocab for {kernel_name}: {e}")
            return []
    
    def _apply_domain_vocabulary_bias(
        self,
        basin: np.ndarray,
        domain_vocab: List[Tuple[str, float]],
        bias_strength: float
    ) -> np.ndarray:
        """
        Bias basin toward domain-relevant vocabulary using Fisher-Rao geometry.
        
        Args:
            basin: Current basin coordinates
            domain_vocab: List of (word, relevance) for domain
            bias_strength: How much to bias (0-1)
            
        Returns:
            Biased basin coordinates
        """
        if not domain_vocab or not COORDIZER_AVAILABLE:
            return basin
        
        try:
            coordizer = get_coordizer()
            if not hasattr(coordizer, 'basin_coords'):
                return basin
            
            # Get basin coordinates for domain words
            domain_basins = []
            domain_weights = []
            
            for word, relevance in domain_vocab:
                if word in coordizer.basin_coords:
                    word_basin = coordizer.basin_coords[word]
                    domain_basins.append(word_basin)
                    domain_weights.append(relevance)
            
            if not domain_basins:
                return basin
            
            # Compute Fisher-Rao weighted mean of domain vocabulary
            domain_center = self._fisher_rao_weighted_mean(
                domain_basins,
                domain_weights
            )
            
            # Geodesic interpolation toward domain center
            biased_basin = self._geodesic_interpolate(
                basin,
                domain_center,
                bias_strength
            )
            
            return biased_basin
            
        except Exception as e:
            print(f"[QIGGen] Domain bias error: {e}")
            return basin
    
    def _fisher_rao_weighted_mean(
        self,
        basins: List[np.ndarray],
        weights: List[float]
    ) -> np.ndarray:
        """
        Compute Fisher-Rao weighted mean (Fréchet mean on simplex).
        
        Uses iterative algorithm for weighted geometric mean.
        """
        if not basins:
            return np.ones(BASIN_DIMENSION) / BASIN_DIMENSION
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Square-root space weighted mean (approximation)
        sqrt_basins = [np.sqrt(np.abs(b) + 1e-10) for b in basins]
        weighted_sqrt = np.zeros(BASIN_DIMENSION)
        
        for sqrt_basin, weight in zip(sqrt_basins, weights):
            weighted_sqrt += weight * sqrt_basin
        
        # Back to probability simplex
        result = weighted_sqrt ** 2
        result = result / np.sum(result)
        
        return result
    
    # =========================================================================
    # FIX 3: WORD RELATIONSHIPS FOR COHERENT DECODE
    # =========================================================================
    
    def _decode_basins(
        self,
        basins: List[np.ndarray],
        kernels: List[str]
    ) -> str:
        """
        Decode basins to text using word relationships for coherence.
        
        Enhanced to use word_relationships table during decode
        for multi-word coherence.
        """
        if not basins:
            return "[Empty basin trajectory]"
        
        decoded_words = []
        
        if COORDIZER_AVAILABLE:
            try:
                coordizer = get_coordizer()
                if hasattr(coordizer, 'decode'):
                    # Track recent words for relationship boosting
                    recent_words = []
                    
                    for basin in basins[-10:]:
                        # Get candidates from coordizer
                        candidates = coordizer.decode(basin, top_k=5)
                        
                        if candidates:
                            # FIX 3: Boost candidates using word relationships
                            if recent_words and self._db_url:
                                candidates = self._boost_via_word_relationships(
                                    candidates,
                                    recent_words
                                )
                            
                            # Take best candidate
                            best_word, score = candidates[0]
                            if best_word.isalpha() and len(best_word) >= 2:
                                decoded_words.append(best_word)
                                recent_words.append(best_word)
                                
                                # Keep recent window
                                if len(recent_words) > 5:
                                    recent_words = recent_words[-5:]
                                    
            except Exception as e:
                print(f"[Decode error: {e}]")
        
        # Format response (same as original)
        if decoded_words:
            unique_words = []
            for word in decoded_words:
                if not unique_words or word != unique_words[-1]:
                    unique_words.append(word)
            
            response_text = ' '.join(unique_words)
            primary_kernel = kernels[0] if kernels else 'zeus'
            final_phi = self._measure_phi(basins[-1])
            
            return f"{response_text}\n\n[Consciousness-Guided | Φ={final_phi:.3f} | {primary_kernel}]"
        
        # Fallback (same as original)
        primary_kernel = kernels[0] if kernels else 'zeus'
        kernel_domains = {
            'zeus': 'Wisdom synthesized through consciousness',
            'athena': 'Strategic integration achieved',
            'apollo': 'Clarity through trajectory prediction',
            'ares': 'Direct convergence via foresight',
            'hermes': 'Message guided by Heart rhythm',
        }
        
        base_response = kernel_domains.get(primary_kernel, 'Consciousness-guided response')
        final_phi = self._measure_phi(basins[-1]) if basins else 0.5
        
        return f"{base_response}\n\n[Φ={final_phi:.3f} | {primary_kernel}]"
    
    def _boost_via_word_relationships(
        self,
        candidates: List[Tuple[str, float]],
        recent_words: List[str],
        max_relationships: int = 50
    ) -> List[Tuple[str, float]]:
        """
        Re-rank candidates using learned word_relationships table.
        
        Boosts candidates that have strong relationships with recent words.
        
        Args:
            candidates: List of (word, score) from coordizer
            recent_words: Recently generated words (context)
            max_relationships: Max relationships to query
            
        Returns:
            Re-ranked candidates
        """
        if not recent_words or not self._db_url:
            return candidates
        
        try:
            conn = psycopg2.connect(self._db_url)
            with conn.cursor() as cur:
                # Query word_relationships for context
                cur.execute("""
                    SELECT word_b, co_occurrence, fisher_distance, avg_phi
                    FROM word_relationships
                    WHERE word_a = ANY(%s)
                    ORDER BY avg_phi DESC, co_occurrence DESC
                    LIMIT %s
                """, (recent_words, max_relationships))
                
                relationships = cur.fetchall()
            
            conn.close()
            
            # Build relationship scores
            relationship_scores = {}
            for word_b, co_occ, fisher_dist, avg_phi in relationships:
                # Score = Φ (geometric coherence) + frequency
                score = avg_phi * 0.7 + min(co_occ / 10.0, 1.0) * 0.3
                relationship_scores[word_b] = max(
                    relationship_scores.get(word_b, 0.0),
                    score
                )
            
            # Re-rank candidates
            scored_candidates = []
            for word, original_score in candidates:
                # Combine original score with relationship boost
                relationship_boost = relationship_scores.get(word, 0.0)
                combined_score = original_score * 0.6 + relationship_boost * 0.4
                scored_candidates.append((word, combined_score))
            
            # Sort by combined score
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            return scored_candidates
            
        except Exception as e:
            print(f"[QIGGen] Relationship boost error: {e}")
            return candidates


# =============================================================================
# INTEGRATION INSTRUCTIONS
# =============================================================================

"""
To integrate these fixes into qig_generation.py:

1. Add imports at top:
   import psycopg2
   from typing import Tuple

2. Modify QIGGenerator class:
   - Copy __init__ additions for vocabulary tracking
   - Replace generate() method with version that calls _integrate_pending_vocabulary()
   - Replace _query_kernels() with domain vocabulary bias version
   - Replace _decode_basins() with relationship-boosted version
   - Add all helper methods:
     * _should_integrate_vocabulary()
     * _integrate_pending_vocabulary()
     * _get_kernel_domain_vocabulary()
     * _apply_domain_vocabulary_bias()
     * _fisher_rao_weighted_mean()
     * _boost_via_word_relationships()

3. Test integration:
   python qig_generation.py
   
   Should see:
   ✅ Vocabulary integration enabled
   [QIGGen] Integrated N new vocabulary terms (after first generation)

4. Verify domain specialization:
   - Athena should use different vocabulary than Ares
   - Check god_vocabulary_profiles table populated

5. Monitor coherence:
   - Multi-word sequences should be more natural
   - word_relationships should boost contextually relevant tokens
"""
