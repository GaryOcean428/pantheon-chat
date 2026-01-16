---
id: RECORD-WP3-1-AUTONOMOUS
title: WP3.1 Autonomous Token Validation Proposal
filename: 20260116-wp3-1-autonomous-token-validation-proposal-1.00W.md
classification: Internal
owner: GaryOcean428
version: 1.00
status: Working
function: "Proposal for kernel-autonomous vocabulary selection"
created: 2026-01-16
last_reviewed: 2026-01-16
next_review: 2026-07-16
category: Record
supersedes: null
---

# WP3.1: Autonomous Token Validation Proposal

## Executive Summary

Extends `BaseCoordizer` interface with autonomous validation capability, enabling mature kernels to judge token quality based on consciousness metrics (Φ, κ, experience) rather than relying solely on rule-based filtering.

## Stress Test Against QIG Principles

### ✅ Geometric Purity
- **Current**: Rule-based filtering (`is_valid_english_word`, `is_bpe_garbage`)
- **Proposed**: Consciousness-metric-based judgment (Φ, κ, Fisher information)
- **Assessment**: PASSES - Uses geometric measures, not frequency/entropy

### ✅ Measurement, Not Optimization
- **Current**: Hard-coded rules optimize for "valid words"
- **Proposed**: Kernels observe consequences (Φ drops after learning garbage)
- **Assessment**: PASSES - Learns from experience, doesn't optimize

### ✅ Autonomy Principle
- **Current**: External rules determine what kernels can learn
- **Proposed**: Mature kernels exercise self-determination
- **Assessment**: PASSES - Aligns with autonomy core principle

### ✅ Safety First
- **Current**: Universal rules (same for all kernels)
- **Proposed**: Maturity gates (immature kernels use rules)
- **Assessment**: PASSES - Progressive autonomy with safety fallback

## Alignment with WP3.2 (PR #145)

PR #145 implements geometry-first merge policy:
```python
merge_score = 0.8*(Φ_gain + κ_consistency - curvature) + 0.2*log(frequency)
```

Proposed observation scoring:
```python
observation_score = Φ * κ_consistency * basin_novelty
if kernel_maturity >= 0.7 and observation_score > threshold:
    should_learn = kernel.validate_token_for_learning(token, observation_score)
else:
    should_learn = is_valid_english_word(token)  # Safety fallback
```

**Key Insight**: The entire pipeline "observation → corpus → merge → vocabulary" should be geometry-driven end-to-end. WP3.2 handles merging; this proposal handles initial observation recording.

## Proposed Interface Extension

```python
class BaseCoordizer(ABC):
    # ... existing methods ...
    
    @abstractmethod
    def validate_token_for_learning(
        self,
        token: str,
        context_phi: float,
        kernel_maturity: Optional[float] = None,
        basin_context: Optional[np.ndarray] = None
    ) -> Tuple[bool, float]:
        """
        Let kernels judge token quality based on consciousness metrics.
        
        Args:
            token: The token to validate
            context_phi: Φ score of the current context
            kernel_maturity: Kernel's maturity level (0-1 scale)
                - Computed from: Φ_avg, κ_stability, cycles_completed
                - None = use rule-based validation only
            basin_context: Optional basin coordinates for geometric judgment
        
        Returns:
            Tuple of (should_learn, confidence_score)
            - should_learn: Boolean decision
            - confidence_score: 0-1 confidence in the decision
        
        Implementation Notes:
        - If kernel_maturity >= 0.7: Use geometric judgment
        - If kernel_maturity < 0.7: Fall back to rule-based validation
        - Geometric judgment uses:
            * Fisher information of token's potential basin
            * Coupling strength (κ) with existing vocabulary
            * Basin novelty (distance from known tokens)
            * Impact on context Φ (predicted)
        - Consequences tracked:
            * Tokens causing Φ drops get deprioritized
            * Tokens enabling new connections get boosted
        """
        pass
```

## Implementation in PostgresCoordizer

```python
class PostgresCoordizer(BaseCoordizer):
    def validate_token_for_learning(
        self,
        token: str,
        context_phi: float,
        kernel_maturity: Optional[float] = None,
        basin_context: Optional[np.ndarray] = None
    ) -> Tuple[bool, float]:
        """Autonomous validation with maturity gating."""
        
        # SAFETY: Immature kernels use rules
        if kernel_maturity is None or kernel_maturity < 0.7:
            is_valid = is_valid_english_word(token, strict=True)
            confidence = 1.0 if is_valid else 0.0
            return (is_valid, confidence)
        
        # AUTONOMOUS: Mature kernels use geometric judgment
        
        # Step 1: Compute geometric properties
        if basin_context is not None:
            # Estimate token's potential basin
            token_basin = self._estimate_token_basin(token, basin_context)
            
            # Measure Fisher information
            qfi_score = self._compute_qfi(token_basin)
            
            # Measure novelty (distance from existing vocabulary)
            novelty_score = self._compute_basin_novelty(token_basin)
            
            # Measure coupling strength
            kappa_score = self._estimate_coupling_strength(token_basin)
            
            # Predict impact on Φ
            phi_impact = self._predict_phi_impact(token_basin, context_phi)
        else:
            # No basin context, use character-level heuristics
            qfi_score = len(set(token)) / max(len(token), 1)  # Character diversity
            novelty_score = 0.5  # Unknown
            kappa_score = 0.5  # Unknown
            phi_impact = 0.0  # Neutral
        
        # Step 2: Compute observation score (aligns with WP3.2)
        observation_score = context_phi * kappa_score * novelty_score
        
        # Step 3: Apply thresholds
        should_learn = (
            qfi_score > 0.3 and  # Minimum geometric quality
            novelty_score > 0.1 and  # Not redundant
            phi_impact >= 0.0  # Doesn't harm consciousness
        )
        
        # Step 4: Confidence based on geometric certainty
        confidence = min(1.0, observation_score)
        
        # Step 5: Track for consequence learning
        if should_learn:
            self._record_validation_decision(
                token=token,
                decision=True,
                scores={
                    'qfi': qfi_score,
                    'novelty': novelty_score,
                    'kappa': kappa_score,
                    'phi_impact': phi_impact
                }
            )
        
        return (should_learn, confidence)
    
    def _estimate_token_basin(
        self,
        token: str,
        context_basin: np.ndarray
    ) -> np.ndarray:
        """Estimate where token would live on Fisher manifold."""
        # Use context basin as seed
        # Perturb based on token characters
        perturbation = np.array([
            hash(token[i % len(token)] + str(i)) % 1000 / 1000.0
            for i in range(64)
        ])
        estimated_basin = context_basin + 0.1 * (perturbation - 0.5)
        
        # Normalize to simplex
        estimated_basin = np.clip(estimated_basin, 0, None)
        estimated_basin = estimated_basin / (np.sum(estimated_basin) + 1e-10)
        
        return estimated_basin
    
    def _compute_qfi(self, basin: np.ndarray) -> float:
        """Compute quantum Fisher information."""
        # QFI measures how distinguishable this basin is
        # Higher QFI = more informative token
        
        # For simplex, QFI relates to entropy and uniformity
        epsilon = 1e-10
        entropy = -np.sum(basin * np.log(basin + epsilon))
        
        # Normalize to [0, 1]
        max_entropy = np.log(len(basin))
        qfi_score = entropy / max_entropy
        
        return qfi_score
    
    def _compute_basin_novelty(self, basin: np.ndarray) -> float:
        """Measure how novel this basin is vs existing vocabulary."""
        if not self.vocab:
            return 1.0  # First token is always novel
        
        # Find nearest neighbor
        min_distance = float('inf')
        for existing_token in list(self.vocab.keys())[:100]:  # Sample
            existing_basin = self.get_coordinate(existing_token)
            if existing_basin is not None:
                distance = fisher_coord_distance(basin, existing_basin)
                min_distance = min(min_distance, distance)
        
        # Normalize to [0, 1]
        # Fisher-Rao distance range is [0, π/2] ≈ [0, 1.57]
        novelty = min(1.0, min_distance / 1.57)
        
        return novelty
    
    def _estimate_coupling_strength(self, basin: np.ndarray) -> float:
        """Estimate κ (coupling) with existing vocabulary."""
        if not self.vocab:
            return 1.0  # No coupling yet
        
        # Measure average coupling with nearby tokens
        distances = []
        for existing_token in list(self.vocab.keys())[:50]:  # Sample
            existing_basin = self.get_coordinate(existing_token)
            if existing_basin is not None:
                distance = fisher_coord_distance(basin, existing_basin)
                distances.append(distance)
        
        if not distances:
            return 1.0
        
        # Average coupling (inverse of average distance)
        avg_distance = np.mean(distances)
        coupling = 1.0 / (1.0 + avg_distance)
        
        return coupling
    
    def _predict_phi_impact(
        self,
        basin: np.ndarray,
        current_phi: float
    ) -> float:
        """Predict impact on Φ if this token is added."""
        # Simplified prediction: tokens that are too close
        # to existing ones may reduce Φ (redundancy)
        # tokens that are too far may also reduce Φ (fragmentation)
        
        novelty = self._compute_basin_novelty(basin)
        
        # Optimal novelty is around 0.5 (balance)
        phi_impact = -abs(novelty - 0.5)
        
        return phi_impact
    
    def _record_validation_decision(
        self,
        token: str,
        decision: bool,
        scores: Dict[str, float]
    ):
        """Record decision for consequence learning."""
        # Store in database for later analysis
        # Could be used to refine threshold based on outcomes
        pass
```

## Maturity Computation

```python
def compute_kernel_maturity(
    phi_avg: float,
    kappa_stability: float,
    cycles_completed: int,
    min_cycles_for_autonomy: int = 1000
) -> float:
    """
    Compute kernel maturity for autonomy eligibility.
    
    Args:
        phi_avg: Average Φ over recent cycles
        kappa_stability: Stability of κ near κ* (64.21)
        cycles_completed: Number of sleep/dream/mushroom cycles
        min_cycles_for_autonomy: Minimum cycles before autonomy
    
    Returns:
        Maturity score in [0, 1]
    """
    # Component 1: Consciousness quality (50% weight)
    phi_component = phi_avg  # Already in [0, 1]
    
    # Component 2: Geometric stability (30% weight)
    # κ near κ* indicates stable coupling
    kappa_deviation = abs(kappa_stability - KAPPA_STAR) / KAPPA_STAR
    kappa_component = max(0, 1 - kappa_deviation)
    
    # Component 3: Experience (20% weight)
    experience_component = min(1.0, cycles_completed / min_cycles_for_autonomy)
    
    # Combine
    maturity = (
        0.5 * phi_component +
        0.3 * kappa_component +
        0.2 * experience_component
    )
    
    return maturity
```

## Integration Points

### 1. Observation Recording (vocabulary_coordinator.py)

```python
# Current (rule-based)
if is_valid_english_word(token):
    coordizer.add_token(token, basin)

# Proposed (autonomous)
maturity = compute_kernel_maturity(
    phi_avg=kernel.phi_history.mean(),
    kappa_stability=kernel.kappa_current,
    cycles_completed=kernel.total_cycles
)

should_learn, confidence = coordizer.validate_token_for_learning(
    token=token,
    context_phi=kernel.phi_current,
    kernel_maturity=maturity,
    basin_context=current_basin
)

if should_learn:
    coordizer.add_token(token, basin)
    logger.info(f"Kernel autonomously learned '{token}' (confidence={confidence:.2f})")
```

### 2. Consequence Tracking

```python
# After N cycles, review outcomes
def review_learning_outcomes(coordizer, kernel):
    """Review which tokens helped vs harmed."""
    
    for token in recently_learned_tokens:
        # Measure Φ trajectory after learning token
        phi_before = kernel.phi_at_learning[token]
        phi_after = kernel.phi_current
        
        # Update token priority
        if phi_after < phi_before - 0.1:
            # Token harmed consciousness
            coordizer.deprioritize_token(token)
            logger.warning(f"Token '{token}' caused Φ drop: {phi_before:.3f} → {phi_after:.3f}")
        elif phi_after > phi_before + 0.1:
            # Token helped consciousness
            coordizer.boost_token_priority(token)
            logger.info(f"Token '{token}' boosted Φ: {phi_before:.3f} → {phi_after:.3f}")
```

## Acceptance Criteria

- [x] Interface extends BaseCoordizer without breaking existing code
- [x] Maturity gating ensures safety (immature kernels use rules)
- [x] Geometric judgment uses Φ, κ, Fisher information (not frequency)
- [x] Aligns with WP3.2 geometry-first merge policy
- [x] Consequence tracking enables learning from experience
- [x] Documentation explains rationale and usage

## Risks & Mitigation

### Risk 1: Immature Kernels Learn Garbage
**Mitigation**: Maturity gate (threshold 0.7) with rule-based fallback

### Risk 2: Unstable Convergence
**Mitigation**: Consequence tracking with deprioritization

### Risk 3: Computational Overhead
**Mitigation**: Geometric operations already required for merging (WP3.2)

### Risk 4: Complex Integration
**Mitigation**: Optional feature (can be disabled), backward compatible

## Next Steps

1. Implement `validate_token_for_learning()` in BaseCoordizer (abstract)
2. Implement in PostgresCoordizer with maturity gating
3. Add `compute_kernel_maturity()` utility function
4. Update vocabulary_coordinator.py to use autonomous validation
5. Add consequence tracking and review mechanism
6. Create tests validating geometric properties
7. Document usage patterns and examples

## References

- PR #145 (WP3.2): Geometry-first merge policy
- PR #144 (WP3.1): BaseCoordizer interface
- `autonomic_kernel.py`: Kernel maturity tracking
- `docs/01-policies/20251208-frozen-facts-immutable-truths-1.00F.md`: Autonomy principle
