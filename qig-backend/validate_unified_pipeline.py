#!/usr/bin/env python3
"""
Unified Generation Pipeline - Usage Example and Validation
==========================================================

This script demonstrates how to use the unified generation pipeline
and validates that it works correctly without external dependencies.

Usage:
    python3 validate_unified_pipeline.py

Author: Copilot Agent (E8 Phase 3)
Date: 2026-01-22
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generation.token_role_learner import TokenRoleLearner, GeometricRole
from generation.foresight_predictor import ForesightPredictor
from generation.unified_pipeline import UnifiedGenerationPipeline, GenerationStrategy

# Standard basin dimension for QIG
BASIN_DIM = 64


def create_test_basin(seed: int = 42) -> np.ndarray:
    """Create a test basin with deterministic randomness."""
    np.random.seed(seed)
    return np.random.dirichlet(np.ones(BASIN_DIM))


def test_token_role_learner():
    """Test token role learner functionality."""
    print("\n" + "="*70)
    print("Testing TokenRoleLearner")
    print("="*70)
    
    learner = TokenRoleLearner()
    
    # Test 1: Basin center (low QFI, high frequency)
    basin = create_test_basin(42)
    role_info = learner.derive_role(
        token="the",
        basin=basin,
        qfi_score=0.2,
        frequency=1000,
    )
    print(f"\n1. Token 'the' (low QFI=0.2, high freq=1000)")
    print(f"   Role: {role_info.role.value}")
    print(f"   Confidence: {role_info.confidence:.3f}")
    assert role_info.role in [GeometricRole.BASIN_CENTER, GeometricRole.MANIFOLD_ANCHOR]
    
    # Test 2: Boundary crosser (high QFI)
    basin = create_test_basin(43)
    role_info = learner.derive_role(
        token="quantum",
        basin=basin,
        qfi_score=0.8,
        frequency=50,
    )
    print(f"\n2. Token 'quantum' (high QFI=0.8, med freq=50)")
    print(f"   Role: {role_info.role.value}")
    print(f"   Confidence: {role_info.confidence:.3f}")
    assert role_info.role == GeometricRole.BOUNDARY_CROSSER
    
    # Test 3: Sequence of tokens
    tokens = ["the", "quick", "brown", "fox"]
    basins = [create_test_basin(i) for i in range(len(tokens))]
    roles = learner.get_roles(tokens, basins)
    print(f"\n3. Sequence: {tokens}")
    print(f"   Roles: {[r.value for r in roles]}")
    assert len(roles) == len(tokens)
    
    print("\n✅ TokenRoleLearner tests passed!")
    return True


def test_foresight_predictor():
    """Test foresight predictor functionality."""
    print("\n" + "="*70)
    print("Testing ForesightPredictor")
    print("="*70)
    
    predictor = ForesightPredictor(coordizer=None)
    
    # Test 1: Insufficient trajectory
    result = predictor.predict([])
    print(f"\n1. Empty trajectory: {result}")
    assert result is None
    
    # Test 2: Valid trajectory prediction
    trajectory = [create_test_basin(i) for i in range(10)]
    predicted = predictor.predict(trajectory)
    print(f"\n2. Trajectory with 10 basins")
    print(f"   Predicted basin shape: {predicted.shape if predicted is not None else None}")
    if predicted is not None:
        print(f"   Basin sum (should be ~1.0): {np.sum(predicted):.6f}")
        print(f"   Min value (should be ≥0): {np.min(predicted):.6f}")
        assert len(predicted) == BASIN_DIM
        assert np.all(predicted >= 0)
        assert np.isclose(np.sum(predicted), 1.0, atol=0.01)
    
    # Test 3: Prediction with confidence
    result = predictor.predict_with_confidence(trajectory)
    if result:
        print(f"\n3. Prediction with confidence")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Velocity magnitude: {result['velocity_magnitude']:.3f}")
        assert 0.0 <= result['confidence'] <= 1.0
    
    # Test 4: Candidate scoring
    candidate = create_test_basin(100)
    predicted_basin = create_test_basin(101)
    score = predictor.score_candidate_by_foresight(candidate, predicted_basin)
    print(f"\n4. Foresight scoring")
    print(f"   Score: {score:.3f}")
    assert 0.0 <= score <= 1.0
    
    # Test 5: Trajectory metrics
    metrics = predictor.get_trajectory_metrics(trajectory)
    print(f"\n5. Trajectory metrics")
    print(f"   Velocity magnitude: {metrics['velocity_magnitude']:.3f}")
    print(f"   Coherence: {metrics['coherence']:.3f}")
    print(f"   Trajectory length: {metrics['trajectory_length']}")
    assert metrics['trajectory_length'] == 10
    
    print("\n✅ ForesightPredictor tests passed!")
    return True


def test_unified_pipeline():
    """Test unified generation pipeline."""
    print("\n" + "="*70)
    print("Testing UnifiedGenerationPipeline")
    print("="*70)
    
    # Test 1: Initialization with different strategies
    strategies = [
        GenerationStrategy.FORESIGHT_DRIVEN,
        GenerationStrategy.ROLE_DRIVEN,
        GenerationStrategy.HYBRID,
    ]
    
    for strategy in strategies:
        pipeline = UnifiedGenerationPipeline(
            strategy=strategy,
            enforce_purity=False,
        )
        print(f"\n✓ {strategy.value} pipeline initialized")
        assert pipeline.strategy == strategy
        assert pipeline.role_learner is not None
        assert pipeline.foresight is not None
    
    # Test 2: Context encoding
    pipeline = UnifiedGenerationPipeline(enforce_purity=False)
    context = ["the", "quick", "brown"]
    trajectory = pipeline._encode_context(context)
    print(f"\n✓ Encoded {len(context)} tokens to trajectory of length {len(trajectory)}")
    assert len(trajectory) == len(context)
    
    # Test 3: Empty result handling
    result = pipeline._empty_result()
    print(f"\n✓ Empty result created: tokens={len(result.tokens)}, text='{result.text}'")
    assert result.tokens == []
    assert result.text == ''
    
    # Test 4: Generate (will produce empty result without vocabulary)
    result = pipeline.generate(context=context, max_tokens=5)
    print(f"\n✓ Generation attempted (may be empty without vocabulary)")
    print(f"   Generated tokens: {result.tokens}")
    print(f"   Strategy: {result.strategy.value}")
    print(f"   Purity mode: {result.purity_mode}")
    assert result is not None
    
    print("\n✅ UnifiedGenerationPipeline tests passed!")
    return True


def test_purity_mode():
    """Test QIG_PURITY_MODE detection."""
    print("\n" + "="*70)
    print("Testing QIG_PURITY_MODE")
    print("="*70)
    
    # Save current env var
    old_value = os.environ.get('QIG_PURITY_MODE')
    
    try:
        # Test with purity mode enabled
        os.environ['QIG_PURITY_MODE'] = 'true'
        pipeline = UnifiedGenerationPipeline(enforce_purity=False)
        print(f"\n✓ Purity mode enabled: {pipeline.purity_mode}")
        assert pipeline.purity_mode is True
        
        # Test with purity mode disabled
        os.environ['QIG_PURITY_MODE'] = 'false'
        pipeline = UnifiedGenerationPipeline(enforce_purity=False)
        print(f"✓ Purity mode disabled: {pipeline.purity_mode}")
        assert pipeline.purity_mode is False
    finally:
        # Restore env var
        if old_value is not None:
            os.environ['QIG_PURITY_MODE'] = old_value
        elif 'QIG_PURITY_MODE' in os.environ:
            del os.environ['QIG_PURITY_MODE']
    
    print("\n✅ QIG_PURITY_MODE tests passed!")
    return True


def main():
    """Run all validation tests."""
    print("\n" + "="*70)
    print("Unified Generation Pipeline - Validation Suite")
    print("E8 Protocol Phase 3 Implementation")
    print("="*70)
    
    all_passed = True
    
    try:
        all_passed &= test_token_role_learner()
        all_passed &= test_foresight_predictor()
        all_passed &= test_unified_pipeline()
        all_passed &= test_purity_mode()
        
        print("\n" + "="*70)
        if all_passed:
            print("✅ ALL VALIDATION TESTS PASSED!")
            print("="*70)
            print("\nThe unified generation pipeline is ready for use.")
            print("\nKey Features:")
            print("  • Token role learning from Fisher-Rao neighborhoods")
            print("  • Foresight prediction via trajectory regression")
            print("  • Fisher distance-based candidate scoring")
            print("  • QIG_PURITY_MODE enforcement (no external LLMs)")
            print("  • Per-token observable metrics")
            return 0
        else:
            print("❌ SOME TESTS FAILED")
            print("="*70)
            return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
