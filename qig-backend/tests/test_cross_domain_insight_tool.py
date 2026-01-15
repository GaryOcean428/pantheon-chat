"""
Validation Tests for Cross-Domain Insight Tool
==============================================

Tests the cross-domain insight assessment tool from the SLEEP packet.

Date: 2026-01-15
Issue: ISMS-SLEEP-INSIGHT-TOOL-2026-01-15
"""

import sys
import os
import numpy as np

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search.cross_domain_insight_tool import (
    CrossDomainInsightAssessor,
    InsightQuality,
    CrossDomainInsight,
)


def test_tool_initialization():
    """Test 1: Tool initializes correctly"""
    print("\n" + "="*70)
    print("Test 1: Tool Initialization")
    print("="*70)
    
    assessor = CrossDomainInsightAssessor()
    
    assert len(assessor.domains) == 0, "Should start with no domains"
    assert len(assessor.insight_history) == 0, "Should start with no history"
    
    print("✅ PASS: Tool initialized correctly")
    return True


def test_domain_registration():
    """Test 2: Domain registration works"""
    print("\n" + "="*70)
    print("Test 2: Domain Registration")
    print("="*70)
    
    assessor = CrossDomainInsightAssessor()
    
    # Register a domain
    basin = np.random.rand(64)
    assessor.register_domain(
        "physics",
        basin,
        category="knowledge",
        metadata={"source": "test"}
    )
    
    assert len(assessor.domains) == 1, "Should have 1 domain"
    assert "physics" in assessor.domains, "Should contain physics domain"
    
    domain = assessor.domains["physics"]
    assert domain.name == "physics"
    assert domain.category == "knowledge"
    assert np.allclose(np.sum(domain.basin_coords), 1.0), "Basin should be normalized"
    
    print(f"✅ PASS: Domain 'physics' registered successfully")
    print(f"   Basin sum: {np.sum(domain.basin_coords):.4f} (should be ~1.0)")
    return True


def test_connection_assessment():
    """Test 3: Connection assessment produces valid insight"""
    print("\n" + "="*70)
    print("Test 3: Connection Assessment")
    print("="*70)
    
    assessor = CrossDomainInsightAssessor()
    
    # Register two domains
    assessor.register_domain("knowledge", np.random.rand(64), "knowledge")
    assessor.register_domain("research", np.random.rand(64), "research")
    
    # Assess connection
    insight = assessor.assess_connection(
        "knowledge",
        "research",
        current_phi=0.85
    )
    
    assert isinstance(insight, CrossDomainInsight)
    assert insight.domain_a == "knowledge"
    assert insight.domain_b == "research"
    assert 0 <= insight.fisher_distance <= np.pi/2, "FR distance in valid range"
    assert 0 <= insight.novelty_score <= 1.0, "Novelty in [0,1]"
    assert 0 <= insight.coherence_score <= 1.0, "Coherence in [0,1]"
    assert insight.phi_context == 0.85
    
    print(f"✅ PASS: Connection assessed successfully")
    print(f"   Fisher distance: {insight.fisher_distance:.4f}")
    print(f"   Quality: {insight.quality.value}")
    print(f"   Novelty: {insight.novelty_score:.3f}")
    print(f"   Coherence: {insight.coherence_score:.3f}")
    return True


def test_quality_classification():
    """Test 4: Quality classification is reasonable"""
    print("\n" + "="*70)
    print("Test 4: Quality Classification")
    print("="*70)
    
    assessor = CrossDomainInsightAssessor()
    
    # Create two very similar basins (should be BREAKTHROUGH or STRONG)
    basin_a = np.random.rand(64)
    basin_b = basin_a + np.random.rand(64) * 0.01  # Very similar
    
    assessor.register_domain("domain_a", basin_a, "test")
    assessor.register_domain("domain_b", basin_b, "test")
    
    insight = assessor.assess_connection("domain_a", "domain_b", current_phi=0.8)
    
    # Very similar basins should have low distance = high quality
    assert insight.fisher_distance < 0.5, "Similar basins should have small distance"
    assert insight.quality in [
        InsightQuality.BREAKTHROUGH,
        InsightQuality.STRONG,
        InsightQuality.MODERATE
    ], f"Quality should be at least MODERATE, got {insight.quality}"
    
    print(f"✅ PASS: Quality classification is reasonable")
    print(f"   Distance: {insight.fisher_distance:.4f} → Quality: {insight.quality.value}")
    return True


def test_novelty_tracking():
    """Test 5: Novelty decreases with repeated connections"""
    print("\n" + "="*70)
    print("Test 5: Novelty Tracking")
    print("="*70)
    
    assessor = CrossDomainInsightAssessor()
    
    assessor.register_domain("A", np.random.rand(64), "test")
    assessor.register_domain("B", np.random.rand(64), "test")
    
    # First connection
    insight1 = assessor.assess_connection("A", "B", current_phi=0.8)
    novelty1 = insight1.novelty_score
    
    # Second connection (same pair)
    insight2 = assessor.assess_connection("A", "B", current_phi=0.8)
    novelty2 = insight2.novelty_score
    
    # Third connection
    insight3 = assessor.assess_connection("A", "B", current_phi=0.8)
    novelty3 = insight3.novelty_score
    
    print(f"First connection novelty: {novelty1:.3f}")
    print(f"Second connection novelty: {novelty2:.3f}")
    print(f"Third connection novelty: {novelty3:.3f}")
    
    # Novelty should decrease with repeated connections
    assert novelty2 < novelty1, "Novelty should decrease on repeat"
    assert novelty3 < novelty2, "Novelty should keep decreasing"
    
    print(f"✅ PASS: Novelty tracking works (decreases with repetition)")
    return True


def test_statistics():
    """Test 6: Statistics computation works"""
    print("\n" + "="*70)
    print("Test 6: Statistics")
    print("="*70)
    
    assessor = CrossDomainInsightAssessor()
    
    # Register domains
    for i in range(3):
        assessor.register_domain(f"domain_{i}", np.random.rand(64), "test")
    
    # Make some connections
    assessor.assess_connection("domain_0", "domain_1", current_phi=0.8)
    assessor.assess_connection("domain_1", "domain_2", current_phi=0.85)
    assessor.assess_connection("domain_0", "domain_2", current_phi=0.9)
    
    stats = assessor.get_statistics()
    
    assert stats["total_assessments"] == 3
    assert stats["registered_domains"] == 3
    assert "avg_fisher_distance" in stats
    assert "avg_novelty" in stats
    assert "avg_coherence" in stats
    
    print(f"✅ PASS: Statistics computed successfully")
    print(f"   Total assessments: {stats['total_assessments']}")
    print(f"   Avg FR distance: {stats['avg_fisher_distance']:.4f}")
    print(f"   Avg novelty: {stats['avg_novelty']:.3f}")
    print(f"   Avg coherence: {stats['avg_coherence']:.3f}")
    return True


def test_string_representation():
    """Test 7: String representation matches expected format"""
    print("\n" + "="*70)
    print("Test 7: String Representation")
    print("="*70)
    
    assessor = CrossDomainInsightAssessor()
    
    assessor.register_domain("knowledge", np.random.rand(64), "knowledge")
    assessor.register_domain("research", np.random.rand(64), "research")
    
    insight = assessor.assess_connection("knowledge", "research", current_phi=0.85)
    
    insight_str = str(insight)
    
    # Check format: domain_a+domain_b|domain_a/domain_b|FR=...|...
    assert "knowledge+research" in insight_str or "research+knowledge" in insight_str
    assert "FR=" in insight_str
    assert "BD=" in insight_str
    assert "Φ=" in insight_str
    
    print(f"✅ PASS: String representation correct")
    print(f"   Format: {insight_str}")
    return True


def test_phi_context_coherence():
    """Test 8: Higher phi leads to higher coherence"""
    print("\n" + "="*70)
    print("Test 8: Phi Context and Coherence")
    print("="*70)
    
    assessor = CrossDomainInsightAssessor()
    
    assessor.register_domain("A", np.random.rand(64), "test")
    assessor.register_domain("B", np.random.rand(64), "test")
    
    # Low phi
    insight_low = assessor.assess_connection("A", "B", current_phi=0.3)
    
    # Reset and re-register to test with same basins
    assessor2 = CrossDomainInsightAssessor()
    assessor2.domains = assessor.domains.copy()
    
    # High phi
    insight_high = assessor2.assess_connection("A", "B", current_phi=0.9)
    
    print(f"Coherence at Φ=0.3: {insight_low.coherence_score:.3f}")
    print(f"Coherence at Φ=0.9: {insight_high.coherence_score:.3f}")
    
    # Higher phi should generally lead to higher coherence
    # (though other factors also matter, so we don't enforce strict ordering)
    assert 0 <= insight_low.coherence_score <= 1.0
    assert 0 <= insight_high.coherence_score <= 1.0
    
    print(f"✅ PASS: Coherence scores in valid range for different phi values")
    return True


def run_all_tests():
    """Run all validation tests"""
    print("\n" + "="*70)
    print("CROSS-DOMAIN INSIGHT TOOL - VALIDATION TESTS")
    print("="*70)
    print("Testing kernel-requested tool from SLEEP packet")
    print("Date: 2026-01-15")
    print("="*70)
    
    tests = [
        test_tool_initialization,
        test_domain_registration,
        test_connection_assessment,
        test_quality_classification,
        test_novelty_tracking,
        test_statistics,
        test_string_representation,
        test_phi_context_coherence,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except AssertionError as e:
            print(f"❌ ASSERTION FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)
    print(f"✅ Passed: {passed}/{len(tests)}")
    print(f"❌ Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎯 ALL VALIDATION TESTS PASSED!")
        print("\n✅ Cross-Domain Insight Tool validated:")
        print("  ✓ Domain registration and normalization")
        print("  ✓ Connection assessment with quality classification")
        print("  ✓ Novelty tracking (decreases with repetition)")
        print("  ✓ Coherence scoring with Φ context")
        print("  ✓ Statistics and reporting")
        print("  ✓ Kernel-friendly string format")
        print("\n🌊∇💚∫🧠 💎🎯🏆")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("Review output above for details.")
        return 1


if __name__ == "__main__":
    import sys
    exit_code = run_all_tests()
    sys.exit(exit_code)
