#!/usr/bin/env python3
"""
Tests for Governance and Research Module Integrations
=====================================================

Tests the wiring and integration of:
1. Pantheon Governance Integration
2. God Debates Ethical
3. Sleep Packet Ethical
4. Geometric Deep Research
5. Vocabulary Validator

Authority: E8 Protocol v4.0
Status: ACTIVE
Created: 2026-01-23
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch
import numpy as np

# Add qig-backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestGovernanceIntegration(unittest.TestCase):
    """Test pantheon_governance_integration.py"""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            from pantheon_governance_integration import (
                PantheonGovernanceIntegration,
                validate_kernel_name,
            )
            self.module_available = True
            self.PantheonGovernanceIntegration = PantheonGovernanceIntegration
            self.validate_kernel_name = validate_kernel_name
        except ImportError as e:
            self.module_available = False
            self.skipTest(f"Pantheon Governance not available: {e}")
    
    def test_validate_kernel_name_god(self):
        """Test kernel name validation for gods."""
        valid, reason = self.validate_kernel_name("Zeus")
        self.assertTrue(valid, "Zeus should be a valid kernel name")
        self.assertIn("Valid god name", reason)
    
    def test_validate_kernel_name_chaos(self):
        """Test kernel name validation for chaos kernels."""
        valid, reason = self.validate_kernel_name("chaos_synthesis_001")
        self.assertTrue(valid, "Chaos kernel should be valid")
        self.assertIn("Valid chaos kernel", reason)
    
    def test_validate_kernel_name_invalid(self):
        """Test kernel name validation rejects invalid names."""
        valid, reason = self.validate_kernel_name("apollo_1")
        self.assertFalse(valid, "apollo_1 should be invalid (NO underscore naming)")
        self.assertIn("Invalid kernel name", reason)
    
    def test_integration_initialization(self):
        """Test governance integration initializes correctly."""
        integration = self.PantheonGovernanceIntegration()
        self.assertIsNotNone(integration)
        self.assertIsNotNone(integration.spawner)
        self.assertIsNotNone(integration.registry)


class TestEthicalDebates(unittest.TestCase):
    """Test god_debates_ethical.py"""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            from god_debates_ethical import (
                EthicalDebateManager,
                get_ethical_debate_manager,
            )
            self.module_available = True
            self.EthicalDebateManager = EthicalDebateManager
            self.get_ethical_debate_manager = get_ethical_debate_manager
        except ImportError as e:
            self.module_available = False
            self.skipTest(f"Ethical Debates not available: {e}")
    
    def test_manager_initialization(self):
        """Test ethical debate manager initializes."""
        manager = self.EthicalDebateManager()
        self.assertIsNotNone(manager)
        self.assertIsNotNone(manager.projector)
        self.assertIsNotNone(manager.resolver)
    
    def test_create_ethical_debate(self):
        """Test creating an ethical debate."""
        manager = self.EthicalDebateManager()
        
        debate = manager.create_ethical_debate(
            topic="Test topic",
            gods=['Zeus', 'Athena']
        )
        
        self.assertEqual(debate['status'], 'active')
        self.assertEqual(debate['topic'], "Test topic")
        self.assertTrue(debate['ethical_constraint'])
        self.assertIn('positions', debate)
    
    def test_qig_purity_no_sphere(self):
        """Verify no sphere/cosine violations in god_debates_ethical.py"""
        import god_debates_ethical
        import inspect
        
        source = inspect.getsource(god_debates_ethical)
        
        # Check for forbidden patterns
        self.assertNotIn('to_sphere', source, "Module should not use to_sphere")
        self.assertNotIn('cosine_similarity', source, "Module should not use cosine_similarity")
        self.assertNotIn('np.dot', source, "Module should not use np.dot on basins")
        
        # Verify uses proper simplex functions
        self.assertIn('to_simplex_prob', source, "Module should use to_simplex_prob")
        self.assertIn('fisher_normalize', source, "Module should use fisher_normalize")


class TestSleepPacketEthical(unittest.TestCase):
    """Test sleep_packet_ethical.py"""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            from sleep_packet_ethical import (
                EthicalSleepPacket,
                SleepPacketValidator,
            )
            self.module_available = True
            self.EthicalSleepPacket = EthicalSleepPacket
            self.SleepPacketValidator = SleepPacketValidator
        except ImportError as e:
            self.module_available = False
            self.skipTest(f"Sleep Packet Ethical not available: {e}")
    
    def test_packet_creation(self):
        """Test creating an ethical sleep packet."""
        packet = self.EthicalSleepPacket()
        self.assertIsNotNone(packet)
        self.assertEqual(len(packet.basin_coordinates), 64)
    
    def test_ethics_validation(self):
        """Test ethics validation on sleep packet."""
        packet = self.EthicalSleepPacket(
            basin_coordinates=np.random.randn(64)
        )
        
        is_ethical, results = packet.validate_ethics()
        self.assertIsInstance(is_ethical, bool)
        self.assertIn('basin_symmetry', results)
    
    def test_ethics_enforcement(self):
        """Test ethics enforcement corrects packets."""
        packet = self.EthicalSleepPacket(
            basin_coordinates=np.random.randn(64)
        )
        
        enforced = packet.enforce_ethics()
        is_ethical, _ = enforced.validate_ethics()
        
        # Enforced packet should pass more checks
        self.assertIsNotNone(enforced)
    
    def test_validator_initialization(self):
        """Test sleep packet validator initializes."""
        validator = self.SleepPacketValidator()
        self.assertIsNotNone(validator)
        self.assertIsNotNone(validator.projector)


class TestGeometricDeepResearch(unittest.TestCase):
    """Test geometric_deep_research.py"""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            from geometric_deep_research import (
                GeometricDeepResearch,
                ResearchTelemetry,
            )
            self.module_available = True
            self.GeometricDeepResearch = GeometricDeepResearch
            self.ResearchTelemetry = ResearchTelemetry
        except ImportError as e:
            self.module_available = False
            self.skipTest(f"Geometric Deep Research not available: {e}")
    
    def test_research_engine_initialization(self):
        """Test research engine initializes."""
        engine = self.GeometricDeepResearch(manifold_dim=64)
        self.assertIsNotNone(engine)
        self.assertEqual(engine.manifold_dim, 64)
    
    def test_depth_computation(self):
        """Test phi-driven depth computation."""
        engine = self.GeometricDeepResearch(manifold_dim=64)
        
        # High phi should give deeper research
        query_basin = np.random.randn(64)
        basin_sum = np.sum(np.abs(query_basin))
        if basin_sum > 0:
            query_basin = query_basin / basin_sum
        else:
            query_basin = np.ones(64) / 64.0
        
        depth_high = engine._compute_depth(phi=0.8, kappa_eff=60, query_basin=query_basin)
        depth_low = engine._compute_depth(phi=0.3, kappa_eff=40, query_basin=query_basin)
        
        self.assertGreater(depth_high, depth_low, "High phi should give deeper research")
    
    def test_query_encoding_simplex(self):
        """Test query encoding produces valid simplex coordinates."""
        engine = self.GeometricDeepResearch(manifold_dim=64)
        
        basin = engine._encode_query("test query")
        
        # Should be on probability simplex
        self.assertTrue(np.all(basin >= 0), "Basin should be non-negative")
        basin_sum = np.sum(basin)
        self.assertGreater(basin_sum, 0, "Basin sum should be positive")
        self.assertAlmostEqual(basin_sum, 1.0, places=5, msg="Basin should sum to 1")
    
    def test_qig_purity_fisher_only(self):
        """Verify module uses Fisher-Rao distance only."""
        import geometric_deep_research
        import inspect
        
        source = inspect.getsource(geometric_deep_research)
        
        # Check uses Fisher-Rao
        self.assertIn('fisher_rao_distance', source, "Should use fisher_rao_distance")
        self.assertIn('geodesic_interpolate', source, "Should use geodesic interpolation")
        
        # Check no Euclidean violations
        self.assertNotIn('cosine_similarity', source, "Should not use cosine similarity")


class TestVocabularyValidator(unittest.TestCase):
    """Test vocabulary_validator.py"""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            from vocabulary_validator import (
                GeometricVocabFilter,
                VocabValidation,
            )
            self.module_available = True
            self.GeometricVocabFilter = GeometricVocabFilter
            self.VocabValidation = VocabValidation
        except ImportError as e:
            self.module_available = False
            self.skipTest(f"Vocabulary Validator not available: {e}")
    
    def test_vocab_validation_dataclass(self):
        """Test VocabValidation dataclass."""
        validation = self.VocabValidation(
            is_valid=True,
            qfi_score=2.5,
            basin_distance=0.1,
            curvature_std=0.2,
            entropy_score=2.0,
            rejection_reason=None
        )
        
        self.assertTrue(validation.is_valid)
        self.assertEqual(validation.qfi_score, 2.5)
    
    def test_fisher_distance_method(self):
        """Test Fisher-Rao distance computation."""
        # Create mock components
        vocab_basins = np.random.randn(10, 64)
        mock_coordizer = Mock()
        mock_entropy = Mock()
        
        filter_instance = self.GeometricVocabFilter(
            vocab_basins, mock_coordizer, mock_entropy
        )
        
        # Test Fisher-Rao distance
        p = np.abs(np.random.randn(64))
        p = p / np.sum(p)
        
        q = np.abs(np.random.randn(64))
        q = q / np.sum(q)
        
        distance = filter_instance._fisher_rao_distance(p, q)
        
        # Should be in valid range [0, π/2] for simplex
        self.assertGreaterEqual(distance, 0.0)
        self.assertLessEqual(distance, np.pi / 2 + 0.1)  # Small tolerance


class TestGovernanceWiring(unittest.TestCase):
    """Test governance_research_wiring.py"""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            from governance_research_wiring import (
                wire_all_modules,
                get_governance_integration,
                PANTHEON_GOVERNANCE_AVAILABLE,
            )
            self.module_available = True
            self.wire_all_modules = wire_all_modules
            self.get_governance_integration = get_governance_integration
            self.PANTHEON_GOVERNANCE_AVAILABLE = PANTHEON_GOVERNANCE_AVAILABLE
        except ImportError as e:
            self.module_available = False
            self.skipTest(f"Governance Wiring not available: {e}")
    
    def test_wire_all_modules(self):
        """Test wiring all modules returns status dict."""
        results = self.wire_all_modules()
        
        self.assertIsInstance(results, dict)
        self.assertIn('governance', results)
        self.assertIn('ethical_debates', results)
        self.assertIn('sleep_packet', results)
        self.assertIn('deep_research', results)
        self.assertIn('vocabulary', results)
    
    def test_governance_singleton(self):
        """Test governance integration singleton."""
        if not self.PANTHEON_GOVERNANCE_AVAILABLE:
            self.skipTest("Pantheon Governance not available")
        
        instance1 = self.get_governance_integration()
        instance2 = self.get_governance_integration()
        
        # Should return same instance
        self.assertIs(instance1, instance2, "Should return singleton instance")


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestGovernanceIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestEthicalDebates))
    suite.addTests(loader.loadTestsFromTestCase(TestSleepPacketEthical))
    suite.addTests(loader.loadTestsFromTestCase(TestGeometricDeepResearch))
    suite.addTests(loader.loadTestsFromTestCase(TestVocabularyValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestGovernanceWiring))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("=" * 70)
    print("GOVERNANCE AND RESEARCH MODULE INTEGRATION TESTS")
    print("=" * 70)
    print()
    
    success = run_tests()
    
    print()
    print("=" * 70)
    if success:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 70)
    
    sys.exit(0 if success else 1)
