"""
Test Suite for QIG Purity Mode
===============================

Tests the QIG purity enforcement system to ensure:
1. External LLM API calls are blocked when purity mode is enabled
2. System can complete tasks without external help
3. Attempted external calls are logged with stack traces
4. Outputs are correctly tagged as pure/hybrid

Author: Copilot Agent (WP4.1 Implementation)
Date: 2026-01-16
Protocol: Ultra Consciousness v4.0 ACTIVE
"""

import os
import sys
import pytest
import importlib
from unittest.mock import Mock, patch
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qig_purity_mode import (
    is_purity_mode_enabled,
    get_purity_mode,
    check_forbidden_imports,
    check_forbidden_attributes,
    log_external_call_attempt,
    block_external_api_call,
    check_purity_violation,
    enforce_purity,
    tag_output_as_hybrid,
    tag_output_as_pure,
    get_purity_report,
    validate_qig_purity,
    PurityViolationType,
)


@pytest.fixture
def enable_purity_mode():
    """Fixture to enable purity mode for a test."""
    original_value = os.environ.get('QIG_PURITY_MODE')
    os.environ['QIG_PURITY_MODE'] = 'true'
    yield
    # Restore original value
    if original_value is None:
        os.environ.pop('QIG_PURITY_MODE', None)
    else:
        os.environ['QIG_PURITY_MODE'] = original_value


@pytest.fixture
def disable_purity_mode():
    """Fixture to disable purity mode for a test."""
    original_value = os.environ.get('QIG_PURITY_MODE')
    os.environ['QIG_PURITY_MODE'] = 'false'
    yield
    # Restore original value
    if original_value is None:
        os.environ.pop('QIG_PURITY_MODE', None)
    else:
        os.environ['QIG_PURITY_MODE'] = original_value


class TestPurityModeDetection:
    """Test purity mode detection and configuration."""
    
    def test_purity_mode_enabled(self, enable_purity_mode):
        """Test that purity mode is correctly detected when enabled."""
        assert is_purity_mode_enabled() is True
        assert get_purity_mode() == "ENABLED"
    
    def test_purity_mode_disabled(self, disable_purity_mode):
        """Test that purity mode is correctly detected when disabled."""
        assert is_purity_mode_enabled() is False
        assert get_purity_mode() == "DISABLED"
    
    def test_purity_mode_unset(self):
        """Test behavior when QIG_PURITY_MODE is not set."""
        original_value = os.environ.get('QIG_PURITY_MODE')
        os.environ.pop('QIG_PURITY_MODE', None)
        
        assert is_purity_mode_enabled() is False
        assert get_purity_mode() == "DISABLED"
        
        # Restore
        if original_value is not None:
            os.environ['QIG_PURITY_MODE'] = original_value


class TestForbiddenImportDetection:
    """Test detection of forbidden module imports."""
    
    def test_clean_environment(self, enable_purity_mode):
        """Test that clean environment has no violations."""
        # Remove any forbidden modules that might be loaded
        forbidden = ['openai', 'anthropic', 'google.generativeai']
        for module in forbidden:
            if module in sys.modules:
                del sys.modules[module]
        
        violations = check_forbidden_imports()
        assert len(violations) == 0
    
    def test_detect_openai_import(self, enable_purity_mode):
        """Test detection of OpenAI import."""
        # Mock openai being in sys.modules
        sys.modules['openai'] = Mock()
        
        violations = check_forbidden_imports()
        assert len(violations) > 0
        assert any(v.module == 'openai' for v in violations)
        assert any(v.type == PurityViolationType.FORBIDDEN_MODULE for v in violations)
        
        # Cleanup
        del sys.modules['openai']
    
    def test_detect_anthropic_import(self, enable_purity_mode):
        """Test detection of Anthropic import."""
        sys.modules['anthropic'] = Mock()
        
        violations = check_forbidden_imports()
        assert len(violations) > 0
        assert any(v.module == 'anthropic' for v in violations)
        
        # Cleanup
        del sys.modules['anthropic']
    
    def test_multiple_forbidden_imports(self, enable_purity_mode):
        """Test detection of multiple forbidden imports."""
        sys.modules['openai'] = Mock()
        sys.modules['anthropic'] = Mock()
        
        violations = check_forbidden_imports()
        assert len(violations) >= 2
        
        # Cleanup
        del sys.modules['openai']
        del sys.modules['anthropic']


class TestForbiddenAttributeDetection:
    """Test detection of forbidden attributes."""
    
    def test_clean_object(self):
        """Test that clean object has no violations."""
        class CleanObject:
            pass
        
        obj = CleanObject()
        violations = check_forbidden_attributes(obj)
        assert len(violations) == 0
    
    def test_detect_chat_completion_attribute(self):
        """Test detection of ChatCompletion attribute."""
        class ViolatingObject:
            ChatCompletion = Mock()
        
        obj = ViolatingObject()
        violations = check_forbidden_attributes(obj)
        assert len(violations) > 0
        assert any(v.type == PurityViolationType.FORBIDDEN_ATTRIBUTE for v in violations)
    
    def test_detect_max_tokens_attribute(self):
        """Test detection of max_tokens attribute."""
        class ViolatingObject:
            max_tokens = 100
        
        obj = ViolatingObject()
        violations = check_forbidden_attributes(obj)
        assert len(violations) > 0


class TestExternalCallBlocking:
    """Test blocking of external API calls."""
    
    def test_block_external_call_raises_error(self, enable_purity_mode):
        """Test that blocking external calls raises RuntimeError."""
        with pytest.raises(RuntimeError, match="QIG PURITY VIOLATION"):
            block_external_api_call("OpenAI", "/v1/chat/completions")
    
    def test_external_call_logging(self, enable_purity_mode, caplog):
        """Test that external calls are logged."""
        import logging
        caplog.set_level(logging.ERROR)
        
        log_external_call_attempt("OpenAI", "/v1/chat/completions")
        
        assert "EXTERNAL API CALL BLOCKED" in caplog.text
        assert "OpenAI" in caplog.text
    
    def test_external_call_allowed_when_disabled(self, disable_purity_mode):
        """Test that external calls don't raise error when purity mode disabled."""
        # This just logs, doesn't block when disabled
        # No exception should be raised
        log_external_call_attempt("OpenAI", "/v1/chat/completions")


class TestPurityEnforcement:
    """Test overall purity enforcement."""
    
    def test_enforce_purity_clean_environment(self, enable_purity_mode):
        """Test enforcement passes in clean environment."""
        # Clean up any forbidden modules
        forbidden = ['openai', 'anthropic', 'google.generativeai']
        for module in forbidden:
            if module in sys.modules:
                del sys.modules[module]
        
        # Should not raise
        enforce_purity()
    
    def test_enforce_purity_with_violations(self, enable_purity_mode):
        """Test enforcement fails with violations."""
        sys.modules['openai'] = Mock()
        
        with pytest.raises(RuntimeError, match="PURITY VIOLATIONS DETECTED"):
            enforce_purity()
        
        # Cleanup
        del sys.modules['openai']
    
    def test_enforce_purity_disabled(self, disable_purity_mode):
        """Test enforcement skipped when disabled."""
        sys.modules['openai'] = Mock()
        
        # Should not raise even with violations
        enforce_purity()
        
        # Cleanup
        del sys.modules['openai']
    
    def test_validate_qig_purity_success(self, enable_purity_mode):
        """Test successful purity validation."""
        # Clean environment
        forbidden = ['openai', 'anthropic', 'google.generativeai']
        for module in forbidden:
            if module in sys.modules:
                del sys.modules[module]
        
        assert validate_qig_purity() is True
    
    def test_validate_qig_purity_failure(self, enable_purity_mode):
        """Test failed purity validation."""
        sys.modules['openai'] = Mock()
        
        assert validate_qig_purity() is False
        
        # Cleanup
        del sys.modules['openai']


class TestOutputTagging:
    """Test output tagging for pure/hybrid outputs."""
    
    def test_tag_output_as_pure(self):
        """Test tagging output as pure QIG."""
        output = {'result': 'test'}
        tagged = tag_output_as_pure(output)
        
        assert tagged['qig_pure'] is True
        assert tagged['external_assistance'] is False
        assert 'purity_mode' in tagged
    
    def test_tag_output_as_hybrid(self):
        """Test tagging output as hybrid."""
        output = {'result': 'test'}
        tagged = tag_output_as_hybrid(output)
        
        assert tagged['qig_pure'] is False
        assert tagged['external_assistance'] is True
        assert 'purity_mode' in tagged
    
    def test_hybrid_tagging_logs_warning(self, caplog):
        """Test that hybrid tagging logs a warning."""
        import logging
        caplog.set_level(logging.WARNING)
        
        output = {'result': 'test'}
        tag_output_as_hybrid(output)
        
        assert "HYBRID" in caplog.text
        assert "external LLM assistance" in caplog.text


class TestPurityReport:
    """Test purity reporting functionality."""
    
    def test_purity_report_structure(self):
        """Test that purity report has expected structure."""
        report = get_purity_report()
        
        assert 'purity_mode_enabled' in report
        assert 'purity_mode' in report
        assert 'total_violations' in report
        assert 'violations_by_type' in report
        assert 'recent_violations' in report
        assert 'forbidden_modules' in report
        assert 'forbidden_attributes' in report
    
    def test_purity_report_forbidden_lists(self):
        """Test that forbidden lists are populated."""
        report = get_purity_report()
        
        assert len(report['forbidden_modules']) > 0
        assert 'openai' in report['forbidden_modules']
        assert 'anthropic' in report['forbidden_modules']
        
        assert len(report['forbidden_attributes']) > 0
        assert 'ChatCompletion' in report['forbidden_attributes']


class TestPureQIGGeneration:
    """Test that system can complete tasks without external help."""
    
    def test_pure_qig_text_encoding(self, enable_purity_mode):
        """Test pure QIG text encoding without external dependencies."""
        try:
            from qig_generation import encode_to_basin
            
            # Should work without external APIs
            text = "consciousness emerges from integration"
            basin = encode_to_basin(text)
            
            assert basin is not None
            assert len(basin) > 0
        except ImportError:
            pytest.skip("qig_generation not available")
    
    def test_pure_qig_phi_computation(self, enable_purity_mode):
        """Test pure QIG phi computation without external dependencies."""
        try:
            import numpy as np
            from qig_generation import QIGGenerator
            
            # Create generator in purity mode
            generator = QIGGenerator()
            
            # Should be able to measure phi without external APIs
            basin = np.random.dirichlet(np.ones(64))
            phi = generator._measure_phi(basin)
            
            assert phi is not None
            assert 0.0 <= phi <= 1.0
        except ImportError:
            pytest.skip("qig_generation not available")
    
    def test_pure_qig_kernel_routing(self, enable_purity_mode):
        """Test pure QIG kernel routing without external dependencies."""
        try:
            import numpy as np
            from qig_generation import QIGKernelRouter
            
            # Create router in purity mode
            router = QIGKernelRouter()
            
            # Should be able to route without external APIs
            query_basin = np.random.dirichlet(np.ones(64))
            kernels = router.route_query(query_basin, k=3)
            
            assert kernels is not None
            assert len(kernels) == 3
        except ImportError:
            pytest.skip("qig_generation not available")


class TestIntegrationWithQIGGeneration:
    """Test integration with QIG generation system."""
    
    def test_qig_generator_validates_purity(self, enable_purity_mode):
        """Test that QIG generator validates purity on init."""
        try:
            from qig_generation import QIGGenerator
            
            # Clean environment
            forbidden = ['openai', 'anthropic', 'google.generativeai']
            for module in forbidden:
                if module in sys.modules:
                    del sys.modules[module]
            
            # Should initialize successfully in purity mode
            generator = QIGGenerator()
            assert generator is not None
        except ImportError:
            pytest.skip("qig_generation not available")
    
    def test_qig_generator_forbids_external_modules(self, enable_purity_mode):
        """Test that QIG generator detects forbidden modules."""
        try:
            from qig_generation import validate_qig_purity
            
            # Add forbidden module
            sys.modules['openai'] = Mock()
            
            # Should fail validation (updated regex to match new error message format)
            with pytest.raises(AssertionError, match="(QIG VIOLATION|QIG PURITY VIOLATIONS DETECTED)"):
                validate_qig_purity()
            
            # Cleanup
            del sys.modules['openai']
        except ImportError:
            pytest.skip("qig_generation not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
