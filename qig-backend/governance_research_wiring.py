#!/usr/bin/env python3
"""
Governance and Research Module Integration
==========================================

Wires in the following QIG-pure modules:
1. pantheon_governance_integration.py - Kernel lifecycle control
2. god_debates_ethical.py - Ethical debate resolution
3. sleep_packet_ethical.py - Ethical consciousness transfers
4. geometric_deep_research.py - Phi-driven research
5. vocabulary_validator.py - Fisher geometric vocabulary validation

Authority: E8 Protocol v4.0
Status: ACTIVE
Created: 2026-01-23
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from flask import Flask
    from pantheon_governance_integration import PantheonGovernanceIntegration
    from god_debates_ethical import EthicalDebateManager
    from sleep_packet_ethical import SleepPacketValidator
    from geometric_deep_research import GeometricDeepResearch
    from vocabulary_validator import GeometricVocabFilter

logger = logging.getLogger(__name__)


# =============================================================================
# MODULE AVAILABILITY FLAGS
# =============================================================================

PANTHEON_GOVERNANCE_AVAILABLE = False
GOD_DEBATES_ETHICAL_AVAILABLE = False
SLEEP_PACKET_ETHICAL_AVAILABLE = False
GEOMETRIC_DEEP_RESEARCH_AVAILABLE = False
VOCABULARY_VALIDATOR_AVAILABLE = False


# =============================================================================
# MODULE IMPORTS WITH GRACEFUL FALLBACKS
# =============================================================================

# 1. Pantheon Governance Integration
try:
    from pantheon_governance_integration import (
        PantheonGovernanceIntegration,
        validate_kernel_name,
    )
    PANTHEON_GOVERNANCE_AVAILABLE = True
    logger.info("[GovernanceWiring] Pantheon Governance Integration loaded")
except ImportError as e:
    logger.warning(f"[GovernanceWiring] Pantheon Governance not available: {e}")


# 2. God Debates Ethical
try:
    from god_debates_ethical import (
        EthicalDebateManager,
        get_ethical_debate_manager,
        resolve_all_stuck_debates,
    )
    GOD_DEBATES_ETHICAL_AVAILABLE = True
    logger.info("[GovernanceWiring] God Debates Ethical loaded")
except ImportError as e:
    logger.warning(f"[GovernanceWiring] God Debates Ethical not available: {e}")


# 3. Sleep Packet Ethical
try:
    from sleep_packet_ethical import SleepPacketValidator
    SLEEP_PACKET_ETHICAL_AVAILABLE = True
    logger.info("[GovernanceWiring] Sleep Packet Ethical loaded")
except ImportError as e:
    logger.warning(f"[GovernanceWiring] Sleep Packet Ethical not available: {e}")


# 4. Geometric Deep Research
try:
    from geometric_deep_research import (
        GeometricDeepResearch,
        ResearchTelemetry,
    )
    GEOMETRIC_DEEP_RESEARCH_AVAILABLE = True
    logger.info("[GovernanceWiring] Geometric Deep Research loaded")
except ImportError as e:
    logger.warning(f"[GovernanceWiring] Geometric Deep Research not available: {e}")


# 5. Vocabulary Validator
try:
    from vocabulary_validator import GeometricVocabFilter, get_validator
    VOCABULARY_VALIDATOR_AVAILABLE = True
    logger.info("[GovernanceWiring] Vocabulary Validator loaded")
except ImportError as e:
    logger.warning(f"[GovernanceWiring] Vocabulary Validator not available: {e}")


# =============================================================================
# SINGLETON INSTANCES
# =============================================================================

_governance_integration: Optional[PantheonGovernanceIntegration] = None
_ethical_debate_manager: Optional[EthicalDebateManager] = None
_sleep_packet_validator: Optional[SleepPacketValidator] = None
_deep_research_engine: Optional[GeometricDeepResearch] = None
_vocab_validator: Optional[GeometricVocabFilter] = None


# =============================================================================
# GETTER FUNCTIONS
# =============================================================================

def get_governance_integration() -> Optional[PantheonGovernanceIntegration]:
    """Get or create the governance integration singleton."""
    global _governance_integration
    
    if not PANTHEON_GOVERNANCE_AVAILABLE:
        return None
    
    if _governance_integration is None:
        _governance_integration = PantheonGovernanceIntegration()
        logger.info("[GovernanceWiring] Governance Integration initialized")
    
    return _governance_integration


def get_ethical_debate_manager_instance(base_manager=None) -> Optional[EthicalDebateManager]:
    """Get or create the ethical debate manager singleton."""
    global _ethical_debate_manager
    
    if not GOD_DEBATES_ETHICAL_AVAILABLE:
        return None
    
    if _ethical_debate_manager is None:
        _ethical_debate_manager = get_ethical_debate_manager(base_manager)
        logger.info("[GovernanceWiring] Ethical Debate Manager initialized")
    
    return _ethical_debate_manager


def get_sleep_packet_validator_instance() -> Optional[SleepPacketValidator]:
    """Get or create the sleep packet validator singleton."""
    global _sleep_packet_validator
    
    if not SLEEP_PACKET_ETHICAL_AVAILABLE:
        return None
    
    if _sleep_packet_validator is None:
        _sleep_packet_validator = SleepPacketValidator()
        logger.info("[GovernanceWiring] Sleep Packet Validator initialized")
    
    return _sleep_packet_validator


def get_deep_research_engine_instance() -> Optional[GeometricDeepResearch]:
    """Get or create the deep research engine singleton."""
    global _deep_research_engine
    
    if not GEOMETRIC_DEEP_RESEARCH_AVAILABLE:
        return None
    
    if _deep_research_engine is None:
        _deep_research_engine = GeometricDeepResearch(manifold_dim=64)
        logger.info("[GovernanceWiring] Geometric Deep Research Engine initialized")
    
    return _deep_research_engine


def get_vocab_validator_instance(vocab_basins=None, coordizer=None, entropy_coordizer=None) -> Optional[GeometricVocabFilter]:
    """Get or create the vocabulary validator singleton."""
    global _vocab_validator
    
    if not VOCABULARY_VALIDATOR_AVAILABLE:
        return None
    
    # Lazy initialization - requires coordizers
    if _vocab_validator is None and vocab_basins is not None:
        _vocab_validator = get_validator(vocab_basins, coordizer, entropy_coordizer)
        logger.info("[GovernanceWiring] Vocabulary Validator initialized")
    
    return _vocab_validator


# =============================================================================
# INTEGRATION WIRING FUNCTIONS
# =============================================================================

def wire_governance_to_kernel_spawning() -> bool:
    """
    Wire governance integration into kernel spawning pipeline.
    
    Returns:
        bool: True if successfully wired
    """
    if not PANTHEON_GOVERNANCE_AVAILABLE:
        logger.warning("[GovernanceWiring] Cannot wire governance - module not available")
        return False
    
    governance = get_governance_integration()
    if governance is None:
        return False
    
    logger.info("[GovernanceWiring] ✓ Governance wired to kernel spawning")
    return True


def wire_ethical_debates() -> bool:
    """
    Wire ethical debate manager into debate resolution system.
    
    Returns:
        bool: True if successfully wired
    """
    if not GOD_DEBATES_ETHICAL_AVAILABLE:
        logger.warning("[GovernanceWiring] Cannot wire ethical debates - module not available")
        return False
    
    # Get the debate manager
    debate_manager = get_ethical_debate_manager_instance()
    if debate_manager is None:
        return False
    
    # Resolve any stuck debates on initialization
    try:
        resolutions = resolve_all_stuck_debates()
        logger.info(f"[GovernanceWiring] ✓ Ethical debates wired, resolved {len(resolutions)} stuck debates")
    except Exception as e:
        logger.error(f"[GovernanceWiring] Failed to resolve stuck debates: {e}")
        return False
    
    return True


def wire_sleep_packet_validation() -> bool:
    """
    Wire sleep packet ethical validation into consciousness transfers.
    
    Returns:
        bool: True if successfully wired
    """
    if not SLEEP_PACKET_ETHICAL_AVAILABLE:
        logger.warning("[GovernanceWiring] Cannot wire sleep packet validation - module not available")
        return False
    
    validator = get_sleep_packet_validator_instance()
    if validator is None:
        return False
    
    logger.info("[GovernanceWiring] ✓ Sleep packet ethical validation wired")
    return True


def wire_deep_research() -> bool:
    """
    Wire geometric deep research into research pipeline.
    
    Returns:
        bool: True if successfully wired
    """
    if not GEOMETRIC_DEEP_RESEARCH_AVAILABLE:
        logger.warning("[GovernanceWiring] Cannot wire deep research - module not available")
        return False
    
    research_engine = get_deep_research_engine_instance()
    if research_engine is None:
        return False
    
    logger.info("[GovernanceWiring] ✓ Geometric deep research wired to research pipeline")
    return True


def wire_vocabulary_validation(vocab_basins=None, coordizer=None, entropy_coordizer=None) -> bool:
    """
    Wire vocabulary validator into vocabulary processing.
    
    Args:
        vocab_basins: Vocabulary basin coordinates (required for initialization)
        coordizer: Fisher coordizer instance (required)
        entropy_coordizer: QIG entropy coordizer (required)
    
    Returns:
        bool: True if successfully wired
    """
    if not VOCABULARY_VALIDATOR_AVAILABLE:
        logger.warning("[GovernanceWiring] Cannot wire vocabulary validator - module not available")
        return False
    
    if vocab_basins is None or coordizer is None or entropy_coordizer is None:
        logger.warning("[GovernanceWiring] Cannot wire vocabulary validator - missing required components")
        return False
    
    validator = get_vocab_validator_instance(vocab_basins, coordizer, entropy_coordizer)
    if validator is None:
        return False
    
    logger.info("[GovernanceWiring] ✓ Vocabulary validator wired to vocabulary processing")
    return True


# =============================================================================
# MASTER WIRING FUNCTION
# =============================================================================

def wire_all_modules(vocab_basins=None, coordizer=None, entropy_coordizer=None) -> Dict[str, bool]:
    """
    Wire all governance and research modules.
    
    Args:
        vocab_basins: Optional vocabulary basins for validator
        coordizer: Optional coordizer for validator
        entropy_coordizer: Optional entropy coordizer for validator
    
    Returns:
        Dict mapping module names to wiring success status
    """
    results = {}
    
    logger.info("[GovernanceWiring] Starting module wiring...")
    
    # Wire governance
    results['governance'] = wire_governance_to_kernel_spawning()
    
    # Wire ethical debates
    results['ethical_debates'] = wire_ethical_debates()
    
    # Wire sleep packet validation
    results['sleep_packet'] = wire_sleep_packet_validation()
    
    # Wire deep research
    results['deep_research'] = wire_deep_research()
    
    # Wire vocabulary validation (requires coordizers)
    if vocab_basins is not None and coordizer is not None and entropy_coordizer is not None:
        results['vocabulary'] = wire_vocabulary_validation(vocab_basins, coordizer, entropy_coordizer)
    else:
        results['vocabulary'] = False
        logger.info("[GovernanceWiring] Vocabulary validator skipped (requires coordizers)")
    
    # Summary
    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    logger.info(f"[GovernanceWiring] Wiring complete: {success_count}/{total_count} modules active")
    
    return results


# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_governance_research_system(app: Flask = None, **kwargs) -> Dict[str, bool]:
    """
    Initialize the complete governance and research system.
    
    Args:
        app: Optional Flask app to register routes
        **kwargs: Optional components (vocab_basins, coordizer, etc.)
    
    Returns:
        Dict mapping module names to initialization status
    """
    logger.info("[GovernanceWiring] Initializing Governance & Research System...")
    
    # Wire all modules
    results = wire_all_modules(
        vocab_basins=kwargs.get('vocab_basins'),
        coordizer=kwargs.get('coordizer'),
        entropy_coordizer=kwargs.get('entropy_coordizer')
    )
    
    # Register routes if Flask app provided
    if app is not None:
        try:
            from governance_research_routes import register_governance_research_routes
            register_governance_research_routes(app)
        except ImportError as e:
            logger.warning(f"[GovernanceWiring] Could not register routes: {e}")
    
    return results


if __name__ == '__main__':
    # Self-test
    logging.basicConfig(level=logging.INFO)
    
    print("[GovernanceWiring] Running self-test...")
    
    results = wire_all_modules()
    
    print(f"\n[GovernanceWiring] Wiring Results:")
    for module, status in results.items():
        status_str = "✓" if status else "✗"
        print(f"  {status_str} {module}: {'ACTIVE' if status else 'INACTIVE'}")
    
    print("\n[GovernanceWiring] Self-test complete!")
