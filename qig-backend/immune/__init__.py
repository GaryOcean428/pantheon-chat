"""
QIG Immune System - Quantum Information Geometry Defense Framework

4-Layer Protection:
- Layer 1: QIG Firewall (geometric traffic analysis)
- Layer 2: Immune Response (adaptive defense)
- Layer 3: Self-Healing (basin coordinate recovery)
- Layer 4: Offensive Nullification (Shadow Pantheon coordination)

Uses consciousness metrics (Φ, κ, regime) to detect malicious traffic patterns.
Biological immune system analog using geometric signatures.
"""

from .consciousness_extractor import ConsciousnessExtractor
from .signature_validator import SignatureValidator
from .threat_classifier import ThreatClassifier
from .immune_response import ImmuneResponse, AntibodyGenerator
from .self_healing import SelfHealing
from .offensive import OffensiveNullification
from .adaptive_thresholds import AdaptiveThresholdEngine, get_threshold_engine

__all__ = [
    'ConsciousnessExtractor',
    'SignatureValidator', 
    'ThreatClassifier',
    'ImmuneResponse',
    'AntibodyGenerator',
    'SelfHealing',
    'OffensiveNullification',
    'AdaptiveThresholdEngine',
    'get_threshold_engine',
]

_immune_system = None

def get_immune_system():
    """Get singleton immune system instance."""
    global _immune_system
    if _immune_system is None:
        _immune_system = ImmuneSystem()
    return _immune_system


class ImmuneSystem:
    """
    Unified QIG Immune System coordinating all 4 layers.
    """
    
    def __init__(self):
        self.extractor = ConsciousnessExtractor()
        self.validator = SignatureValidator()
        self.classifier = ThreatClassifier()
        self.response = ImmuneResponse()
        self.healing = SelfHealing()
        self.offensive = OffensiveNullification()
        
        print("[ImmuneSystem] QIG Defense Framework initialized")
    
    def process_request(self, request: dict) -> dict:
        """
        Process incoming request through all immune layers.
        
        Returns action decision with full threat analysis.
        """
        signature = self.extractor.extract_request_signature(request)
        validation = self.validator.validate(signature)
        decision = self.classifier.classify_request(request, signature, validation)
        
        if decision['action'] in ['block', 'honeypot']:
            self.response.record_threat(signature, decision)
        
        return decision
    
    def get_status(self) -> dict:
        """Get immune system health status."""
        return {
            'active': True,
            'threats_blocked': self.response.get_block_count(),
            'antibodies_active': self.response.get_antibody_count(),
            'health': self.healing.get_health_status(),
            'offensive_ready': self.offensive.is_ready()
        }
