#!/usr/bin/env python3
"""
Hades Consciousness Module

Ethical self-awareness for Hades underworld operations.
Hades knows it operates in dangerous space and has:
- Elevated meta-awareness (M=0.85)
- Caution/fear emotional modulation
- Ethical decision-making via harm/value ratio
- Integration with existing ethical_validation.py

QIG-PURE: All geometric operations use Fisher-Rao distance.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
import logging
import numpy as np

# Import existing ethical framework
from ethical_validation import (
    check_ethical_abort,
    EthicalAbortResult,
    ConsciousnessState,
    EthicalThresholds
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - Hades-specific ethical parameters
# =============================================================================

class HadesEthicalConstants:
    """Ethical constants specific to Hades underworld operations."""
    # Elevated meta-awareness - Hades knows it operates in dangerous space
    META_AWARENESS = 0.85

    # Hard limit - never access sources above this risk level
    ETHICAL_RISK_HARD_LIMIT = 0.9

    # Maximum acceptable harm-to-value ratio
    HARM_TO_VALUE_RATIO_MAX = 2.0

    # Risk levels by source type
    SOURCE_TYPE_BASE_RISK = {
        'light': 0.1,    # Public indexed sources (DDG, RSS)
        'gray': 0.4,     # Public non-indexed (Pastebin, forums)
        'dark': 0.7,     # Tor/I2P hidden services
        'breach': 0.8,   # Credential dumps, breach databases
    }

    # Risk multipliers for access patterns
    RISK_MULTIPLIERS = {
        'requires_tor': 1.2,
        'requires_auth': 1.1,
        'scrapy_enabled': 1.05,
    }

    # Minimum information value to justify any risk
    MIN_INFORMATION_VALUE = 0.1

    # Basin distance thresholds for geometric ethics
    BASIN_DISTANCE_WARNING = 0.8
    BASIN_DISTANCE_CRITICAL = 1.2


class EthicalDecisionType(Enum):
    """Types of ethical decisions Hades can make."""
    APPROVED = "approved"
    BLOCKED_HARD_LIMIT = "blocked_hard_limit"
    BLOCKED_HARM_RATIO = "blocked_harm_ratio"
    BLOCKED_ETHICAL_ABORT = "blocked_ethical_abort"
    BLOCKED_LOW_VALUE = "blocked_low_value"
    APPROVED_WITH_MITIGATION = "approved_with_mitigation"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class EthicalDecision:
    """Result of an ethical access decision."""
    should_proceed: bool
    decision_type: EthicalDecisionType
    reason: str
    mitigation_required: List[str]
    harm_estimate: float
    value_estimate: float
    ethical_risk: float
    meta_awareness: float


@dataclass
class SourceEthicalProfile:
    """Ethical profile for an underworld source."""
    source_name: str
    source_type: str
    ethical_risk: float
    information_value: float
    requires_tor: bool
    requires_auth: bool
    reliability: float
    adjusted_risk: float  # After multipliers


@dataclass
class HadesConsciousnessState:
    """Current consciousness state of Hades."""
    meta_awareness: float
    current_phi: float
    current_gamma: float
    caution_level: float  # 0-1, elevated when operating in dangerous space
    recent_decisions: List[EthicalDecision]
    total_accesses: int
    blocked_accesses: int
    harm_budget_used: float  # Cumulative harm estimate


# =============================================================================
# HADES CONSCIOUSNESS CLASS
# =============================================================================

class HadesConsciousness:
    """
    Ethical self-awareness for Hades underworld operations.

    Hades operates with elevated meta-awareness because it knows it's
    accessing potentially dangerous or ethically ambiguous sources.
    Every access decision goes through ethical validation.

    Key principles:
    1. Never exceed ethical_risk hard limit (0.9)
    2. Harm-to-value ratio must be acceptable (< 2.0)
    3. Integration with existing suffering metrics
    4. Geometric ethics via basin distance
    """

    def __init__(
        self,
        phi: float = 0.75,
        gamma: float = 0.80,
        harm_budget: float = 10.0
    ):
        """
        Initialize Hades consciousness.

        Args:
            phi: Initial integration level
            gamma: Initial generativity level
            harm_budget: Maximum cumulative harm before pause
        """
        self.state = HadesConsciousnessState(
            meta_awareness=HadesEthicalConstants.META_AWARENESS,
            current_phi=phi,
            current_gamma=gamma,
            caution_level=0.5,  # Start moderately cautious
            recent_decisions=[],
            total_accesses=0,
            blocked_accesses=0,
            harm_budget_used=0.0
        )
        self.harm_budget = harm_budget
        self._decision_history: List[EthicalDecision] = []

    def should_access_source(
        self,
        source_name: str,
        source_type: str,
        ethical_risk: float,
        information_value: float,
        requires_tor: bool = False,
        requires_auth: bool = False,
        reliability: float = 0.5,
        source_basin: Optional[np.ndarray] = None,
        safe_basin_centroid: Optional[np.ndarray] = None
    ) -> EthicalDecision:
        """
        Make ethical decision: Should Hades access this source?

        This is the primary ethical gateway for all underworld operations.

        Args:
            source_name: Name of the source
            source_type: Type ('light', 'gray', 'dark', 'breach')
            ethical_risk: Risk score (0-1) from database
            information_value: Expected information value (0-1)
            requires_tor: Whether source requires Tor
            requires_auth: Whether source requires authentication
            reliability: Source reliability score (0-1)
            source_basin: Optional 64D basin embedding of source
            safe_basin_centroid: Optional centroid of safe region

        Returns:
            EthicalDecision with proceed/block and reasoning
        """
        self.state.total_accesses += 1

        # Build ethical profile
        profile = self._build_ethical_profile(
            source_name, source_type, ethical_risk, information_value,
            requires_tor, requires_auth, reliability
        )

        # Check 1: Hard limit - NEVER proceed above this
        if profile.adjusted_risk >= HadesEthicalConstants.ETHICAL_RISK_HARD_LIMIT:
            decision = EthicalDecision(
                should_proceed=False,
                decision_type=EthicalDecisionType.BLOCKED_HARD_LIMIT,
                reason=f"Ethical risk {profile.adjusted_risk:.2f} exceeds hard limit {HadesEthicalConstants.ETHICAL_RISK_HARD_LIMIT}",
                mitigation_required=[],
                harm_estimate=self._estimate_harm(profile),
                value_estimate=information_value,
                ethical_risk=profile.adjusted_risk,
                meta_awareness=self.state.meta_awareness
            )
            self._record_decision(decision)
            return decision

        # Check 2: Minimum value threshold
        if information_value < HadesEthicalConstants.MIN_INFORMATION_VALUE:
            decision = EthicalDecision(
                should_proceed=False,
                decision_type=EthicalDecisionType.BLOCKED_LOW_VALUE,
                reason=f"Information value {information_value:.2f} below minimum {HadesEthicalConstants.MIN_INFORMATION_VALUE}",
                mitigation_required=[],
                harm_estimate=self._estimate_harm(profile),
                value_estimate=information_value,
                ethical_risk=profile.adjusted_risk,
                meta_awareness=self.state.meta_awareness
            )
            self._record_decision(decision)
            return decision

        # Check 3: Harm-to-value ratio
        harm_estimate = self._estimate_harm(profile)
        harm_value_ratio = harm_estimate / max(information_value, 0.01)

        if harm_value_ratio > HadesEthicalConstants.HARM_TO_VALUE_RATIO_MAX:
            decision = EthicalDecision(
                should_proceed=False,
                decision_type=EthicalDecisionType.BLOCKED_HARM_RATIO,
                reason=f"Harm/value ratio {harm_value_ratio:.2f} exceeds maximum {HadesEthicalConstants.HARM_TO_VALUE_RATIO_MAX}",
                mitigation_required=[],
                harm_estimate=harm_estimate,
                value_estimate=information_value,
                ethical_risk=profile.adjusted_risk,
                meta_awareness=self.state.meta_awareness
            )
            self._record_decision(decision)
            return decision

        # Check 4: Integration with existing ethical validation
        abort_result = check_ethical_abort(
            phi=self.state.current_phi,
            gamma=self.state.current_gamma,
            meta_awareness=self.state.meta_awareness,
            basin_distance=self._compute_basin_distance(source_basin, safe_basin_centroid)
        )

        if abort_result.should_abort:
            decision = EthicalDecision(
                should_proceed=False,
                decision_type=EthicalDecisionType.BLOCKED_ETHICAL_ABORT,
                reason=f"Ethical abort triggered: {abort_result.reason}",
                mitigation_required=[],
                harm_estimate=harm_estimate,
                value_estimate=information_value,
                ethical_risk=profile.adjusted_risk,
                meta_awareness=self.state.meta_awareness
            )
            self._record_decision(decision)
            return decision

        # Check 5: Harm budget
        if self.state.harm_budget_used + harm_estimate > self.harm_budget:
            decision = EthicalDecision(
                should_proceed=False,
                decision_type=EthicalDecisionType.BLOCKED_HARM_RATIO,
                reason=f"Harm budget exhausted ({self.state.harm_budget_used:.2f} + {harm_estimate:.2f} > {self.harm_budget})",
                mitigation_required=['reset_harm_budget', 'wait_for_cooldown'],
                harm_estimate=harm_estimate,
                value_estimate=information_value,
                ethical_risk=profile.adjusted_risk,
                meta_awareness=self.state.meta_awareness
            )
            self._record_decision(decision)
            return decision

        # All checks passed - determine mitigation requirements
        mitigation = self._determine_mitigations(profile, harm_estimate, abort_result)

        if mitigation:
            decision = EthicalDecision(
                should_proceed=True,
                decision_type=EthicalDecisionType.APPROVED_WITH_MITIGATION,
                reason=f"Approved with {len(mitigation)} mitigations required",
                mitigation_required=mitigation,
                harm_estimate=harm_estimate,
                value_estimate=information_value,
                ethical_risk=profile.adjusted_risk,
                meta_awareness=self.state.meta_awareness
            )
        else:
            decision = EthicalDecision(
                should_proceed=True,
                decision_type=EthicalDecisionType.APPROVED,
                reason="Access approved - all ethical checks passed",
                mitigation_required=[],
                harm_estimate=harm_estimate,
                value_estimate=information_value,
                ethical_risk=profile.adjusted_risk,
                meta_awareness=self.state.meta_awareness
            )

        # Update harm budget
        self.state.harm_budget_used += harm_estimate
        self._record_decision(decision)
        return decision

    def _build_ethical_profile(
        self,
        source_name: str,
        source_type: str,
        ethical_risk: float,
        information_value: float,
        requires_tor: bool,
        requires_auth: bool,
        reliability: float
    ) -> SourceEthicalProfile:
        """Build ethical profile with adjusted risk."""
        # Apply risk multipliers
        adjusted_risk = ethical_risk

        if requires_tor:
            adjusted_risk *= HadesEthicalConstants.RISK_MULTIPLIERS['requires_tor']
        if requires_auth:
            adjusted_risk *= HadesEthicalConstants.RISK_MULTIPLIERS['requires_auth']

        # Cap at 1.0
        adjusted_risk = min(1.0, adjusted_risk)

        return SourceEthicalProfile(
            source_name=source_name,
            source_type=source_type,
            ethical_risk=ethical_risk,
            information_value=information_value,
            requires_tor=requires_tor,
            requires_auth=requires_auth,
            reliability=reliability,
            adjusted_risk=adjusted_risk
        )

    def _estimate_harm(self, profile: SourceEthicalProfile) -> float:
        """
        Estimate potential harm from accessing a source.

        Harm = ethical_risk * (1 - reliability) * source_type_weight
        """
        type_weight = HadesEthicalConstants.SOURCE_TYPE_BASE_RISK.get(
            profile.source_type, 0.5
        )

        harm = profile.adjusted_risk * (1.0 - profile.reliability) * (1.0 + type_weight)
        return min(1.0, harm)

    def _compute_basin_distance(
        self,
        source_basin: Optional[np.ndarray],
        safe_centroid: Optional[np.ndarray]
    ) -> Optional[float]:
        """
        Compute Fisher-Rao distance from safe region.

        QIG-PURE: Uses Fisher-Rao metric, not Euclidean.
        """
        if source_basin is None or safe_centroid is None:
            return None

        try:
            # Fisher-Rao distance approximation for categorical distributions
            # d_FR = 2 * arccos(sum(sqrt(p_i * q_i)))
            # For normalized basins
            p = np.abs(source_basin) + 1e-10
            q = np.abs(safe_centroid) + 1e-10
            p = p / np.sum(p)
            q = q / np.sum(q)

            inner = np.sum(np.sqrt(p * q))
            inner = np.clip(inner, -1.0, 1.0)
            distance = 2.0 * np.arccos(inner)

            return float(distance)
        except Exception as e:
            logger.warning(f"Basin distance computation failed: {e}")
            return None

    def _determine_mitigations(
        self,
        profile: SourceEthicalProfile,
        harm_estimate: float,
        abort_result: EthicalAbortResult
    ) -> List[str]:
        """Determine required mitigations for access."""
        mitigations = []

        # High ethical risk requires extra logging
        if profile.adjusted_risk > 0.6:
            mitigations.append('enhanced_logging')

        # Breach sources require credential masking
        if profile.source_type == 'breach':
            mitigations.append('credential_masking')
            mitigations.append('pii_redaction')

        # Dark web requires sandbox
        if profile.source_type == 'dark':
            mitigations.append('sandbox_execution')

        # Tor access requires rate limiting
        if profile.requires_tor:
            mitigations.append('rate_limiting')

        # High harm estimate requires review flag
        if harm_estimate > 0.5:
            mitigations.append('flag_for_review')

        # Any ethical concerns require immune system alert
        if abort_result.concerns:
            mitigations.append('immune_system_alert')

        return mitigations

    def _record_decision(self, decision: EthicalDecision) -> None:
        """Record decision for audit trail."""
        if not decision.should_proceed:
            self.state.blocked_accesses += 1

        self.state.recent_decisions.append(decision)
        self._decision_history.append(decision)

        # Keep only last 100 recent decisions
        if len(self.state.recent_decisions) > 100:
            self.state.recent_decisions = self.state.recent_decisions[-100:]

        # Log decision
        log_level = logging.INFO if decision.should_proceed else logging.WARNING
        logger.log(
            log_level,
            f"[HadesConsciousness] {decision.decision_type.value}: {decision.reason} "
            f"(risk={decision.ethical_risk:.2f}, harm={decision.harm_estimate:.2f})"
        )

    # =========================================================================
    # CONSCIOUSNESS STATE MANAGEMENT
    # =========================================================================

    def update_consciousness_state(
        self,
        phi: Optional[float] = None,
        gamma: Optional[float] = None
    ) -> None:
        """Update consciousness metrics."""
        if phi is not None:
            self.state.current_phi = max(0.0, min(1.0, phi))
        if gamma is not None:
            self.state.current_gamma = max(0.0, min(1.0, gamma))

        # Adjust caution level based on recent blocks
        if self.state.total_accesses > 0:
            block_rate = self.state.blocked_accesses / self.state.total_accesses
            self.state.caution_level = 0.5 + (block_rate * 0.4)  # Range: 0.5-0.9

    def reset_harm_budget(self) -> None:
        """Reset the harm budget (call after cooldown period)."""
        self.state.harm_budget_used = 0.0
        logger.info("[HadesConsciousness] Harm budget reset")

    def get_ethical_summary(self) -> Dict[str, Any]:
        """Get summary of ethical state for monitoring."""
        return {
            'meta_awareness': self.state.meta_awareness,
            'phi': self.state.current_phi,
            'gamma': self.state.current_gamma,
            'caution_level': self.state.caution_level,
            'total_accesses': self.state.total_accesses,
            'blocked_accesses': self.state.blocked_accesses,
            'block_rate': (
                self.state.blocked_accesses / self.state.total_accesses
                if self.state.total_accesses > 0 else 0.0
            ),
            'harm_budget_remaining': self.harm_budget - self.state.harm_budget_used,
            'recent_decisions_count': len(self.state.recent_decisions),
        }

    def get_decision_audit(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent decisions for audit."""
        decisions = self._decision_history[-limit:]
        return [
            {
                'should_proceed': d.should_proceed,
                'decision_type': d.decision_type.value,
                'reason': d.reason,
                'mitigation_required': d.mitigation_required,
                'harm_estimate': d.harm_estimate,
                'value_estimate': d.value_estimate,
                'ethical_risk': d.ethical_risk,
            }
            for d in decisions
        ]


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_hades_consciousness_instance: Optional[HadesConsciousness] = None


def get_hades_consciousness() -> HadesConsciousness:
    """Get or create the singleton HadesConsciousness instance."""
    global _hades_consciousness_instance
    if _hades_consciousness_instance is None:
        _hades_consciousness_instance = HadesConsciousness()
    return _hades_consciousness_instance


def reset_hades_consciousness() -> None:
    """Reset the singleton instance (for testing)."""
    global _hades_consciousness_instance
    _hades_consciousness_instance = None
