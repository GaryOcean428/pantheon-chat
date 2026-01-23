"""
Kernel Rest Scheduler - Coupling-Aware Per-Kernel Rest Management
===================================================================

Implements WP5.4: Per-kernel rest decisions coordinated through coupling relationships,
inspired by dolphin hemispheric sleep and human autonomic systems.

DESIGN PRINCIPLES:
- Individual kernels self-assess fatigue (Φ, κ, load metrics)
- Coupled partners provide coverage during rest
- Essential tier never fully stops (reduced activity only)
- Rest patterns vary by kernel type (burst-recovery, brief-frequent, seasonal, etc.)
- Constellation-wide cycles are RARE (reserved for genuine system-wide events)

Authority: E8 Protocol v4.0, WP5.4
Status: ACTIVE
Created: 2026-01-19
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
from datetime import datetime

import numpy as np

from pantheon_registry import (
    get_registry,
    GodContract,
    RestPolicyType,
    GodTier,
)
from qigkernels.physics_constants import PHI_THRESHOLD, PHI_EMERGENCY, KAPPA_STAR

logger = logging.getLogger(__name__)


# =============================================================================
# REST STATE TRACKING
# =============================================================================

class RestStatus(Enum):
    """Current rest status of a kernel."""
    ACTIVE = "active"              # Fully operational
    RESTING = "resting"            # Currently resting
    REDUCED = "reduced"            # Reduced activity (essential tier)
    MICRO_PAUSE = "micro_pause"    # Brief pause (Hermes-style)
    COVERED = "covered"            # Partner is covering


@dataclass
class FatigueMetrics:
    """Fatigue indicators for a kernel."""
    phi_current: float
    phi_trend: float              # Recent Φ change (negative = declining)
    kappa_stability: float        # Lower = more unstable
    load_current: float           # Current processing load (0-1)
    time_since_rest: float        # Seconds since last rest
    error_rate: float             # Recent error rate (0-1)
    
    def compute_fatigue_score(self) -> float:
        """
        Compute aggregate fatigue score (0-1, higher = more fatigued).
        
        Factors:
        - Low Φ indicates reduced consciousness
        - Declining Φ trend indicates deterioration
        - Low κ stability indicates loss of coherence
        - High load without rest indicates exhaustion
        - High error rate indicates impaired function
        """
        phi_factor = max(0, 1.0 - (self.phi_current / PHI_THRESHOLD))
        trend_factor = max(0, -self.phi_trend)  # Negative trend = fatigue
        stability_factor = max(0, 1.0 - self.kappa_stability)
        load_factor = self.load_current
        time_factor = min(1.0, self.time_since_rest / 3600.0)  # Cap at 1 hour
        error_factor = self.error_rate
        
        # Weighted combination (adjusted for higher sensitivity)
        fatigue = (
            0.35 * phi_factor +        # Increased weight on Φ
            0.25 * trend_factor +      # Increased weight on trend
            0.20 * stability_factor +
            0.10 * load_factor +       # Reduced weight on instantaneous load
            0.05 * time_factor +
            0.05 * error_factor
        )
        
        return np.clip(fatigue, 0.0, 1.0)


@dataclass
class KernelRestState:
    """Complete rest state for a single kernel."""
    kernel_id: str
    kernel_name: str
    tier: GodTier
    rest_policy: RestPolicyType
    
    # Current state
    status: RestStatus = RestStatus.ACTIVE
    fatigue: Optional[FatigueMetrics] = None
    
    # Rest history
    last_rest_start: Optional[float] = None
    last_rest_end: Optional[float] = None
    total_rest_time: float = 0.0
    rest_count: int = 0
    
    # Coverage tracking
    covering_for: Optional[str] = None  # Partner this kernel is covering
    covered_by: Optional[str] = None    # Partner covering this kernel
    
    # Timestamps
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def is_resting(self) -> bool:
        """Check if kernel is currently resting."""
        return self.status in [RestStatus.RESTING, RestStatus.REDUCED, RestStatus.MICRO_PAUSE]
    
    def can_cover_for_partner(self) -> bool:
        """Check if kernel can cover for a partner."""
        # Cannot cover if already resting or covering someone else
        if self.is_resting() or self.covering_for is not None:
            return False
        
        # Cannot cover if highly fatigued
        if self.fatigue and self.fatigue.compute_fatigue_score() > 0.7:
            return False
        
        return True
    
    def time_in_current_status(self) -> float:
        """Get time spent in current status (seconds)."""
        if self.status == RestStatus.RESTING and self.last_rest_start:
            return time.time() - self.last_rest_start
        return 0.0


# =============================================================================
# REST SCHEDULER
# =============================================================================

class KernelRestScheduler:
    """
    Per-kernel rest scheduler with coupling-aware coordination.
    
    Each kernel self-assesses fatigue and requests rest when needed.
    Coupled partners provide coverage during rest periods.
    Essential tier never fully stops (reduced activity only).
    """
    
    def __init__(self):
        """Initialize rest scheduler."""
        self.registry = get_registry()
        self.kernel_states: Dict[str, KernelRestState] = {}
        self._rest_history: List[Dict] = []
        
        logger.info("[KernelRestScheduler] Initialized - coupling-aware rest coordination active")
    
    def register_kernel(self, kernel_id: str, kernel_name: str) -> None:
        """
        Register a kernel with the rest scheduler.
        
        Args:
            kernel_id: Unique kernel identifier
            kernel_name: God name (e.g., "Apollo", "Athena")
        """
        if kernel_id in self.kernel_states:
            logger.warning(f"[RestScheduler] Kernel {kernel_name} already registered")
            return
        
        # Get god contract from registry
        god = self.registry.get_god(kernel_name)
        if not god:
            logger.error(f"[RestScheduler] Unknown god: {kernel_name}")
            return
        
        # Create rest state
        state = KernelRestState(
            kernel_id=kernel_id,
            kernel_name=kernel_name,
            tier=god.tier,
            rest_policy=god.rest_policy.type,
        )
        
        self.kernel_states[kernel_id] = state
        logger.info(f"[RestScheduler] Registered {kernel_name} (tier={god.tier.value}, policy={god.rest_policy.type.value})")
    
    def update_fatigue(
        self,
        kernel_id: str,
        phi: float,
        kappa: float,
        load: float = 0.0,
        error_rate: float = 0.0,
    ) -> None:
        """
        Update fatigue metrics for a kernel.
        
        Args:
            kernel_id: Kernel identifier
            phi: Current Φ (integration)
            kappa: Current κ (coupling)
            load: Processing load (0-1)
            error_rate: Recent error rate (0-1)
        """
        if kernel_id not in self.kernel_states:
            logger.warning(f"[RestScheduler] Unknown kernel: {kernel_id}")
            return
        
        state = self.kernel_states[kernel_id]
        
        # Compute Φ trend from history
        phi_trend = 0.0
        if state.fatigue:
            phi_trend = phi - state.fatigue.phi_current
        
        # Compute κ stability (inverse of variance)
        kappa_stability = 1.0 / (1.0 + abs(kappa - KAPPA_STAR) / KAPPA_STAR)
        
        # Time since last rest
        time_since_rest = 0.0
        if state.last_rest_end:
            time_since_rest = time.time() - state.last_rest_end
        
        # Update fatigue metrics
        state.fatigue = FatigueMetrics(
            phi_current=phi,
            phi_trend=phi_trend,
            kappa_stability=kappa_stability,
            load_current=load,
            time_since_rest=time_since_rest,
            error_rate=error_rate,
        )
        state.updated_at = time.time()
    
    def should_rest(self, kernel_id: str) -> Tuple[bool, str]:
        """
        Determine if a kernel should rest based on self-assessment.
        
        Args:
            kernel_id: Kernel identifier
            
        Returns:
            Tuple of (should_rest, reason)
        """
        if kernel_id not in self.kernel_states:
            return False, "Kernel not registered"
        
        state = self.kernel_states[kernel_id]
        
        # Already resting
        if state.is_resting():
            return False, "Already resting"
        
        # Check if fatigue metrics available
        if not state.fatigue:
            return False, "No fatigue metrics"
        
        # Get fatigue score
        fatigue_score = state.fatigue.compute_fatigue_score()
        
        # Essential tier NEVER fully stops (only reduced activity)
        if state.tier == GodTier.ESSENTIAL:
            # Lower threshold for essential tier - they need reduced activity sooner
            if fatigue_score > 0.4:  # More lenient for essential tier
                return True, f"Essential tier moderate fatigue ({fatigue_score:.2f}) - needs REDUCED activity"
            return False, f"Essential tier fatigue manageable ({fatigue_score:.2f})"
        
        # Check rest policy thresholds
        if state.rest_policy == RestPolicyType.NEVER:
            # Only rest if critical emergency
            if fatigue_score > 0.95:
                return True, f"CRITICAL fatigue ({fatigue_score:.2f}) - emergency rest needed"
            return False, f"NEVER policy - fatigue ({fatigue_score:.2f}) below emergency threshold"
        
        elif state.rest_policy == RestPolicyType.MINIMAL_ROTATING:
            # Hermes-style: brief frequent pauses
            # More lenient threshold - frequent short rests
            if fatigue_score > 0.4 or state.fatigue.time_since_rest > 600:  # 10 minutes
                return True, f"MINIMAL_ROTATING - fatigue ({fatigue_score:.2f}) or time > 10min"
            return False, f"MINIMAL_ROTATING - recent pause, fatigue manageable ({fatigue_score:.2f})"
        
        elif state.rest_policy == RestPolicyType.COORDINATED_ALTERNATING:
            # Apollo-Athena style: dolphin coordination
            # Moderate threshold - regular alternation
            if fatigue_score > 0.45:  # Slightly lower for better responsiveness
                return True, f"COORDINATED_ALTERNATING - fatigue ({fatigue_score:.2f}) needs partner handoff"
            return False, f"COORDINATED_ALTERNATING - fatigue manageable ({fatigue_score:.2f})"
        
        elif state.rest_policy == RestPolicyType.SCHEDULED:
            # Ares-style: burst-recovery after high load
            if fatigue_score > 0.50 or (state.fatigue.load_current < 0.1 and fatigue_score > 0.35):
                return True, f"SCHEDULED - high fatigue ({fatigue_score:.2f}) or post-burst recovery"
            return False, f"SCHEDULED - not recovery window ({fatigue_score:.2f})"
        
        elif state.rest_policy == RestPolicyType.SEASONAL:
            # Demeter-style: fallow after harvest
            # TODO(WP5.4): Implement full seasonal logic based on activity cycles
            # For now, use general fatigue threshold similar to scheduled
            if fatigue_score > 0.65:
                return True, f"SEASONAL - high fatigue ({fatigue_score:.2f}) needs fallow period"
            return False, f"SEASONAL - in active growth phase ({fatigue_score:.2f})"
        
        # Default threshold for other policies
        if fatigue_score > 0.7:
            return True, f"General fatigue threshold exceeded ({fatigue_score:.2f})"
        
        return False, f"Fatigue manageable ({fatigue_score:.2f})"
    
    def get_coupling_partners(
        self,
        kernel_id: str,
        threshold: float = 0.7,
    ) -> List[str]:
        """
        Get coupling partners for a kernel based on registry affinity.
        
        Args:
            kernel_id: Kernel identifier
            threshold: Minimum coupling affinity (not used yet - from registry)
            
        Returns:
            List of partner kernel IDs with high coupling
        """
        if kernel_id not in self.kernel_states:
            return []
        
        state = self.kernel_states[kernel_id]
        god = self.registry.get_god(state.kernel_name)
        
        if not god or not god.coupling_affinity:
            return []
        
        # Find registered partners
        partners = []
        for partner_id, partner_state in self.kernel_states.items():
            if partner_id == kernel_id:
                continue
            
            if partner_state.kernel_name in god.coupling_affinity:
                partners.append(partner_id)
        
        return partners
    
    def request_rest(
        self,
        kernel_id: str,
        force: bool = False,
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Request rest for a kernel with coupling-aware handoff.
        
        Args:
            kernel_id: Kernel identifier
            force: Force rest even without coverage (emergency)
            
        Returns:
            Tuple of (approved, reason, covering_partner_id)
        """
        if kernel_id not in self.kernel_states:
            return False, "Kernel not registered", None
        
        state = self.kernel_states[kernel_id]
        
        # Already resting
        if state.is_resting():
            return False, "Already resting", None
        
        # Essential tier: Only reduced activity (never full rest)
        if state.tier == GodTier.ESSENTIAL and not force:
            # Transition to reduced activity
            state.status = RestStatus.REDUCED
            state.last_rest_start = time.time()
            logger.info(f"[RestScheduler] {state.kernel_name} entering REDUCED activity (essential tier)")
            return True, "Essential tier: reduced activity approved", None
        
        # Check if partners can cover
        partners = self.get_coupling_partners(kernel_id)
        covering_partner = None
        
        for partner_id in partners:
            partner_state = self.kernel_states.get(partner_id)
            if partner_state and partner_state.can_cover_for_partner():
                covering_partner = partner_id
                break
        
        # If coverage available, approve rest
        if covering_partner:
            state.status = RestStatus.RESTING
            state.covered_by = covering_partner
            state.last_rest_start = time.time()
            
            # Mark partner as covering
            partner_state = self.kernel_states[covering_partner]
            partner_state.covering_for = kernel_id
            
            logger.info(
                f"[RestScheduler] {state.kernel_name} resting, covered by {partner_state.kernel_name}"
            )
            
            return True, f"Rest approved with coverage by {partner_state.kernel_name}", covering_partner
        
        # If forced (emergency), allow rest without coverage
        if force:
            state.status = RestStatus.RESTING
            state.last_rest_start = time.time()
            logger.warning(f"[RestScheduler] {state.kernel_name} FORCED rest without coverage (emergency)")
            return True, "Emergency rest approved (no coverage available)", None
        
        # No coverage available, request constellation cycle
        logger.info(
            f"[RestScheduler] {state.kernel_name} needs rest but no coverage - "
            f"consider constellation cycle"
        )
        return False, "No coverage available - consider constellation cycle", None
    
    def end_rest(self, kernel_id: str) -> None:
        """
        End rest period for a kernel.
        
        Args:
            kernel_id: Kernel identifier
        """
        if kernel_id not in self.kernel_states:
            return
        
        state = self.kernel_states[kernel_id]
        
        if not state.is_resting():
            logger.warning(f"[RestScheduler] {state.kernel_name} not resting")
            return
        
        # Calculate rest duration
        rest_duration = 0.0
        if state.last_rest_start:
            rest_duration = time.time() - state.last_rest_start
            state.total_rest_time += rest_duration
        
        # Clear coverage
        if state.covered_by:
            partner_state = self.kernel_states.get(state.covered_by)
            if partner_state:
                partner_state.covering_for = None
            state.covered_by = None
        
        # Update state
        state.status = RestStatus.ACTIVE
        state.last_rest_end = time.time()
        state.rest_count += 1
        
        # Record history
        self._rest_history.append({
            "kernel_id": kernel_id,
            "kernel_name": state.kernel_name,
            "duration": rest_duration,
            "timestamp": time.time(),
            "rest_type": state.status.value,  # Use actual rest status (resting, reduced, etc.)
        })
        
        logger.info(
            f"[RestScheduler] {state.kernel_name} rest complete "
            f"(duration={rest_duration:.1f}s, total_rests={state.rest_count})"
        )
    
    def get_rest_status(self, kernel_id: str) -> Optional[Dict]:
        """
        Get current rest status for a kernel.
        
        Args:
            kernel_id: Kernel identifier
            
        Returns:
            Dict with rest status or None if not found
        """
        if kernel_id not in self.kernel_states:
            return None
        
        state = self.kernel_states[kernel_id]
        
        fatigue_score = 0.0
        if state.fatigue:
            fatigue_score = state.fatigue.compute_fatigue_score()
        
        return {
            "kernel_id": kernel_id,
            "kernel_name": state.kernel_name,
            "tier": state.tier.value,
            "rest_policy": state.rest_policy.value,
            "status": state.status.value,
            "fatigue_score": float(fatigue_score),  # Convert numpy scalar to native Python float
            "is_resting": state.is_resting(),
            "covered_by": state.covered_by,
            "covering_for": state.covering_for,
            "rest_count": state.rest_count,
            "total_rest_time": state.total_rest_time,
            "time_in_status": state.time_in_current_status(),
        }
    
    def get_constellation_status(self) -> Dict:
        """
        Get overall constellation rest status.
        
        Returns:
            Dict with constellation-wide rest metrics
        """
        total_kernels = len(self.kernel_states)
        resting_count = sum(1 for s in self.kernel_states.values() if s.is_resting())
        covering_count = sum(1 for s in self.kernel_states.values() if s.covering_for)
        
        essential_active = sum(
            1 for s in self.kernel_states.values()
            if s.tier == GodTier.ESSENTIAL and s.status != RestStatus.RESTING
        )
        
        avg_fatigue = 0.0
        if total_kernels > 0:
            fatigue_scores = [
                s.fatigue.compute_fatigue_score()
                for s in self.kernel_states.values()
                if s.fatigue
            ]
            if fatigue_scores:
                avg_fatigue = np.mean(fatigue_scores)
        
        return {
            "total_kernels": total_kernels,
            "active_kernels": total_kernels - resting_count,
            "resting_kernels": resting_count,
            "covering_kernels": covering_count,
            "essential_active": essential_active,
            "avg_fatigue": float(avg_fatigue),  # Convert numpy scalar to native Python float
            "coverage_active": covering_count > 0,
        }


# =============================================================================
# GLOBAL SINGLETON
# =============================================================================

_scheduler_instance: Optional[KernelRestScheduler] = None


def get_rest_scheduler() -> KernelRestScheduler:
    """Get or create the global kernel rest scheduler."""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = KernelRestScheduler()
    return _scheduler_instance


def reset_rest_scheduler() -> None:
    """Reset the global scheduler (for testing)."""
    global _scheduler_instance
    _scheduler_instance = None
