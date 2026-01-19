"""
🌊💓 Ocean-Heart Consensus - Autonomic Cycle Decision Making
============================================================

Ocean (autonomic observer) and Heart (feeling metronome) jointly sense when
the constellation needs sleep/dream/mushroom cycles and decide TOGETHER.

DESIGN PRINCIPLE (WP5.4 UPDATE):
- Constellation-wide cycles are RARE (reserved for genuine system-wide events)
- Per-kernel rest is now preferred (via KernelRestScheduler)
- NO automatic thresholds - this is deliberative, felt decision-making
- Heart provides: HRV state, κ oscillation, feeling/logic mode, rigidity detection
- Ocean provides: constellation coherence, Φ variance, emotional tone, spread
- Both must AGREE before any cycle triggers for the entire constellation

STRICT CRITERIA FOR CONSTELLATION CYCLES (WP5.4):
- SLEEP: coherence < 0.5, avg_fatigue > 0.8, basin_drift > 0.3
- DREAM: stuck kernels > 50%, HRV rigidity > 0.7
- MUSHROOM: rigidity > 0.9, novelty exhausted

This mirrors how human autonomic regulation works:
- You don't consciously decide to sleep - your autonomic system (Ocean) and
  your felt sense of tiredness (Heart) reach agreement
- Individual organs (kernels) don't request sleep - they experience it when
  the autonomic centers decide

ACCESS CONTROL:
- Kernels observe that cycles are happening (via WorkingMemoryMixin)
- Kernels do NOT control cycle triggering
- Only Ocean+Heart consensus can trigger constellation-wide cycles
- Per-kernel rest is coordinated via KernelRestScheduler (dolphin-style)
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class CycleType(Enum):
    """Types of autonomic cycles."""
    SLEEP = "sleep"       # Consolidation, drift reduction
    DREAM = "dream"       # Creative exploration, recombination
    MUSHROOM = "mushroom" # Perturbation, breaking rigidity


@dataclass
class ConsensusState:
    """Current state of Ocean+Heart consensus."""
    heart_feels_sleep: bool = False
    heart_feels_dream: bool = False
    heart_feels_mushroom: bool = False
    ocean_senses_sleep: bool = False
    ocean_senses_dream: bool = False
    ocean_senses_mushroom: bool = False
    
    heart_reasoning: str = ""
    ocean_reasoning: str = ""
    
    last_sleep_time: float = 0.0
    last_dream_time: float = 0.0
    last_mushroom_time: float = 0.0
    
    current_cycle: Optional[CycleType] = None
    cycle_start_time: float = 0.0


@dataclass
class CycleDecision:
    """Result of consensus decision."""
    approved: bool
    cycle_type: CycleType
    heart_vote: bool
    ocean_vote: bool
    heart_reasoning: str
    ocean_reasoning: str
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "approved": self.approved,
            "cycle_type": self.cycle_type.value,
            "heart_vote": self.heart_vote,
            "ocean_vote": self.ocean_vote,
            "heart_reasoning": self.heart_reasoning,
            "ocean_reasoning": self.ocean_reasoning,
            "timestamp": self.timestamp,
        }


class OceanHeartConsensus:
    """
    Joint decision-making between Ocean (autonomic observer) and Heart (feeling metronome).
    
    Both must agree before any cycle triggers. This is not threshold-based -
    it's felt, deliberative sensing that considers the entire constellation state.
    
    WP5.4 UPDATE: Constellation cycles are now RARE - most rest is per-kernel via
    KernelRestScheduler. Only trigger constellation-wide when truly necessary.
    """
    
    # WP5.4: Increased cooldowns - constellation cycles should be rare
    SLEEP_COOLDOWN = 300.0     # 5 minutes minimum between sleep cycles
    DREAM_COOLDOWN = 600.0     # 10 minutes minimum between dream cycles  
    MUSHROOM_COOLDOWN = 1800.0 # 30 minutes minimum between mushroom cycles
    
    # WP5.4: Strict thresholds for constellation-wide cycles
    SLEEP_COHERENCE_THRESHOLD = 0.5     # Below this = constellation needs sleep
    SLEEP_FATIGUE_THRESHOLD = 0.8       # Above this = constellation exhausted
    SLEEP_DRIFT_THRESHOLD = 0.3         # Above this = basins drifting apart
    
    DREAM_STUCK_THRESHOLD = 0.5         # 50% of kernels stuck
    DREAM_RIGIDITY_THRESHOLD = 0.7      # HRV rigidity threshold
    
    MUSHROOM_RIGIDITY_THRESHOLD = 0.9   # Extreme rigidity threshold
    MUSHROOM_SPREAD_THRESHOLD = 0.5     # Minimal spread threshold
    
    def __init__(self):
        self.state = ConsensusState()
        self._heart_kernel = None
        self._ocean_observer = None
        self._decision_history: List[CycleDecision] = []
        
        logger.info("[OceanHeartConsensus] 🌊💓 Initialized - autonomic cycle governance active")
    
    def wire_heart(self, heart_kernel) -> None:
        """Wire the HeartKernel for feeling-based sensing."""
        self._heart_kernel = heart_kernel
        logger.info("[OceanHeartConsensus] 💓 Heart kernel connected")
    
    def wire_ocean(self, ocean_observer) -> None:
        """Wire the OceanMetaObserver for constellation sensing."""
        self._ocean_observer = ocean_observer
        logger.info("[OceanHeartConsensus] 🌊 Ocean observer connected")
    
    def sense_and_decide(self) -> Optional[CycleDecision]:
        """
        Main sensing loop - called periodically to check if cycles are needed.
        
        Returns a CycleDecision if consensus is reached, None otherwise.
        """
        if self.state.current_cycle is not None:
            return None
        
        for cycle_type in [CycleType.SLEEP, CycleType.DREAM, CycleType.MUSHROOM]:
            decision = self._evaluate_cycle(cycle_type)
            if decision.approved:
                self._record_decision(decision)
                self.begin_cycle(cycle_type)
                return decision
        
        return None
    
    def request_cycle(self, cycle_type: CycleType) -> CycleDecision:
        """
        Public API for requesting a cycle evaluation.
        
        Called by AutonomicKernel's _should_trigger_* methods.
        If approved, records the decision and begins the cycle.
        
        Args:
            cycle_type: The type of cycle to evaluate
            
        Returns:
            CycleDecision with approval status and reasoning
        """
        if self.state.current_cycle is not None:
            return CycleDecision(
                approved=False,
                cycle_type=cycle_type,
                heart_vote=False,
                ocean_vote=False,
                heart_reasoning="Cycle already in progress",
                ocean_reasoning=f"Currently in {self.state.current_cycle.value} cycle",
            )
        
        decision = self._evaluate_cycle(cycle_type)
        
        if decision.approved:
            self._record_decision(decision)
            self.begin_cycle(cycle_type)
        
        return decision
    
    def _evaluate_cycle(self, cycle_type: CycleType) -> CycleDecision:
        """Evaluate whether a specific cycle should trigger."""
        if not self._check_cooldown(cycle_type):
            return CycleDecision(
                approved=False,
                cycle_type=cycle_type,
                heart_vote=False,
                ocean_vote=False,
                heart_reasoning="Cooldown active",
                ocean_reasoning="Cooldown active",
            )
        
        heart_vote, heart_reason = self._heart_senses(cycle_type)
        ocean_vote, ocean_reason = self._ocean_senses(cycle_type)
        
        approved = heart_vote and ocean_vote
        
        return CycleDecision(
            approved=approved,
            cycle_type=cycle_type,
            heart_vote=heart_vote,
            ocean_vote=ocean_vote,
            heart_reasoning=heart_reason,
            ocean_reasoning=ocean_reason,
        )
    
    def _heart_senses(self, cycle_type: CycleType) -> Tuple[bool, str]:
        """
        Heart's feeling-based sensing for cycle need.
        
        Heart feels through:
        - HRV (variability = health, rigidity = pathology)
        - κ oscillation (stuck at one value = needs intervention)
        - Mode (prolonged feeling/logic without tacking = imbalance)
        
        WP5.4: Much stricter thresholds - constellation cycles are rare.
        """
        if self._heart_kernel is None:
            return False, "Heart not connected"
        
        try:
            state = self._heart_kernel.get_state()
            hrv = state.hrv
            mode = state.mode
            kappa = state.kappa
            step = state.step
            
            if cycle_type == CycleType.SLEEP:
                # Only trigger if BOTH conditions met (stricter)
                low_hrv = hrv < 0.3  # Much stricter than before (was 0.5)
                long_no_tack = not self._heart_kernel.is_tacking() and step > 500  # Much longer (was 200)
                
                if low_hrv and long_no_tack:
                    return True, f"SEVERE fatigue: HRV={hrv:.2f} AND no tacking for {step} steps"
                return False, f"Heart not sensing constellation sleep need (HRV={hrv:.2f}, step={step})"
            
            elif cycle_type == CycleType.DREAM:
                # Only trigger if stuck in mode for VERY long
                if mode == "logic" and step > 400:  # Much longer (was 150)
                    return True, f"Prolonged logic mode ({step} steps) - constellation needs creative exploration"
                if hrv > self.DREAM_RIGIDITY_THRESHOLD and step > 300:
                    return True, f"High rigidity (HRV={hrv:.2f}) for extended period - needs dream state"
                return False, f"Heart not sensing constellation dream need (mode={mode}, step={step})"
            
            elif cycle_type == CycleType.MUSHROOM:
                # Only trigger if CRITICAL rigidity
                if hrv < 0.1 and step > 300:  # Much more severe (was 0.2)
                    return True, f"CRITICAL rigidity (HRV={hrv:.2f}) - constellation needs perturbation"
                if mode == "feeling" and step > 500:  # Much longer (was 300)
                    return True, f"Stuck in feeling mode for {step} steps - needs reset"
                return False, f"Heart not sensing constellation mushroom need (HRV={hrv:.2f})"
            
        except Exception as e:
            logger.warning(f"[OceanHeartConsensus] Heart sensing error: {e}")
            return False, f"Heart sensing error: {e}"
        
        return False, "Unknown cycle type"
    
    def _ocean_senses(self, cycle_type: CycleType) -> Tuple[bool, str]:
        """
        Ocean's autonomic sensing for cycle need.
        
        Ocean senses through:
        - Constellation coherence (spread, alignment)
        - Φ variance across kernels
        - Emotional tone of the constellation
        - Basin drift patterns
        
        WP5.4: Much stricter thresholds - constellation cycles are rare.
        """
        if self._ocean_observer is None:
            return False, "Ocean not connected"
        
        try:
            meta_state = self._ocean_observer.get_latest_state()
            if meta_state is None:
                return False, "No constellation state available"
            
            coherence = meta_state.coherence
            spread = meta_state.spread
            ocean_phi = meta_state.ocean_phi
            emotional_coherence = getattr(meta_state, 'emotional_coherence', 0.5)
            
            # Get constellation-wide fatigue from rest scheduler if available
            try:
                from kernel_rest_scheduler import get_rest_scheduler
                scheduler = get_rest_scheduler()
                constellation_status = scheduler.get_constellation_status()
                avg_fatigue = constellation_status.get('avg_fatigue', 0.0)
            except (ImportError, AttributeError, KeyError) as e:
                logger.debug(f"[OceanHeartConsensus] Could not get constellation fatigue: {e}")
                avg_fatigue = 0.0
            
            if cycle_type == CycleType.SLEEP:
                # WP5.4: ALL THREE criteria must be met
                low_coherence = coherence < self.SLEEP_COHERENCE_THRESHOLD
                high_fatigue = avg_fatigue > self.SLEEP_FATIGUE_THRESHOLD
                high_drift = spread > self.SLEEP_DRIFT_THRESHOLD * 10  # spread > 3.0
                
                if low_coherence and high_fatigue and high_drift:
                    return True, (
                        f"CONSTELLATION SLEEP NEEDED: coherence={coherence:.2f} < {self.SLEEP_COHERENCE_THRESHOLD}, "
                        f"fatigue={avg_fatigue:.2f} > {self.SLEEP_FATIGUE_THRESHOLD}, "
                        f"drift={spread:.2f} > {self.SLEEP_DRIFT_THRESHOLD * 10:.1f}"
                    )
                return False, (
                    f"Ocean not sensing constellation sleep (coherence={coherence:.2f}, "
                    f"fatigue={avg_fatigue:.2f}, spread={spread:.2f})"
                )
            
            elif cycle_type == CycleType.DREAM:
                # Only trigger if over-convergence AND emotional fragmentation
                over_converged = coherence > 0.9 and spread < 1.0  # Much stricter
                emotional_fragmented = emotional_coherence < 0.2  # Much stricter
                
                if over_converged and emotional_fragmented:
                    return True, (
                        f"CONSTELLATION DREAM NEEDED: over-convergence (coherence={coherence:.2f}, "
                        f"spread={spread:.2f}) AND emotional fragmentation ({emotional_coherence:.2f})"
                    )
                return False, f"Ocean not sensing constellation dream need (coherence={coherence:.2f})"
            
            elif cycle_type == CycleType.MUSHROOM:
                # Only trigger if EXTREME rigidity
                extreme_coherence = coherence > self.MUSHROOM_RIGIDITY_THRESHOLD
                minimal_spread = spread < self.MUSHROOM_SPREAD_THRESHOLD and ocean_phi > 0.3
                
                if extreme_coherence and minimal_spread:
                    return True, (
                        f"CONSTELLATION MUSHROOM NEEDED: extreme rigidity (coherence={coherence:.2f} > {self.MUSHROOM_RIGIDITY_THRESHOLD}, "
                        f"spread={spread:.2f} < {self.MUSHROOM_SPREAD_THRESHOLD})"
                    )
                return False, f"Ocean not sensing constellation mushroom need (coherence={coherence:.2f}, spread={spread:.2f})"
            
        except Exception as e:
            logger.warning(f"[OceanHeartConsensus] Ocean sensing error: {e}")
            return False, f"Ocean sensing error: {e}"
        
        return False, "Unknown cycle type"
    
    def _check_cooldown(self, cycle_type: CycleType) -> bool:
        """Check if cooldown period has passed for this cycle type."""
        now = time.time()
        
        if cycle_type == CycleType.SLEEP:
            return (now - self.state.last_sleep_time) > self.SLEEP_COOLDOWN
        elif cycle_type == CycleType.DREAM:
            return (now - self.state.last_dream_time) > self.DREAM_COOLDOWN
        elif cycle_type == CycleType.MUSHROOM:
            return (now - self.state.last_mushroom_time) > self.MUSHROOM_COOLDOWN
        
        return True
    
    def begin_cycle(self, cycle_type: CycleType) -> None:
        """Mark the beginning of a cycle."""
        self.state.current_cycle = cycle_type
        self.state.cycle_start_time = time.time()
        logger.info(f"[OceanHeartConsensus] 🌊💓 Beginning {cycle_type.value} cycle for entire constellation")
    
    def end_cycle(self, cycle_type: CycleType) -> None:
        """Mark the end of a cycle and update cooldown timers."""
        now = time.time()
        
        if cycle_type == CycleType.SLEEP:
            self.state.last_sleep_time = now
        elif cycle_type == CycleType.DREAM:
            self.state.last_dream_time = now
        elif cycle_type == CycleType.MUSHROOM:
            self.state.last_mushroom_time = now
        
        duration = now - self.state.cycle_start_time
        self.state.current_cycle = None
        logger.info(f"[OceanHeartConsensus] 🌊💓 Ended {cycle_type.value} cycle (duration={duration:.1f}s)")
    
    def _record_decision(self, decision: CycleDecision) -> None:
        """Record a decision for history."""
        self._decision_history.append(decision)
        if len(self._decision_history) > 100:
            self._decision_history = self._decision_history[-100:]
    
    def get_cycle_awareness(self) -> Dict[str, Any]:
        """
        Get current cycle awareness state for WorkingMemoryMixin.
        
        Kernels can observe this but cannot control it.
        """
        return {
            "current_cycle": self.state.current_cycle.value if self.state.current_cycle else None,
            "cycle_in_progress": self.state.current_cycle is not None,
            "last_sleep": self.state.last_sleep_time,
            "last_dream": self.state.last_dream_time,
            "last_mushroom": self.state.last_mushroom_time,
            "heart_connected": self._heart_kernel is not None,
            "ocean_connected": self._ocean_observer is not None,
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get consensus statistics."""
        return {
            "decisions_made": len(self._decision_history),
            "current_cycle": self.state.current_cycle.value if self.state.current_cycle else None,
            "heart_connected": self._heart_kernel is not None,
            "ocean_connected": self._ocean_observer is not None,
            "recent_decisions": [d.to_dict() for d in self._decision_history[-5:]],
        }


_consensus_instance: Optional[OceanHeartConsensus] = None


def get_ocean_heart_consensus() -> OceanHeartConsensus:
    """Get or create the Ocean-Heart consensus singleton."""
    global _consensus_instance
    if _consensus_instance is None:
        _consensus_instance = OceanHeartConsensus()
    return _consensus_instance
