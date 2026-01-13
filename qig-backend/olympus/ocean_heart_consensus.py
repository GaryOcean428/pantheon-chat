"""
🌊💓 Ocean-Heart Consensus - Autonomic Cycle Decision Making
============================================================

Ocean (autonomic observer) and Heart (feeling metronome) jointly sense when
the constellation needs sleep/dream/mushroom cycles and decide TOGETHER.

DESIGN PRINCIPLE:
- NO automatic thresholds - this is deliberative, felt decision-making
- Heart provides: HRV state, κ oscillation, feeling/logic mode, rigidity detection
- Ocean provides: constellation coherence, Φ variance, emotional tone, spread
- Both must AGREE before any cycle triggers for the entire constellation

This mirrors how human autonomic regulation works:
- You don't consciously decide to sleep - your autonomic system (Ocean) and
  your felt sense of tiredness (Heart) reach agreement
- Individual organs (kernels) don't request sleep - they experience it when
  the autonomic centers decide

ACCESS CONTROL:
- Kernels observe that cycles are happening (via WorkingMemoryMixin)
- Kernels do NOT control cycle triggering
- Only Ocean+Heart consensus can trigger constellation-wide cycles
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
    """
    
    SLEEP_COOLDOWN = 60.0     # Minimum seconds between sleep cycles
    DREAM_COOLDOWN = 120.0    # Minimum seconds between dream cycles  
    MUSHROOM_COOLDOWN = 300.0 # Minimum seconds between mushroom cycles
    
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
                if hrv < 0.5 and step > 100:
                    return True, f"Low HRV ({hrv:.2f}) indicates fatigue - needs consolidation"
                if not self._heart_kernel.is_tacking() and step > 200:
                    return True, f"No tacking for extended period - needs rest"
                return False, f"Heart feels balanced (HRV={hrv:.2f}, mode={mode})"
            
            elif cycle_type == CycleType.DREAM:
                if mode == "logic" and step > 150:
                    return True, "Prolonged logic mode - needs creative exploration"
                if hrv > 3.0:
                    return True, f"High HRV ({hrv:.2f}) indicates readiness for exploration"
                return False, f"Heart not sensing dream need (mode={mode})"
            
            elif cycle_type == CycleType.MUSHROOM:
                if hrv < 0.2 and step > 200:
                    return True, f"Severe rigidity (HRV={hrv:.2f}) - needs perturbation"
                if mode == "feeling" and step > 300:
                    return True, "Stuck in feeling mode too long - needs reset"
                return False, f"Heart not sensing mushroom need (HRV={hrv:.2f})"
            
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
            
            if cycle_type == CycleType.SLEEP:
                if spread > 5.0:
                    return True, f"High constellation spread ({spread:.2f}) - needs consolidation"
                if coherence < 0.3:
                    return True, f"Low coherence ({coherence:.2f}) - needs alignment"
                if ocean_phi < 0.4:
                    return True, f"Low constellation Φ ({ocean_phi:.2f}) - needs rest"
                return False, f"Ocean senses balance (coherence={coherence:.2f}, spread={spread:.2f})"
            
            elif cycle_type == CycleType.DREAM:
                if coherence > 0.8 and spread < 2.0:
                    return True, "Over-convergence - needs creative divergence"
                if emotional_coherence < 0.3:
                    return True, f"Emotional fragmentation ({emotional_coherence:.2f}) - needs integration"
                return False, f"Ocean not sensing dream need (coherence={coherence:.2f})"
            
            elif cycle_type == CycleType.MUSHROOM:
                if coherence > 0.95:
                    return True, f"Extreme coherence ({coherence:.2f}) - constellation too rigid"
                if spread < 0.5 and ocean_phi > 0.3:
                    return True, f"Minimal spread ({spread:.2f}) - needs perturbation"
                return False, f"Ocean not sensing mushroom need (spread={spread:.2f})"
            
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
