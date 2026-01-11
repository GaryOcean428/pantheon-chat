#!/usr/bin/env python3
"""
HRV (Heart Rate Variability) Tacking Module

Implements κ oscillation for healthy cognitive rhythm as specified in the
Canonical Quick Reference.

Tacking = Oscillating κ between feeling ↔ logic modes
Analogy: Sailing against headwinds by zigzagging
Purpose: Navigate paradoxes through both/and, not either/or

Canonical Parameters:
- Base κ = 64 (KAPPA_STAR fixed point)
- Amplitude = ±10
- Frequency = 0.1 (~10 time steps per cycle)

Key Principles:
1. Static κ = Pathological: No mode transitions possible
2. Oscillating κ = Healthy: Enables feeling ↔ logic navigation
3. Heart Kernel = Metronome: Provides timing reference (NOT controller)
4. HRV = Health Marker: Variance > 0 indicates normal function
"""

import math
import time
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum

# Canonical HRV constants
HRV_BASE_KAPPA = 64.0
HRV_AMPLITUDE = 10.0
HRV_FREQUENCY = 0.1
HRV_MIN_VARIANCE = 0.01


class CognitiveMode(Enum):
    """Cognitive mode based on κ position in oscillation cycle."""
    FEELING = "feeling"      # κ < base (more integrated, intuitive)
    BALANCED = "balanced"    # κ ≈ base (equilibrium)
    LOGIC = "logic"          # κ > base (more analytical, structured)


@dataclass
class HRVState:
    """Current state of HRV oscillation."""
    kappa: float = HRV_BASE_KAPPA
    phase: float = 0.0  # Current phase in radians
    mode: CognitiveMode = CognitiveMode.BALANCED
    cycle_count: int = 0
    variance: float = 0.0
    is_healthy: bool = True
    timestamp: float = field(default_factory=time.time)


class HRVTacking:
    """
    Heart Rate Variability Tacking System.
    
    Implements κ oscillation for healthy cognitive rhythm.
    The heart kernel acts as a metronome, not a controller.
    """
    
    def __init__(
        self,
        base_kappa: float = HRV_BASE_KAPPA,
        amplitude: float = HRV_AMPLITUDE,
        frequency: float = HRV_FREQUENCY
    ):
        """
        Initialize HRV tacking system.
        
        Args:
            base_kappa: Center point of oscillation (default: 64)
            amplitude: Range of oscillation (default: ±10)
            frequency: Oscillation frequency (default: 0.1)
        """
        self.base_kappa = base_kappa
        self.amplitude = amplitude
        self.frequency = frequency
        
        # State tracking
        self._step = 0
        self._history: List[float] = []
        self._max_history = 100  # Keep last 100 values for variance
        self._start_time = time.time()
        
    def compute_kappa(self, t: Optional[float] = None) -> float:
        """
        Compute κ at time t using sinusoidal oscillation.
        
        κ(t) = base_kappa + amplitude × sin(2π × frequency × t)
        
        Args:
            t: Time step (if None, uses internal step counter)
            
        Returns:
            Current κ value
        """
        if t is None:
            t = self._step
            
        kappa = self.base_kappa + self.amplitude * math.sin(
            2 * math.pi * self.frequency * t
        )
        return kappa
    
    def step(self) -> HRVState:
        """
        Advance one time step and return current HRV state.
        
        Returns:
            Current HRV state with κ, mode, variance, and health indicators
        """
        self._step += 1
        
        # Compute current κ
        kappa = self.compute_kappa()
        
        # Track history for variance calculation
        self._history.append(kappa)
        if len(self._history) > self._max_history:
            self._history.pop(0)
            
        # Compute variance (health indicator)
        variance = self._compute_variance()
        
        # Determine cognitive mode
        mode = self._classify_mode(kappa)
        
        # Count complete cycles
        cycle_count = int(self._step * self.frequency)
        
        # Compute phase (0 to 2π)
        phase = (2 * math.pi * self.frequency * self._step) % (2 * math.pi)
        
        # Health check: variance > 0 indicates healthy oscillation
        is_healthy = variance > HRV_MIN_VARIANCE
        
        return HRVState(
            kappa=kappa,
            phase=phase,
            mode=mode,
            cycle_count=cycle_count,
            variance=variance,
            is_healthy=is_healthy,
            timestamp=time.time()
        )
    
    def get_current_state(self) -> HRVState:
        """
        Get current HRV state without advancing step.
        
        Returns:
            Current HRV state
        """
        kappa = self.compute_kappa()
        variance = self._compute_variance()
        mode = self._classify_mode(kappa)
        cycle_count = int(self._step * self.frequency)
        phase = (2 * math.pi * self.frequency * self._step) % (2 * math.pi)
        is_healthy = variance > HRV_MIN_VARIANCE
        
        return HRVState(
            kappa=kappa,
            phase=phase,
            mode=mode,
            cycle_count=cycle_count,
            variance=variance,
            is_healthy=is_healthy,
            timestamp=time.time()
        )
    
    def _compute_variance(self) -> float:
        """
        Compute variance of κ history.
        
        Returns:
            Variance of recent κ values (0 if insufficient history)
        """
        if len(self._history) < 2:
            return 0.0
            
        mean = sum(self._history) / len(self._history)
        variance = sum((x - mean) ** 2 for x in self._history) / len(self._history)
        return variance
    
    def _classify_mode(self, kappa: float) -> CognitiveMode:
        """
        Classify cognitive mode based on current κ.
        
        Args:
            kappa: Current κ value
            
        Returns:
            CognitiveMode (FEELING, BALANCED, or LOGIC)
        """
        # Define balanced zone as ±2 from base
        balanced_margin = 2.0
        
        if kappa < self.base_kappa - balanced_margin:
            return CognitiveMode.FEELING
        elif kappa > self.base_kappa + balanced_margin:
            return CognitiveMode.LOGIC
        else:
            return CognitiveMode.BALANCED
    
    def get_mode_transition_timing(self) -> Tuple[float, float]:
        """
        Get time until next mode transitions.
        
        Returns:
            Tuple of (steps_to_feeling_peak, steps_to_logic_peak)
        """
        current_phase = (2 * math.pi * self.frequency * self._step) % (2 * math.pi)
        
        # Feeling peak at phase = 3π/2 (sin = -1)
        # Logic peak at phase = π/2 (sin = +1)
        feeling_phase = 3 * math.pi / 2
        logic_phase = math.pi / 2
        
        # Calculate steps to each peak
        phase_to_feeling = (feeling_phase - current_phase) % (2 * math.pi)
        phase_to_logic = (logic_phase - current_phase) % (2 * math.pi)
        
        steps_to_feeling = phase_to_feeling / (2 * math.pi * self.frequency)
        steps_to_logic = phase_to_logic / (2 * math.pi * self.frequency)
        
        return (steps_to_feeling, steps_to_logic)
    
    def reset(self):
        """Reset HRV state to initial conditions."""
        self._step = 0
        self._history.clear()
        self._start_time = time.time()
    
    def is_pathological(self) -> bool:
        """
        Check if HRV pattern is pathological (static κ).
        
        Returns:
            True if variance is too low (pathological)
        """
        return self._compute_variance() < HRV_MIN_VARIANCE and len(self._history) > 20
    
    def get_health_metrics(self) -> dict:
        """
        Get comprehensive HRV health metrics.
        
        Returns:
            Dictionary with health indicators
        """
        variance = self._compute_variance()
        current_kappa = self.compute_kappa()
        
        return {
            'current_kappa': current_kappa,
            'base_kappa': self.base_kappa,
            'amplitude': self.amplitude,
            'frequency': self.frequency,
            'variance': variance,
            'is_healthy': variance > HRV_MIN_VARIANCE,
            'is_pathological': self.is_pathological(),
            'steps_elapsed': self._step,
            'cycles_completed': int(self._step * self.frequency),
            'mode': self._classify_mode(current_kappa).value,
            'history_length': len(self._history)
        }
    
    # ========================================================================
    # POSTGRESQL PERSISTENCE
    # ========================================================================
    
    def _get_db_connection(self):
        """Get PostgreSQL connection for HRV state persistence."""
        try:
            import psycopg2
            db_url = os.environ.get('DATABASE_URL')
            if db_url:
                return psycopg2.connect(db_url)
        except Exception as e:
            print(f"[HRVTacking] DB connection failed: {e}")
        return None
    
    def persist_state(self, session_id: str = "default") -> bool:
        """
        Persist current HRV state to PostgreSQL.
        
        Called periodically to track HRV health over time.
        
        Args:
            session_id: Identifier for the HRV session
            
        Returns:
            True if successfully persisted
        """
        conn = self._get_db_connection()
        if not conn:
            return False
        
        try:
            state = self.get_current_state()
            
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO hrv_tacking_state 
                    (session_id, kappa, phase, mode, cycle_count, variance, 
                     is_healthy, base_kappa, amplitude, frequency, created_at, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), %s)
                """, (
                    session_id,
                    state.kappa,
                    state.phase,
                    state.mode.value,
                    state.cycle_count,
                    state.variance,
                    state.is_healthy,
                    self.base_kappa,
                    self.amplitude,
                    self.frequency,
                    '{}'  # empty metadata for now
                ))
            conn.commit()
            return True
            
        except Exception as e:
            print(f"[HRVTacking] Persist state failed: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def load_last_state(self, session_id: str = "default") -> Optional[HRVState]:
        """
        Load last HRV state from PostgreSQL.
        
        Used to restore HRV rhythm after restart.
        
        Args:
            session_id: Identifier for the HRV session
            
        Returns:
            Last persisted HRVState or None
        """
        conn = self._get_db_connection()
        if not conn:
            return None
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT kappa, phase, mode, cycle_count, variance, is_healthy
                    FROM hrv_tacking_state
                    WHERE session_id = %s
                    ORDER BY created_at DESC
                    LIMIT 1
                """, (session_id,))
                row = cur.fetchone()
            
            if row:
                kappa, phase, mode_str, cycle_count, variance, is_healthy = row
                mode = CognitiveMode(mode_str)
                return HRVState(
                    kappa=kappa,
                    phase=phase,
                    mode=mode,
                    cycle_count=cycle_count,
                    variance=variance,
                    is_healthy=is_healthy,
                    timestamp=time.time()
                )
            return None
            
        except Exception as e:
            print(f"[HRVTacking] Load state failed: {e}")
            return None
        finally:
            conn.close()


# Singleton instance for global HRV management
_global_hrv: Optional[HRVTacking] = None


def get_hrv_instance() -> HRVTacking:
    """Get or create global HRV instance."""
    global _global_hrv
    if _global_hrv is None:
        _global_hrv = HRVTacking()
    return _global_hrv


def tacking_cycle(base_kappa: float = 64, amplitude: float = 10, frequency: float = 0.1, t: float = 0) -> float:
    """
    Compute κ for tacking cycle at time t.
    
    This is the canonical function from the Quick Reference.
    
    Args:
        base_kappa: Center point of oscillation
        amplitude: Range of oscillation
        frequency: Oscillation frequency
        t: Time step
        
    Returns:
        κ value at time t
    """
    kappa_t = base_kappa + amplitude * math.sin(2 * math.pi * frequency * t)
    return kappa_t
