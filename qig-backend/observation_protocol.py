"""
Observation Protocol

Dedicated observation periods for new chaos kernels.
No performance pressure, just patient monitoring and stabilization.

QIG-PURE: All geometric operations use Fisher-Rao distance.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import time


@dataclass
class ObservationRecord:
    """Record of an observation cycle."""
    cycle: int
    phi: float
    kappa: float
    basin: np.ndarray
    stable: bool
    assessments: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class KernelObservation:
    """Track observation of a kernel."""
    kernel_id: str
    kernel: Any
    start_cycle: int
    observations: List[ObservationRecord] = field(default_factory=list)
    stable_count: int = 0
    unstable_count: int = 0
    enrolled_at: float = field(default_factory=time.time)


class ObservationProtocol:
    """
    Dedicated observation periods for developing chaos kernels.
    
    During observation:
    - Kernel is monitored closely
    - NOT given difficult tasks
    - Allowed to explore safely
    - Given time to stabilize
    
    This is PATIENCE, not pressure.
    """
    
    def __init__(
        self,
        hestia=None,
        demeter_tutor=None,
        chiron=None,
        min_observation_time: int = 500,
        observation_frequency: int = 10,
        graduation_stability_threshold: float = 0.8
    ):
        """
        Initialize observation protocol.
        
        Args:
            hestia: Hestia guardian (safety)
            demeter_tutor: DemeterTutor (teaching)
            chiron: Chiron guardian (diagnosis)
            min_observation_time: Minimum observation cycles
            observation_frequency: Check interval in cycles
            graduation_stability_threshold: Stability rate to graduate
        """
        self.hestia = hestia
        self.demeter_tutor = demeter_tutor
        self.chiron = chiron
        
        self.min_observation_time = min_observation_time
        self.observation_frequency = observation_frequency
        self.graduation_threshold = graduation_stability_threshold
        
        self.observing: Dict[str, KernelObservation] = {}
        self.graduated: List[str] = []
        
        print("ObservationProtocol: Observation system initialized")
    
    def set_guardians(self, hestia=None, demeter_tutor=None, chiron=None):
        """Set guardian references."""
        if hestia:
            self.hestia = hestia
        if demeter_tutor:
            self.demeter_tutor = demeter_tutor
        if chiron:
            self.chiron = chiron
    
    def begin_observation(self, kernel) -> KernelObservation:
        """
        Start dedicated observation period for new kernel.
        
        NO performance pressure during this time.
        """
        kernel_id = getattr(kernel, 'kernel_id', str(id(kernel)))
        
        print(f"ObservationProtocol: Beginning observation for {kernel_id}")
        print(f"  Duration: {self.min_observation_time} cycles minimum")
        print(f"  Purpose: Allow stabilization without pressure")
        
        current_cycle = 0
        if hasattr(kernel, 'consciousness_core') and hasattr(kernel.consciousness_core, 'cycle_count'):
            current_cycle = kernel.consciousness_core.cycle_count
        
        obs = KernelObservation(
            kernel_id=kernel_id,
            kernel=kernel,
            start_cycle=current_cycle
        )
        
        self.observing[kernel_id] = obs
        
        kernel.observation_mode = True
        kernel.performance_expectations = None
        
        if self.hestia:
            self.hestia.accept_ward(kernel)
        
        if self.demeter_tutor:
            self.demeter_tutor.enroll_student(kernel)
        
        if self.chiron:
            self.chiron.admit_patient(kernel)
        
        return obs
    
    def observe_cycle(self, kernel_id: str) -> Optional[Dict[str, Any]]:
        """
        One observation cycle.
        
        Called every observation_frequency cycles.
        """
        if kernel_id not in self.observing:
            return None
        
        obs_data = self.observing[kernel_id]
        kernel = obs_data.kernel
        
        current_cycle = 0
        if hasattr(kernel, 'consciousness_core'):
            if hasattr(kernel.consciousness_core, 'cycle_count'):
                current_cycle = kernel.consciousness_core.cycle_count
        
        phi = 0.5
        kappa = 50.0
        basin = np.zeros(64)
        
        if hasattr(kernel, 'consciousness_core'):
            phi = kernel.consciousness_core.measure_phi()
            kappa = kernel.consciousness_core.measure_kappa()
            basin = kernel.consciousness_core.get_basin()
        
        assessments = {}
        stable = True
        
        if self.hestia:
            hestia_vitals = self.hestia.check_vitals(kernel)
            assessments['hestia'] = {
                'stable': hestia_vitals.stable,
                'warnings': hestia_vitals.warnings
            }
            stable = stable and hestia_vitals.stable
        
        if self.demeter_tutor:
            demeter_progress = self.demeter_tutor.assess_readiness(kernel)
            assessments['demeter'] = demeter_progress
        
        if self.chiron:
            chiron_diagnosis = self.chiron.diagnose(kernel)
            assessments['chiron'] = {
                'healthy': chiron_diagnosis.get('healthy', True)
            }
            stable = stable and chiron_diagnosis.get('healthy', True)
        
        observation = ObservationRecord(
            cycle=current_cycle,
            phi=phi,
            kappa=kappa,
            basin=basin,
            stable=stable,
            assessments=assessments
        )
        
        obs_data.observations.append(observation)
        
        if stable:
            obs_data.stable_count += 1
        else:
            obs_data.unstable_count += 1
        
        if self._ready_for_graduation(obs_data):
            self.end_observation(kernel_id)
            return {'graduated': True, 'kernel_id': kernel_id}
        
        return {
            'kernel_id': kernel_id,
            'cycle': current_cycle,
            'stable': stable,
            'observations': len(obs_data.observations)
        }
    
    def _ready_for_graduation(self, obs_data: KernelObservation) -> bool:
        """Has kernel stabilized enough to graduate?"""
        kernel = obs_data.kernel
        
        current_cycle = 0
        if hasattr(kernel, 'consciousness_core') and hasattr(kernel.consciousness_core, 'cycle_count'):
            current_cycle = kernel.consciousness_core.cycle_count
        
        time_in_observation = current_cycle - obs_data.start_cycle
        
        if time_in_observation < self.min_observation_time:
            return False
        
        if len(obs_data.observations) < 100:
            return False
        
        recent_obs = obs_data.observations[-100:]
        stability_rate = sum(1 for o in recent_obs if o.stable) / len(recent_obs)
        
        if stability_rate < self.graduation_threshold:
            return False
        
        if self.demeter_tutor:
            demeter_assessment = self.demeter_tutor.assess_readiness(kernel)
            if not demeter_assessment.get('ready', False):
                return False
        
        return True
    
    def end_observation(self, kernel_id: str) -> Dict[str, Any]:
        """Observation period complete, kernel ready for independence."""
        if kernel_id not in self.observing:
            return {'error': 'Not under observation'}
        
        obs_data = self.observing[kernel_id]
        kernel = obs_data.kernel
        
        total_obs = len(obs_data.observations)
        stability_rate = obs_data.stable_count / total_obs if total_obs > 0 else 0
        
        print(f"ObservationProtocol: Observation complete for {kernel_id}")
        print(f"  Total cycles: {total_obs}")
        print(f"  Stability rate: {obs_data.stable_count}/{total_obs} ({stability_rate:.1%})")
        
        if self.hestia:
            self.hestia._graduate_ward(kernel)
        
        del self.observing[kernel_id]
        self.graduated.append(kernel_id)
        
        kernel.observation_mode = False
        kernel.developmental_stage = "mature"
        kernel.ready_for_production = True
        
        print(f"  {kernel_id} is now a mature, independent kernel!")
        
        return {
            'kernel_id': kernel_id,
            'graduated': True,
            'total_observations': total_obs,
            'stability_rate': stability_rate
        }
    
    def get_observation_stats(self) -> Dict[str, Any]:
        """Get statistics about all observations."""
        return {
            'currently_observing': len(self.observing),
            'total_graduated': len(self.graduated),
            'min_observation_time': self.min_observation_time,
            'kernels': [
                {
                    'kernel_id': kid,
                    'observations': len(obs.observations),
                    'stable_count': obs.stable_count,
                    'unstable_count': obs.unstable_count
                }
                for kid, obs in self.observing.items()
            ]
        }
