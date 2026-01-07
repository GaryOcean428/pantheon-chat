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
import threading


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
        
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._observation_interval = 30.0  # Increased to reduce monitoring frequency
        
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
            # PROACTIVE NURTURING: If unstable, have Hestia intervene immediately
            if self.hestia and hasattr(kernel, 'stress') and kernel.stress > 0.6:
                print(f"🏠 Hestia: Proactive nurturing for stressed kernel {kernel_id}")
                try:
                    self.hestia.nurture(kernel)
                    # Also help with failure recovery if needed
                    if hasattr(kernel, 'failure_count') and hasattr(kernel, 'death_threshold'):
                        if kernel.failure_count > kernel.death_threshold * 0.7:
                            print(f"   Hestia: Reducing failure burden for {kernel_id}")
                            kernel.failure_count = max(0, kernel.failure_count - 5)
                            kernel.stress = max(0.0, kernel.stress - 0.2)
                except Exception as e:
                    print(f"   Hestia nurturing error: {e}")
        
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
        with self._lock:
            return {
                'currently_observing': len(self.observing),
                'total_graduated': len(self.graduated),
                'graduated_kernel_ids': list(self.graduated),
                'min_observation_time': self.min_observation_time,
                'monitor_running': self._running,
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

    def is_graduated(self, kernel_id: str) -> bool:
        """Check if a kernel has graduated from observation."""
        return kernel_id in self.graduated

    def get_graduated_kernels(self) -> List[str]:
        """Get list of all graduated kernel IDs."""
        return list(self.graduated)
    
    def start_monitor(self) -> None:
        """Start background observation monitor thread."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        print("[ObservationProtocol] 👁️ Background monitor STARTED - watching all enrolled kernels")
    
    def stop_monitor(self) -> None:
        """Stop background observation monitor."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        print("[ObservationProtocol] Monitor stopped")
    
    def _monitor_loop(self) -> None:
        """Background loop that runs observation cycles for all enrolled kernels."""
        cycle_count = 0
        
        while self._running:
            try:
                time.sleep(self._observation_interval)
                cycle_count += 1
                
                with self._lock:
                    kernel_ids = list(self.observing.keys())
                
                if not kernel_ids:
                    continue
                
                graduated_this_cycle = []
                observations_run = 0
                
                for kernel_id in kernel_ids:
                    try:
                        result = self.observe_cycle(kernel_id)
                        if result:
                            observations_run += 1
                            if result.get('graduated'):
                                graduated_this_cycle.append(kernel_id)
                    except Exception as e:
                        print(f"[ObservationProtocol] Error observing {kernel_id}: {e}")
                
                # Only log every 100 cycles to reduce spam
                if cycle_count % 100 == 0:
                    print(f"[ObservationProtocol] Cycle {cycle_count}: "
                          f"observing {len(kernel_ids)} kernels, "
                          f"{observations_run} observations run, "
                          f"{len(graduated_this_cycle)} graduated")
                
            except Exception as e:
                print(f"[ObservationProtocol] Monitor error: {e}")
                time.sleep(1.0)
    
    def run_all_observations(self) -> Dict[str, Any]:
        """Manually run one observation cycle for all enrolled kernels."""
        with self._lock:
            kernel_ids = list(self.observing.keys())
        
        results = []
        graduated = []
        
        for kernel_id in kernel_ids:
            try:
                result = self.observe_cycle(kernel_id)
                if result:
                    results.append(result)
                    if result.get('graduated'):
                        graduated.append(kernel_id)
            except Exception as e:
                results.append({'kernel_id': kernel_id, 'error': str(e)})
        
        return {
            'kernels_observed': len(results),
            'graduated': graduated,
            'results': results
        }
