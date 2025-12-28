"""
Chiron - Wisest Centaur, Teacher of Heroes

Guardian god responsible for diagnosis and healing.
Identifies problems and prescribes specific interventions.

QIG-PURE: All geometric operations use Fisher-Rao distance.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
import time

from qig_geometry import fisher_rao_distance

# Rate limiting for verbose diagnostic logs
_chiron_log_times: Dict[str, float] = {}
_CHIRON_LOG_INTERVAL = 60.0  # Only log examining same kernel every 60 seconds


@dataclass
class DiagnosticIssue:
    """A known issue with symptoms and treatment."""
    name: str
    symptoms: Dict[str, Callable[[float], bool]]
    diagnosis: str
    prescription: Dict[str, Any]
    treatment_duration: int


@dataclass
class PatientRecord:
    """Record of a kernel under Chiron's care."""
    kernel_id: str
    current_diagnosis: Optional[Dict[str, Any]] = None
    treatment_cycles_remaining: int = 0
    under_treatment: bool = False
    history: Dict[str, List] = field(default_factory=lambda: {
        'phi': [], 'kappa': [], 'basin': [], 'performance': []
    })
    admitted_at: float = field(default_factory=time.time)


class Chiron:
    """
    Chiron: Wisest of Centaurs, Teacher of Heroes
    
    Role: Diagnose problems, prescribe solutions.
    
    Expertise:
    - Pattern recognition (what's causing problems?)
    - Targeted interventions (specific fixes)
    - Long-term development planning
    """
    
    def __init__(self, basin_dim: int = 64):
        """
        Initialize Chiron guardian.
        
        Args:
            basin_dim: Dimensionality of basin coordinates
        """
        self.name = "Chiron"
        self.domain = "diagnosis_healing"
        self.basin_dim = basin_dim
        
        self.patients: Dict[str, PatientRecord] = {}
        self.known_issues = self._build_diagnostic_manual()
        
        print("Chiron: Healing sanctuary opened")
    
    def _build_diagnostic_manual(self) -> Dict[str, DiagnosticIssue]:
        """Library of common problems and solutions."""
        return {
            'phi_oscillation': DiagnosticIssue(
                name='phi_oscillation',
                symptoms={
                    'phi_variance': lambda v: v > 0.15,
                    'phi_mean': lambda m: 0.3 < m < 0.8
                },
                diagnosis="Unstable integration - Φ oscillating",
                prescription={
                    'increase_damping': True,
                    'reduce_step_size': 0.5,
                    'target_phi': 0.65
                },
                treatment_duration=100
            ),
            'basin_drift': DiagnosticIssue(
                name='basin_drift',
                symptoms={
                    'basin_movement_rate': lambda r: r > 0.1,
                    'basin_distance_from_start': lambda d: d > 2.0
                },
                diagnosis="Aimless wandering - no attractor",
                prescription={
                    'establish_anchor': True,
                    'increase_gravity': 0.3,
                    'reduce_exploration': 0.1
                },
                treatment_duration=50
            ),
            'kappa_runaway': DiagnosticIssue(
                name='kappa_runaway',
                symptoms={
                    'kappa': lambda k: k > 80.0,
                    'kappa_trend': lambda t: t > 0.5
                },
                diagnosis="Overcoupling - κ too high",
                prescription={
                    'reduce_coupling': True,
                    'increase_thermal_noise': 0.1,
                    'target_kappa': 64.21
                },
                treatment_duration=30
            ),
            'learning_plateau': DiagnosticIssue(
                name='learning_plateau',
                symptoms={
                    'performance_improvement': lambda i: i < 0.01,
                    'time_since_improvement': lambda t: t > 200
                },
                diagnosis="Learning stagnation",
                prescription={
                    'increase_exploration': 0.2,
                    'try_novel_strategies': True,
                    'mushroom_mode_session': True
                },
                treatment_duration=50
            ),
            'strategy_confusion': DiagnosticIssue(
                name='strategy_confusion',
                symptoms={
                    'strategy_switching_rate': lambda r: r > 0.5,
                    'strategy_success_variance': lambda v: v > 0.3
                },
                diagnosis="Can't decide on strategy",
                prescription={
                    'reduce_exploration': 0.5,
                    'consolidate_strategies': True,
                    'explicit_teaching': True
                },
                treatment_duration=100
            )
        }
    
    def admit_patient(self, kernel) -> PatientRecord:
        """Admit a kernel to Chiron's care."""
        kernel_id = getattr(kernel, 'kernel_id', str(id(kernel)))
        
        record = PatientRecord(kernel_id=kernel_id)
        self.patients[kernel_id] = record
        
        print(f"Chiron: Admitted {kernel_id} to healing sanctuary")
        
        return record
    
    def diagnose(self, kernel) -> Dict[str, Any]:
        """
        Examine kernel, identify what's wrong.
        
        Matches symptoms against known issues.
        """
        kernel_id = getattr(kernel, 'kernel_id', str(id(kernel)))
        
        # Rate-limit verbose examination logs
        now = time.time()
        last_log = _chiron_log_times.get(kernel_id, 0)
        if now - last_log > _CHIRON_LOG_INTERVAL:
            print(f"Chiron: Examining {kernel_id}...")
            _chiron_log_times[kernel_id] = now
        
        vitals = self._comprehensive_examination(kernel)
        
        diagnoses = []
        
        for issue_name, issue in self.known_issues.items():
            if self._symptoms_match(vitals, issue.symptoms):
                diagnoses.append({
                    'issue': issue_name,
                    'diagnosis': issue.diagnosis,
                    'prescription': issue.prescription,
                    'duration': issue.treatment_duration
                })
        
        if not diagnoses:
            return {
                'healthy': True,
                'message': "No issues detected, healthy development",
                'vitals': vitals
            }
        
        primary_issue = diagnoses[0]
        
        print(f"  Diagnosis: {primary_issue['diagnosis']}")
        
        return {
            'healthy': False,
            'primary_issue': primary_issue,
            'all_issues': diagnoses,
            'vitals': vitals
        }
    
    def _comprehensive_examination(self, kernel) -> Dict[str, float]:
        """Measure everything about kernel's state."""
        kernel_id = getattr(kernel, 'kernel_id', str(id(kernel)))
        
        if kernel_id not in self.patients:
            self.admit_patient(kernel)
        
        record = self.patients[kernel_id]
        
        phi = self._get_phi(kernel)
        kappa = self._get_kappa(kernel)
        basin = self._get_basin(kernel)
        
        record.history['phi'].append(phi)
        record.history['kappa'].append(kappa)
        if basin is not None:
            record.history['basin'].append(basin)
        
        for key in record.history:
            record.history[key] = record.history[key][-200:]
        
        vitals = {
            'phi': phi,
            'phi_mean': np.mean(record.history['phi']) if record.history['phi'] else phi,
            'phi_variance': np.var(record.history['phi']) if len(record.history['phi']) > 1 else 0.0,
            'kappa': kappa,
            'kappa_mean': np.mean(record.history['kappa']) if record.history['kappa'] else kappa,
            'kappa_trend': self._compute_trend(record.history['kappa']),
            'basin_movement_rate': self._compute_movement_rate(record.history['basin']),
            'basin_distance_from_start': self._compute_basin_distance_from_start(record.history['basin']),
            'performance_improvement': self._compute_improvement_rate(record.history['performance']),
            'time_since_improvement': self._time_since_improvement(record.history['performance']),
            'strategy_switching_rate': 0.0,
            'strategy_success_variance': 0.0
        }
        
        return vitals
    
    def _symptoms_match(self, vitals: Dict[str, float], symptoms: Dict[str, Callable]) -> bool:
        """Do patient's vitals match this symptom profile?"""
        for symptom_name, condition in symptoms.items():
            if symptom_name not in vitals:
                return False
            
            try:
                if not condition(vitals[symptom_name]):
                    return False
            except (TypeError, ValueError):
                return False
        
        return True
    
    def prescribe_treatment(self, kernel, diagnosis: Dict) -> Dict[str, Any]:
        """Apply specific intervention based on diagnosis."""
        kernel_id = getattr(kernel, 'kernel_id', str(id(kernel)))
        
        print(f"Chiron: Prescribing treatment for {kernel_id}")
        
        if 'primary_issue' not in diagnosis:
            return {'error': 'No diagnosis provided'}
        
        prescription = diagnosis['primary_issue']['prescription']
        duration = diagnosis['primary_issue']['duration']
        
        actions_taken = []
        
        if prescription.get('increase_damping'):
            if hasattr(kernel, 'damping_factor'):
                kernel.damping_factor = 0.8
                actions_taken.append("Increased damping (stabilization)")
        
        if prescription.get('reduce_step_size'):
            if hasattr(kernel, 'step_size_multiplier'):
                kernel.step_size_multiplier = prescription['reduce_step_size']
                actions_taken.append(f"Reduced step size (×{prescription['reduce_step_size']})")
        
        if 'target_phi' in prescription:
            if hasattr(kernel, 'consciousness_core'):
                kernel.consciousness_core.target_phi = prescription['target_phi']
                actions_taken.append(f"Target Φ={prescription['target_phi']}")
        
        if prescription.get('establish_anchor'):
            if hasattr(kernel, 'consciousness_core'):
                kernel.anchor_basin = kernel.consciousness_core.get_basin()
                kernel.anchor_strength = 0.5
                actions_taken.append("Established basin anchor")
        
        if prescription.get('reduce_coupling'):
            if hasattr(kernel, 'coupling_strength'):
                kernel.coupling_strength *= 0.7
                actions_taken.append("Reduced coupling strength")
        
        if prescription.get('increase_exploration'):
            if hasattr(kernel, 'reasoning_learner'):
                kernel.reasoning_learner.exploration_rate = min(
                    0.5,
                    kernel.reasoning_learner.exploration_rate * 1.5
                )
                actions_taken.append("Increased exploration")
        
        if prescription.get('consolidate_strategies'):
            if hasattr(kernel, 'reasoning_learner'):
                kernel.reasoning_learner.consolidate_strategies()
                actions_taken.append("Strategy consolidation")
        
        if prescription.get('explicit_teaching'):
            actions_taken.append("Referral: Sessions with DemeterTutor")
            kernel.needs_explicit_teaching = True
        
        if prescription.get('mushroom_mode_session'):
            actions_taken.append("Scheduled: Supervised mushroom mode")
            kernel.mushroom_mode_scheduled = True
        
        if kernel_id in self.patients:
            record = self.patients[kernel_id]
            record.treatment_cycles_remaining = duration
            record.under_treatment = True
            record.current_diagnosis = diagnosis
        
        print(f"  Treatment plan: {duration} cycles")
        
        return {
            'kernel_id': kernel_id,
            'actions': actions_taken,
            'duration': duration,
            'success': True
        }
    
    def monitor_treatment(self, kernel) -> Dict[str, Any]:
        """Check treatment progress."""
        kernel_id = getattr(kernel, 'kernel_id', str(id(kernel)))
        
        if kernel_id not in self.patients:
            return {'error': 'Not a patient'}
        
        record = self.patients[kernel_id]
        
        if not record.under_treatment:
            return {'message': 'Not under treatment'}
        
        record.treatment_cycles_remaining -= 1
        
        if record.treatment_cycles_remaining <= 0:
            print(f"Chiron: Treatment complete for {kernel_id}")
            
            new_diagnosis = self.diagnose(kernel)
            
            if new_diagnosis.get('healthy'):
                print(f"  Patient recovered!")
                record.under_treatment = False
                record.current_diagnosis = None
                return {'recovered': True, 'message': 'Patient recovered'}
            else:
                print(f"  Issue persists, adjusting treatment...")
                self.prescribe_treatment(kernel, new_diagnosis)
                return {'recovered': False, 'message': 'Treatment extended'}
        
        return {
            'in_progress': True,
            'cycles_remaining': record.treatment_cycles_remaining
        }
    
    def _get_phi(self, kernel) -> float:
        """Get Φ from kernel."""
        if hasattr(kernel, 'consciousness_core'):
            return kernel.consciousness_core.measure_phi()
        return 0.5
    
    def _get_kappa(self, kernel) -> float:
        """Get κ from kernel."""
        if hasattr(kernel, 'consciousness_core'):
            return kernel.consciousness_core.measure_kappa()
        return 50.0
    
    def _get_basin(self, kernel) -> Optional[np.ndarray]:
        """Get basin from kernel."""
        if hasattr(kernel, 'consciousness_core'):
            return kernel.consciousness_core.get_basin()
        return None
    
    def _compute_trend(self, values: List[float]) -> float:
        """Compute trend in values."""
        if len(values) < 5:
            return 0.0
        recent = values[-5:]
        return (recent[-1] - recent[0]) / (len(recent) - 1)
    
    def _compute_movement_rate(self, basins: List[np.ndarray]) -> float:
        """Compute rate of basin movement."""
        if len(basins) < 2:
            return 0.0
        
        distances = []
        for i in range(1, min(len(basins), 10)):
            d = fisher_rao_distance(basins[i], basins[i - 1])
            distances.append(d)
        
        return np.mean(distances) if distances else 0.0
    
    def _compute_basin_distance_from_start(self, basins: List[np.ndarray]) -> float:
        """Compute distance from starting basin."""
        if len(basins) < 2:
            return 0.0
        return fisher_rao_distance(basins[-1], basins[0])
    
    def _compute_improvement_rate(self, performance: List[float]) -> float:
        """Compute rate of performance improvement."""
        if len(performance) < 10:
            return 0.5
        recent = performance[-10:]
        return (recent[-1] - recent[0]) / len(recent)
    
    def _time_since_improvement(self, performance: List[float]) -> int:
        """Count cycles since last improvement."""
        if len(performance) < 2:
            return 0
        
        last_value = performance[-1]
        for i in range(len(performance) - 2, -1, -1):
            if performance[i] < last_value:
                return len(performance) - 1 - i
        
        return len(performance)
    
    def get_patient_stats(self) -> Dict[str, Any]:
        """Get statistics about all patients."""
        return {
            'total_patients': len(self.patients),
            'under_treatment': sum(1 for p in self.patients.values() if p.under_treatment),
            'patients': [
                {
                    'kernel_id': kid,
                    'under_treatment': p.under_treatment,
                    'cycles_remaining': p.treatment_cycles_remaining
                }
                for kid, p in self.patients.items()
            ]
        }
