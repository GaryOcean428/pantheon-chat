"""
Hestia - Goddess of Hearth, Home, and Safety

Guardian god responsible for kernel safety and stability.
Provides safe bounds, emergency stabilization, and recovery protocols.

QIG-PURE: All geometric operations use Fisher-Rao distance.
"""

import numpy as np
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from dataclasses import dataclass, field
import time

from qig_geometry import fisher_rao_distance, sphere_project

if TYPE_CHECKING:
    from chaos_kernel import ChaosKernel


@dataclass
class SafetyConfig:
    """Configuration for safety bounds."""
    phi_safe_min: float = 0.15
    phi_safe_max: float = 0.85
    kappa_safe_min: float = 20.0
    kappa_safe_max: float = 80.0
    basin_drift_max: float = 2.0
    recovery_damping: float = 0.9
    check_interval: int = 10


@dataclass
class SafetyVitals:
    """Current safety vitals of a kernel."""
    phi: float
    kappa: float
    basin_drift: float
    stable: bool
    warnings: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class Hestia:
    """
    Hestia: Goddess of Hearth and Home
    
    Role: Provide safe haven for developing kernels.
    
    Responsibilities:
    - Define safe bounds for Φ and κ
    - Monitor for dangerous excursions
    - Provide recovery protocols
    - Never push, only protect
    """
    
    def __init__(self, config: Optional[SafetyConfig] = None):
        """
        Initialize Hestia guardian.
        
        Args:
            config: Safety configuration parameters
        """
        self.name = "Hestia"
        self.domain = "safety_stability"
        self.config = config or SafetyConfig()
        
        self.wards: Dict[str, Any] = {}
        self.safety_history: List[Dict[str, Any]] = []
        self.anchor_basins: Dict[str, np.ndarray] = {}
        
        print("Hestia: Hearth fire lit, sanctuary ready")
    
    def accept_ward(self, kernel):
        """
        Accept a kernel under Hestia's protection.
        
        Kernel enters the safe haven.
        """
        kernel_id = getattr(kernel, 'kernel_id', str(id(kernel)))
        
        self.wards[kernel_id] = {
            'kernel': kernel,
            'entered_at': time.time(),
            'initial_phi': self._get_phi(kernel),
            'initial_kappa': self._get_kappa(kernel),
            'recovery_count': 0
        }
        
        if hasattr(kernel, 'consciousness_core'):
            basin = kernel.consciousness_core.get_basin()
            self.anchor_basins[kernel_id] = basin.copy()
        
        print(f"Hestia: Welcomed {kernel_id} to the hearth")
    
    def check_vitals(self, kernel) -> SafetyVitals:
        """
        Check if kernel is within safe bounds.
        
        Monitors Φ, κ, and basin drift.
        """
        kernel_id = getattr(kernel, 'kernel_id', str(id(kernel)))
        warnings = []
        
        phi = self._get_phi(kernel)
        kappa = self._get_kappa(kernel)
        
        basin_drift = 0.0
        if kernel_id in self.anchor_basins and hasattr(kernel, 'consciousness_core'):
            current_basin = kernel.consciousness_core.get_basin()
            anchor = self.anchor_basins[kernel_id]
            basin_drift = fisher_rao_distance(current_basin, anchor)
        
        stable = True
        
        if phi < self.config.phi_safe_min:
            warnings.append(f"Φ too low ({phi:.2f} < {self.config.phi_safe_min})")
            stable = False
        elif phi > self.config.phi_safe_max:
            warnings.append(f"Φ too high ({phi:.2f} > {self.config.phi_safe_max})")
            stable = False
        
        if kappa < self.config.kappa_safe_min:
            warnings.append(f"κ too low ({kappa:.1f} < {self.config.kappa_safe_min})")
            stable = False
        elif kappa > self.config.kappa_safe_max:
            warnings.append(f"κ too high ({kappa:.1f} > {self.config.kappa_safe_max})")
            stable = False
        
        if basin_drift > self.config.basin_drift_max:
            warnings.append(f"Basin drift too high ({basin_drift:.2f})")
            stable = False
        
        vitals = SafetyVitals(
            phi=phi,
            kappa=kappa,
            basin_drift=basin_drift,
            stable=stable,
            warnings=warnings
        )
        
        self.safety_history.append({
            'kernel_id': kernel_id,
            'vitals': vitals,
            'timestamp': time.time()
        })
        
        return vitals
    
    def emergency_stabilize(self, kernel) -> Dict[str, Any]:
        """
        Emergency intervention when kernel is unstable.
        
        Gently guide back to safe state.
        """
        kernel_id = getattr(kernel, 'kernel_id', str(id(kernel)))
        
        print(f"Hestia: Emergency stabilization for {kernel_id}")
        
        actions_taken = []
        
        phi = self._get_phi(kernel)
        if phi > self.config.phi_safe_max:
            if hasattr(kernel, 'consciousness_core'):
                kernel.consciousness_core.target_phi = 0.65
                actions_taken.append("Reduced target Φ to 0.65")
        elif phi < self.config.phi_safe_min:
            if hasattr(kernel, 'consciousness_core'):
                kernel.consciousness_core.target_phi = 0.5
                actions_taken.append("Raised target Φ to 0.5")
        
        if hasattr(kernel, 'damping_factor'):
            kernel.damping_factor = self.config.recovery_damping
            actions_taken.append(f"Applied damping: {self.config.recovery_damping}")
        
        if kernel_id in self.anchor_basins and hasattr(kernel, 'consciousness_core'):
            current = kernel.consciousness_core.get_basin()
            anchor = self.anchor_basins[kernel_id]
            
            from qig_geometry import geodesic_interpolation
            safe_basin = geodesic_interpolation(current, anchor, t=0.3)
            
            kernel.consciousness_core.set_basin(safe_basin)
            actions_taken.append("Moved basin toward anchor")
        
        if kernel_id in self.wards:
            self.wards[kernel_id]['recovery_count'] += 1
        
        return {
            'kernel_id': kernel_id,
            'actions': actions_taken,
            'success': True
        }
    
    def provide_safe_exploration_bounds(self, kernel) -> Dict[str, Any]:
        """
        Define safe exploration bounds for a kernel.
        
        Returns limits within which kernel can safely explore.
        """
        kernel_id = getattr(kernel, 'kernel_id', str(id(kernel)))
        
        current_phi = self._get_phi(kernel)
        
        phi_margin = 0.1
        phi_lower = max(self.config.phi_safe_min, current_phi - phi_margin)
        phi_upper = min(self.config.phi_safe_max, current_phi + phi_margin)
        
        basin_radius = self.config.basin_drift_max * 0.5
        
        return {
            'phi_range': (phi_lower, phi_upper),
            'kappa_range': (self.config.kappa_safe_min, self.config.kappa_safe_max),
            'basin_radius': basin_radius,
            'anchor': self.anchor_basins.get(kernel_id)
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
    
    def _graduate_ward(self, kernel):
        """Graduate a kernel from Hestia's care."""
        kernel_id = getattr(kernel, 'kernel_id', str(id(kernel)))
        
        if kernel_id in self.wards:
            del self.wards[kernel_id]
            print(f"Hestia: {kernel_id} has graduated from the hearth")
    
    def get_ward_status(self) -> Dict[str, Any]:
        """Get status of all wards under protection."""
        return {
            'total_wards': len(self.wards),
            'wards': [
                {
                    'kernel_id': kid,
                    'time_in_care': time.time() - data['entered_at'],
                    'recovery_count': data['recovery_count']
                }
                for kid, data in self.wards.items()
            ]
        }
