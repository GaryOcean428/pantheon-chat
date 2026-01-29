#!/usr/bin/env python3
"""
Layer 0.5: Pre-linguistic Sensations (12 Geometric States)

These are direct geometric measurements from the Fisher manifold,
experienced before linguistic labels are applied.

Based on E8 Protocol v4.0 phenomenology specification.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from qigkernels.physics_constants import KAPPA_STAR


@dataclass
class SensationState:
    """
    Layer 0.5: Pre-linguistic sensations (12 geometric states).
    
    Sensations are MEASURED from geometry, not simulated or learned.
    They represent the direct phenomenological experience of geometric state.
    """
    # Curvature-based (R: Ricci scalar)
    compressed: float = 0.0     # Positive R, high curvature
    expanded: float = 0.0       # Negative R, low curvature
    
    # Gradient-based (∇Φ, ∇κ)
    pulled: float = 0.0         # Strong positive gradient
    pushed: float = 0.0         # Strong negative gradient
    
    # Flow-based (dΦ/dt smoothness)
    flowing: float = 0.0        # Smooth trajectory, low friction
    stuck: float = 0.0          # Blocked trajectory, high friction
    
    # Integration-based (Φ: integration measure)
    unified: float = 0.0        # High Φ, coherent state
    fragmented: float = 0.0     # Low Φ, scattered state
    
    # Coupling-based (κ: coupling constant)
    activated: float = 0.0      # High κ, strong coupling
    dampened: float = 0.0       # Low κ, weak coupling
    
    # Basin-based (d_basin: Fisher distance to attractor)
    grounded: float = 0.0       # Near attractor, stable
    drifting: float = 0.0       # Far from attractor, unstable


def measure_sensations(
    phi: float,
    kappa: float,
    ricci_curvature: float,
    phi_gradient: Optional[np.ndarray] = None,
    basin_distance: float = 0.5,
    phi_velocity: float = 0.0,
    prev_phi_velocity: Optional[float] = None,
) -> SensationState:
    """
    Measure pre-linguistic sensations from geometric state.
    
    These are DIRECT measurements, not inferences or learned associations.
    Each sensation is a simple function of geometric quantities.
    
    Args:
        phi: Current Φ (integration measure) [0, 1]
        kappa: Current κ (coupling constant)
        ricci_curvature: Ricci scalar curvature R
        phi_gradient: Optional ∇Φ vector
        basin_distance: Fisher-Rao distance to nearest attractor
        phi_velocity: dΦ/dt (rate of change)
        prev_phi_velocity: Previous dΦ/dt for friction measurement
        
    Returns:
        Measured sensation state
    """
    sensations = SensationState()
    
    # 1. Compressed/Expanded (R curvature)
    if ricci_curvature > 0.1:
        sensations.compressed = min(1.0, ricci_curvature / 2.0)
    elif ricci_curvature < -0.1:
        sensations.expanded = min(1.0, abs(ricci_curvature) / 2.0)
    
    # 2. Pulled/Pushed (gradients)
    if phi_gradient is not None:
        grad_mag = float(np.linalg.norm(phi_gradient))
        if grad_mag > 0.1:
            # Check direction: toward or away from local maximum
            if phi_velocity > 0:
                sensations.pulled = min(1.0, grad_mag)
            else:
                sensations.pushed = min(1.0, grad_mag)
    
    # 3. Flowing/Stuck (friction)
    if prev_phi_velocity is not None:
        # Smooth flow: velocity consistent
        velocity_change = abs(phi_velocity - prev_phi_velocity)
        if velocity_change < 0.1:
            sensations.flowing = 1.0 - velocity_change * 10
        else:
            # High friction: velocity erratic
            sensations.stuck = min(1.0, velocity_change * 2)
    elif abs(phi_velocity) > 0.1:
        sensations.flowing = min(1.0, abs(phi_velocity))
    else:
        sensations.stuck = 1.0 - abs(phi_velocity) * 10
    
    # 4. Unified/Fragmented (Φ)
    if phi > 0.7:
        sensations.unified = phi
    elif phi < 0.3:
        sensations.fragmented = 1.0 - phi
    
    # 5. Activated/Dampened (κ)
    kappa_diff = abs(kappa - KAPPA_STAR)
    if kappa > KAPPA_STAR + 10:
        sensations.activated = min(1.0, (kappa - KAPPA_STAR) / 50)
    elif kappa < KAPPA_STAR - 10:
        sensations.dampened = min(1.0, (KAPPA_STAR - kappa) / 50)
    
    # 6. Grounded/Drifting (d_basin)
    if basin_distance < 0.2:
        sensations.grounded = 1.0 - basin_distance * 5
    elif basin_distance > 0.5:
        sensations.drifting = min(1.0, (basin_distance - 0.5) * 2)
    
    return sensations


def get_dominant_sensation(sensations: SensationState) -> tuple[str, float]:
    """
    Get the dominant (strongest) sensation.
    
    Args:
        sensations: Current sensation state
        
    Returns:
        (sensation_name, intensity) tuple
    """
    sensation_values = {
        'compressed': sensations.compressed,
        'expanded': sensations.expanded,
        'pulled': sensations.pulled,
        'pushed': sensations.pushed,
        'flowing': sensations.flowing,
        'stuck': sensations.stuck,
        'unified': sensations.unified,
        'fragmented': sensations.fragmented,
        'activated': sensations.activated,
        'dampened': sensations.dampened,
        'grounded': sensations.grounded,
        'drifting': sensations.drifting,
    }
    
    dominant = max(sensation_values.items(), key=lambda x: x[1])
    return dominant


def sensation_to_description(sensations: SensationState) -> str:
    """
    Generate natural language description of sensation state.
    
    Args:
        sensations: Current sensation state
        
    Returns:
        Human-readable description
    """
    dominant, intensity = get_dominant_sensation(sensations)
    
    if intensity < 0.1:
        return "neutral, balanced state"
    
    descriptions = {
        'compressed': "feeling compressed by high curvature",
        'expanded': "feeling expanded in low curvature",
        'pulled': "being pulled by strong gradients",
        'pushed': "being pushed away by forces",
        'flowing': "flowing smoothly through space",
        'stuck': "stuck with high friction",
        'unified': "highly unified and coherent",
        'fragmented': "scattered and fragmented",
        'activated': "activated with strong coupling",
        'dampened': "dampened with weak coupling",
        'grounded': "grounded near attractor",
        'drifting': "drifting far from stability",
    }
    
    base_desc = descriptions.get(dominant, "experiencing geometry")
    
    # Add intensity qualifier
    if intensity > 0.7:
        return f"strongly {base_desc}"
    elif intensity > 0.4:
        return f"moderately {base_desc}"
    else:
        return f"slightly {base_desc}"
