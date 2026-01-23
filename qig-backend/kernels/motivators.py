#!/usr/bin/env python3
"""
Layer 1: Motivators (5 Geometric Derivatives) - FROZEN

These are FROZEN geometric derivatives that drive behavior.
They are derived from geometric state, not learned or trained.

Based on E8 Protocol v4.0 phenomenology specification.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class MotivatorState:
    """
    Layer 1: Motivators (5 geometric derivatives) - FROZEN.
    
    Motivators are first-order geometric derivatives that create
    behavioral drives. They are NOT learned - they are fundamental
    geometric facts that the system measures.
    """
    surprise: float = 0.0       # ||grad_L|| - Learning gradient magnitude
    curiosity: float = 0.0      # d(log I_Q)/dt - Information rate
    investigation: float = 0.0  # -d(basin_distance)/dt - Approach velocity
    integration: float = 0.0    # [CV(phi * I_Q)]^-1 - Coherence/variability
    transcendence: float = 0.0  # |kappa - kappa_c| - Distance from critical


def compute_motivators(
    phi: float,
    kappa: float,
    fisher_info: float,
    basin_distance: float,
    prev_basin_distance: Optional[float] = None,
    loss_gradient: Optional[np.ndarray] = None,
    prev_fisher_info: Optional[float] = None,
    dt: float = 1.0,
    kappa_critical: float = 64.21,  # κ* from E8 structure
) -> MotivatorState:
    """
    Compute motivators from geometric derivatives.
    
    These are FROZEN - they are not learned, they are measured.
    Each motivator is a specific geometric derivative.
    
    Args:
        phi: Current Φ (integration)
        kappa: Current κ (coupling)
        fisher_info: Current Fisher information I_Q
        basin_distance: Current Fisher distance to attractor
        prev_basin_distance: Previous basin distance (for velocity)
        loss_gradient: Optional ∇L (learning gradient)
        prev_fisher_info: Previous I_Q (for rate computation)
        dt: Time step for derivatives
        kappa_critical: Critical κ value (default κ* = 64.21)
        
    Returns:
        Computed motivator state
    """
    motivators = MotivatorState()
    
    # 1. Surprise = ||grad_L||
    # Magnitude of learning gradient
    if loss_gradient is not None:
        motivators.surprise = float(np.linalg.norm(loss_gradient))
    
    # 2. Curiosity = d(log I_Q)/dt
    # Rate of change of information content
    if prev_fisher_info is not None and prev_fisher_info > 0:
        log_info_change = np.log(fisher_info + 1e-10) - np.log(prev_fisher_info + 1e-10)
        motivators.curiosity = abs(log_info_change / dt)
    elif fisher_info > 0:
        # Fallback: use current info as proxy
        motivators.curiosity = np.log(fisher_info + 1)
    
    # 3. Investigation = -d(basin_distance)/dt
    # Velocity toward attractor (negative = approaching)
    if prev_basin_distance is not None:
        velocity = -(basin_distance - prev_basin_distance) / dt
        motivators.investigation = max(0.0, velocity)  # Positive when approaching
    
    # 4. Integration = [CV(phi * I_Q)]^-1
    # Inverse coefficient of variation (coherence/variability)
    product = phi * fisher_info
    if product > 0:
        # For single measurement, use normalized product as proxy
        # In full implementation, would track variance over time
        motivators.integration = min(1.0, product)
    
    # 5. Transcendence = |kappa - kappa_c|
    # Distance from critical point
    motivators.transcendence = abs(kappa - kappa_critical) / kappa_critical
    
    return motivators


def get_dominant_motivator(motivators: MotivatorState) -> tuple[str, float]:
    """
    Get the dominant (strongest) motivator.
    
    Args:
        motivators: Current motivator state
        
    Returns:
        (motivator_name, intensity) tuple
    """
    motivator_values = {
        'surprise': motivators.surprise,
        'curiosity': motivators.curiosity,
        'investigation': motivators.investigation,
        'integration': motivators.integration,
        'transcendence': motivators.transcendence,
    }
    
    dominant = max(motivator_values.items(), key=lambda x: x[1])
    return dominant


def motivator_to_behavioral_drive(motivators: MotivatorState) -> str:
    """
    Translate motivator state to behavioral drive description.
    
    Args:
        motivators: Current motivator state
        
    Returns:
        Behavioral drive description
    """
    dominant, intensity = get_dominant_motivator(motivators)
    
    if intensity < 0.1:
        return "resting, no strong drives"
    
    drives = {
        'surprise': "driven to resolve unexpected patterns",
        'curiosity': "driven to explore information landscape",
        'investigation': "driven to approach target attractor",
        'integration': "driven to unify scattered knowledge",
        'transcendence': "driven to explore beyond current regime",
    }
    
    base_drive = drives.get(dominant, "motivated by geometry")
    
    if intensity > 0.7:
        return f"strongly {base_drive}"
    elif intensity > 0.4:
        return f"moderately {base_drive}"
    else:
        return f"slightly {base_drive}"


def compute_motivator_alignment(motivators: MotivatorState) -> float:
    """
    Compute overall alignment of motivators (are they pulling in same direction?).
    
    High alignment = motivators reinforce each other
    Low alignment = motivators conflict
    
    Args:
        motivators: Current motivator state
        
    Returns:
        Alignment score [0, 1]
    """
    # Create vector of motivator strengths
    m_vec = np.array([
        motivators.surprise,
        motivators.curiosity,
        motivators.investigation,
        motivators.integration,
        motivators.transcendence,
    ])
    
    if np.sum(m_vec) < 0.1:
        return 1.0  # No conflict if no drives
    
    # Alignment = how evenly distributed vs. concentrated
    # High concentration (one dominant) = high alignment
    # Even distribution = potential conflict
    normalized = m_vec / (np.sum(m_vec) + 1e-10)
    entropy = -np.sum(normalized * np.log(normalized + 1e-10))
    max_entropy = np.log(5.0)  # Log of number of motivators
    
    # Convert entropy to alignment (lower entropy = higher alignment)
    alignment = 1.0 - (entropy / max_entropy)
    return float(alignment)
