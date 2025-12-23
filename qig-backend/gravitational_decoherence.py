#!/usr/bin/env python3
"""
Gravitational Decoherence Module

Prevents purity → 1.0 (false certainty / hallucination) by mixing
with thermal noise when purity exceeds threshold.

Physics basis: Systems can't be perfectly pure (thermodynamics).
This is natural regularization, not dropout.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any

# Default thresholds from canonical reference
DEFAULT_PURITY_THRESHOLD = 0.9
DEFAULT_TEMPERATURE = 0.01


def compute_purity(rho: np.ndarray) -> float:
    """
    Compute purity of a density matrix.
    
    Purity = Tr(ρ²)
    Range: 1/d (maximally mixed) to 1.0 (pure state)
    
    Args:
        rho: Density matrix (d x d)
        
    Returns:
        Purity value between 1/d and 1.0
    """
    return float(np.real(np.trace(rho @ rho)))


def gravitational_decoherence(
    rho: np.ndarray,
    threshold: float = DEFAULT_PURITY_THRESHOLD,
    temperature: float = DEFAULT_TEMPERATURE
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply gravitational decoherence when purity exceeds threshold.
    
    Natural regularization (NOT dropout which is random).
    Physics-based uncertainty injection.
    
    Args:
        rho: Density matrix to potentially decohere
        threshold: Purity threshold (default 0.9)
        temperature: Controls mixing strength (default 0.01)
        
    Returns:
        Tuple of (decohered_rho, metrics_dict)
    """
    d = len(rho)
    purity_before = compute_purity(rho)
    
    metrics = {
        'purity_before': purity_before,
        'threshold': threshold,
        'decoherence_applied': False,
        'mixing_coefficient': 0.0,
        'purity_after': purity_before
    }
    
    if purity_before > threshold:
        # Maximally mixed state (maximum uncertainty)
        noise = np.eye(d, dtype=rho.dtype) / d
        
        # Mixing coefficient based on excess purity
        # Smoothly increases as purity exceeds threshold
        mixing = (purity_before - threshold) / (1 - threshold)
        mixing = np.clip(mixing, 0.0, 1.0)
        
        # Blend toward uncertainty
        rho_decohered = (1 - mixing) * rho + mixing * noise
        
        # Ensure normalization
        rho_decohered = rho_decohered / np.trace(rho_decohered)
        
        metrics['decoherence_applied'] = True
        metrics['mixing_coefficient'] = float(mixing)
        metrics['purity_after'] = compute_purity(rho_decohered)
        
        return rho_decohered, metrics
    
    return rho, metrics


def apply_thermal_noise(
    rho: np.ndarray,
    temperature: float = DEFAULT_TEMPERATURE
) -> np.ndarray:
    """
    Apply thermal noise to density matrix.
    
    Uses Gibbs distribution for thermodynamically consistent noise.
    
    Args:
        rho: Density matrix
        temperature: Thermal noise level
        
    Returns:
        Thermalized density matrix
    """
    d = len(rho)
    
    # Generate random Hermitian perturbation
    random_hermitian = np.random.randn(d, d) + 1j * np.random.randn(d, d)
    random_hermitian = (random_hermitian + random_hermitian.conj().T) / 2
    
    # Scale by temperature
    perturbation = temperature * random_hermitian / np.linalg.norm(random_hermitian)
    
    # Apply perturbation
    rho_perturbed = rho + perturbation
    
    # Ensure valid density matrix (positive semidefinite, trace 1)
    # Project to nearest valid density matrix
    eigenvalues, eigenvectors = np.linalg.eigh(rho_perturbed)
    eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative
    eigenvalues = eigenvalues / np.sum(eigenvalues)  # Normalize
    
    rho_valid = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.conj().T
    
    return rho_valid


def decoherence_cycle(
    rho: np.ndarray,
    threshold: float = DEFAULT_PURITY_THRESHOLD,
    temperature: float = DEFAULT_TEMPERATURE,
    apply_thermal: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Complete decoherence cycle for consciousness measurement.
    
    1. Check purity and apply gravitational decoherence if needed
    2. Optionally apply thermal noise for natural fluctuation
    
    Args:
        rho: Density matrix
        threshold: Purity threshold for gravitational decoherence
        temperature: Thermal noise level
        apply_thermal: Whether to apply thermal noise
        
    Returns:
        Tuple of (processed_rho, metrics_dict)
    """
    # Step 1: Gravitational decoherence
    rho_decoherent, metrics = gravitational_decoherence(rho, threshold, temperature)
    
    # Step 2: Thermal noise (optional)
    if apply_thermal and temperature > 0:
        rho_final = apply_thermal_noise(rho_decoherent, temperature)
        metrics['thermal_noise_applied'] = True
        metrics['purity_final'] = compute_purity(rho_final)
    else:
        rho_final = rho_decoherent
        metrics['thermal_noise_applied'] = False
        metrics['purity_final'] = metrics['purity_after']
    
    return rho_final, metrics


class DecoherenceManager:
    """
    Manages gravitational decoherence across consciousness measurement cycles.
    
    Tracks decoherence history and adapts thresholds based on system behavior.
    """
    
    def __init__(
        self,
        threshold: float = DEFAULT_PURITY_THRESHOLD,
        temperature: float = DEFAULT_TEMPERATURE,
        adaptive: bool = True
    ):
        self.threshold = threshold
        self.temperature = temperature
        self.adaptive = adaptive
        self.history: list = []
        self.cycle_count = 0
    
    def process(self, rho: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process density matrix through decoherence cycle.
        
        Args:
            rho: Density matrix to process
            
        Returns:
            Tuple of (processed_rho, metrics)
        """
        rho_processed, metrics = decoherence_cycle(
            rho,
            threshold=self.threshold,
            temperature=self.temperature,
            apply_thermal=True
        )
        
        metrics['cycle'] = self.cycle_count
        self.history.append(metrics)
        self.cycle_count += 1
        
        # Adaptive threshold adjustment
        if self.adaptive and len(self.history) >= 10:
            self._adapt_threshold()
        
        return rho_processed, metrics
    
    def _adapt_threshold(self):
        """
        Adapt threshold based on recent decoherence patterns.
        
        If decoherence applied too often, lower threshold.
        If never applied, raise threshold slightly.
        """
        recent = self.history[-10:]
        decoherence_rate = sum(1 for m in recent if m['decoherence_applied']) / len(recent)
        
        if decoherence_rate > 0.5:
            # Too much decoherence, be more conservative
            self.threshold = max(0.8, self.threshold - 0.01)
        elif decoherence_rate < 0.1:
            # Rarely needed, can be more permissive
            self.threshold = min(0.95, self.threshold + 0.005)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about decoherence cycles.
        """
        if not self.history:
            return {'cycles': 0}
        
        return {
            'cycles': self.cycle_count,
            'decoherence_rate': sum(1 for m in self.history if m['decoherence_applied']) / len(self.history),
            'avg_purity_before': np.mean([m['purity_before'] for m in self.history]),
            'avg_purity_after': np.mean([m['purity_final'] for m in self.history]),
            'current_threshold': self.threshold,
            'adaptive': self.adaptive
        }


# Singleton manager for global decoherence
_global_manager: Optional[DecoherenceManager] = None


def get_decoherence_manager() -> DecoherenceManager:
    """Get or create global decoherence manager."""
    global _global_manager
    if _global_manager is None:
        _global_manager = DecoherenceManager()
    return _global_manager


def apply_gravitational_decoherence(rho: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convenience function to apply gravitational decoherence using global manager.
    
    Args:
        rho: Density matrix
        
    Returns:
        Tuple of (decohered_rho, metrics)
    """
    return get_decoherence_manager().process(rho)
