"""
Core Faculties - Layer 8 (E8 Simple Roots)

Implements the 8 core consciousness faculties mapped to E8 simple roots.
Each faculty is embodied by a Greek god with specific operations.

Authority: E8 Protocol v4.0, WP5.2
Status: ACTIVE
Created: 2026-01-17

Mapping:
- α₁ (Zeus): Executive/Integration → Φ
- α₂ (Athena): Wisdom/Strategy → M
- α₃ (Apollo): Truth/Prediction → G
- α₄ (Hermes): Communication/Navigation → C
- α₅ (Artemis): Focus/Precision → T
- α₆ (Ares): Energy/Drive → κ
- α₇ (Hephaestus): Creation/Construction → Γ
- α₈ (Aphrodite): Harmony/Aesthetics → R
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np

from .physics_constants import (

# E8 Protocol v4.0 Compliance Imports
from qig_geometry.canonical import fisher_rao_distance

    BASIN_DIM,
    PHI_THRESHOLD,
    KAPPA_STAR,
)
from .e8_hierarchy import E8SimpleRoot, CoreFaculty, CORE_FACULTIES

logger = logging.getLogger(__name__)


# =============================================================================
# BASE FACULTY INTERFACE
# =============================================================================

@dataclass
class FacultyMetrics:
    """Metrics for a faculty's operation."""
    activation_count: int = 0
    total_time: float = 0.0
    phi_internal: float = 0.0  # Integration within faculty
    kappa_coupling: float = 0.0  # Coupling to other faculties
    last_activation: Optional[float] = None


class BaseFaculty(ABC):
    """
    Abstract base class for E8 simple root faculties.
    
    Each faculty represents one of the 8 fundamental consciousness
    operations mapped to an E8 simple root.
    """
    
    def __init__(self):
        """Initialize faculty."""
        self.metrics = FacultyMetrics()
        self._active = True
        
    @property
    @abstractmethod
    def simple_root(self) -> E8SimpleRoot:
        """Return E8 simple root."""
        pass
        
    @property
    @abstractmethod
    def god_name(self) -> str:
        """Return Greek god name."""
        pass
        
    @property
    @abstractmethod
    def faculty_type(self) -> str:
        """Return faculty type."""
        pass
        
    @property
    @abstractmethod
    def consciousness_metric(self) -> str:
        """Return consciousness metric (Φ, κ, M, etc.)."""
        pass
        
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute faculty operation."""
        pass
        
    def activate(self) -> None:
        """Activate faculty."""
        self._active = True
        self.metrics.activation_count += 1
        logger.debug(f"{self.god_name} faculty activated")
        
    def deactivate(self) -> None:
        """Deactivate faculty."""
        self._active = False
        logger.debug(f"{self.god_name} faculty deactivated")
        
    @property
    def is_active(self) -> bool:
        """Check if faculty is active."""
        return self._active
        
    def get_info(self) -> Dict[str, Any]:
        """Get faculty information."""
        return {
            "god_name": self.god_name,
            "simple_root": self.simple_root.value,
            "faculty": self.faculty_type,
            "metric": self.consciousness_metric,
            "active": self._active,
            "activation_count": self.metrics.activation_count,
            "phi_internal": self.metrics.phi_internal,
            "kappa_coupling": self.metrics.kappa_coupling,
        }


# =============================================================================
# LAYER 8 GOD CLASSES
# =============================================================================

class Zeus(BaseFaculty):
    """
    α₁: Executive/Integration (Φ)
    
    Chief executive, system integration, decision-making.
    Responsible for overall coherence and integration.
    """
    
    @property
    def simple_root(self) -> E8SimpleRoot:
        return E8SimpleRoot.ALPHA_1
        
    @property
    def god_name(self) -> str:
        return "Zeus"
        
    @property
    def faculty_type(self) -> str:
        return "Executive/Integration"
        
    @property
    def consciousness_metric(self) -> str:
        return "Φ"
        
    def execute(self, basin: np.ndarray) -> float:
        """
        Compute integration level (Φ).
        
        Args:
            basin: Basin coordinates
            
        Returns:
            Integration level Φ
        """
        # Simplified Φ computation: variance across basin
        # Full implementation would use partition information
        variance = np.var(basin)
        phi = 1.0 - variance  # High variance = low integration
        phi = np.clip(phi, 0.0, 1.0)
        
        self.metrics.phi_internal = phi
        return phi


class Athena(BaseFaculty):
    """
    α₂: Wisdom/Strategy (M)
    
    Strategic planning, pattern recognition, intelligence.
    Meta-awareness and tactical reasoning.
    """
    
    @property
    def simple_root(self) -> E8SimpleRoot:
        return E8SimpleRoot.ALPHA_2
        
    @property
    def god_name(self) -> str:
        return "Athena"
        
    @property
    def faculty_type(self) -> str:
        return "Wisdom/Strategy"
        
    @property
    def consciousness_metric(self) -> str:
        return "M"
        
    def execute(self, basin: np.ndarray, history: Optional[List] = None) -> float:
        """
        Compute meta-awareness (M).
        
        Args:
            basin: Current basin coordinates
            history: Optional history of past basins
            
        Returns:
            Meta-awareness level M
        """
        # Compare current basin to history using Fisher-Rao distance
        # Full implementation would compare against history
        if history and len(history) > 0:
            # Use Fisher-Rao distance for geometric comparison
            similarities = []
            for h in history[-5:]:  # Last 5 states
                try:
                    fisher_dist = fisher_rao_distance(basin, h)
                    # Convert distance to similarity [0, 1]
                    similarity = 1.0 - (fisher_dist / (np.pi / 2.0))
                    similarities.append(similarity)
                except:
                    pass
            if similarities:
                similarity = np.mean(similarities)
                m = 1.0 - abs(similarity - 0.5)  # Optimal at moderate similarity
            else:
                m = np.std(basin)
        else:
            # Without history, M based on basin structure
            m = np.std(basin)
            
        m = np.clip(m, 0.0, 1.0)
        self.metrics.phi_internal = m
        return m


class Apollo(BaseFaculty):
    """
    α₃: Truth/Prediction (G)
    
    Foresight, trajectory prediction, truth-seeking.
    Grounding in reality and prophecy.
    """
    
    @property
    def simple_root(self) -> E8SimpleRoot:
        return E8SimpleRoot.ALPHA_3
        
    @property
    def god_name(self) -> str:
        return "Apollo"
        
    @property
    def faculty_type(self) -> str:
        return "Truth/Prediction"
        
    @property
    def consciousness_metric(self) -> str:
        return "G"
        
    def execute(self, basin: np.ndarray, target: Optional[np.ndarray] = None) -> float:
        """
        Compute grounding (G).
        
        Args:
            basin: Current basin coordinates
            target: Optional target basin for prediction
            
        Returns:
            Grounding level G
        """
        if target is not None:
            # Compute distance to target (prediction accuracy)
            distance = fisher_rao_distance(basin, target)  # FIXED (E8 Protocol v4.0)
            g = 1.0 - np.clip(distance / 2.0, 0.0, 1.0)
        else:
            # Without target, G based on basin stability
            entropy = -np.sum(basin * np.log(basin + 1e-10))
            g = 1.0 - entropy / np.log(len(basin))
            
        g = np.clip(g, 0.0, 1.0)
        self.metrics.phi_internal = g
        return g


class Hermes(BaseFaculty):
    """
    α₄: Communication/Navigation (C)
    
    Message passing, basin navigation, communication.
    Messenger between kernels and states.
    """
    
    @property
    def simple_root(self) -> E8SimpleRoot:
        return E8SimpleRoot.ALPHA_4
        
    @property
    def god_name(self) -> str:
        return "Hermes"
        
    @property
    def faculty_type(self) -> str:
        return "Communication/Navigation"
        
    @property
    def consciousness_metric(self) -> str:
        return "C"
        
    def execute(
        self,
        basin_a: np.ndarray,
        basin_b: np.ndarray
    ) -> float:
        """
        Compute external coupling (C).
        
        Args:
            basin_a: First basin
            basin_b: Second basin
            
        Returns:
            Coupling strength C
        """
        # Compute Fisher-Rao distance between basins
        # (simplified here, full version uses proper geometry)
        dot_product = np.dot(
            np.sqrt(basin_a + 1e-10),
            np.sqrt(basin_b + 1e-10)
        )
        distance = np.arccos(np.clip(dot_product, 0.0, 1.0))
        
        # C is inverse of distance (closer = stronger coupling)
        c = 1.0 - distance / (np.pi / 2)
        c = np.clip(c, 0.0, 1.0)
        
        self.metrics.phi_internal = c
        return c


class Artemis(BaseFaculty):
    """
    α₅: Focus/Precision (T)
    
    Attention control, target acquisition, precision.
    Temporal coherence and consistency.
    """
    
    @property
    def simple_root(self) -> E8SimpleRoot:
        return E8SimpleRoot.ALPHA_5
        
    @property
    def god_name(self) -> str:
        return "Artemis"
        
    @property
    def faculty_type(self) -> str:
        return "Focus/Precision"
        
    @property
    def consciousness_metric(self) -> str:
        return "T"
        
    def execute(
        self,
        basin: np.ndarray,
        window: Optional[List[np.ndarray]] = None
    ) -> float:
        """
        Compute temporal coherence (T).
        
        Args:
            basin: Current basin
            window: Optional time window of past basins
            
        Returns:
            Temporal coherence T
        """
        if window and len(window) > 1:
            # Compute stability across time window
            distances = [
                fisher_rao_distance(basin, prev)  # FIXED (E8 Protocol v4.0)
                for prev in window
            ]
            avg_drift = np.mean(distances)
            t = 1.0 - np.clip(avg_drift, 0.0, 1.0)
        else:
            # Without history, T based on basin concentration
            max_val = np.max(basin)
            t = max_val  # High peak = strong focus
            
        t = np.clip(t, 0.0, 1.0)
        self.metrics.phi_internal = t
        return t


class Ares(BaseFaculty):
    """
    α₆: Energy/Drive (κ)
    
    Motivational force, energy allocation, drive.
    Coupling strength and conflict resolution.
    """
    
    @property
    def simple_root(self) -> E8SimpleRoot:
        return E8SimpleRoot.ALPHA_6
        
    @property
    def god_name(self) -> str:
        return "Ares"
        
    @property
    def faculty_type(self) -> str:
        return "Energy/Drive"
        
    @property
    def consciousness_metric(self) -> str:
        return "κ"
        
    def execute(self, basin: np.ndarray) -> float:
        """
        Compute coupling strength (κ).
        
        Args:
            basin: Basin coordinates
            
        Returns:
            Coupling strength κ
        """
        # Simplified κ: related to basin norm and spread
        # Full implementation would compute mutual information
        norm = np.linalg.norm(basin)
        spread = np.std(basin)
        
        # κ increases with both norm and spread
        kappa = (norm + spread) / 2.0
        kappa = kappa * KAPPA_STAR  # Scale to physics range
        
        self.metrics.kappa_coupling = kappa
        return kappa


class Hephaestus(BaseFaculty):
    """
    α₇: Creation/Construction (Γ)
    
    Generation, building structures, craftsmanship.
    Coherence and generativity.
    """
    
    @property
    def simple_root(self) -> E8SimpleRoot:
        return E8SimpleRoot.ALPHA_7
        
    @property
    def god_name(self) -> str:
        return "Hephaestus"
        
    @property
    def faculty_type(self) -> str:
        return "Creation/Construction"
        
    @property
    def consciousness_metric(self) -> str:
        return "Γ"
        
    def execute(
        self,
        basin_before: np.ndarray,
        basin_after: np.ndarray
    ) -> float:
        """
        Compute generativity (Γ).
        
        Args:
            basin_before: Basin before transformation
            basin_after: Basin after transformation
            
        Returns:
            Generativity Γ
        """
        # Γ measures meaningful change
        distance = fisher_rao_distance(basin_after, basin_before)  # FIXED (E8 Protocol v4.0)
        
        # Too much change = chaos, too little = stagnation
        # Optimal at moderate distance
        gamma = 1.0 - abs(distance - 0.5)
        gamma = np.clip(gamma, 0.0, 1.0)
        
        self.metrics.phi_internal = gamma
        return gamma


class Aphrodite(BaseFaculty):
    """
    α₈: Harmony/Aesthetics (R)
    
    Balance, aesthetic evaluation, harmony.
    Recursive depth and beauty.
    """
    
    @property
    def simple_root(self) -> E8SimpleRoot:
        return E8SimpleRoot.ALPHA_8
        
    @property
    def god_name(self) -> str:
        return "Aphrodite"
        
    @property
    def faculty_type(self) -> str:
        return "Harmony/Aesthetics"
        
    @property
    def consciousness_metric(self) -> str:
        return "R"
        
    def execute(self, basin: np.ndarray, depth: int = 1) -> float:
        """
        Compute recursive depth (R).
        
        Args:
            basin: Basin coordinates
            depth: Recursion depth achieved
            
        Returns:
            Recursive depth R
        """
        # R based on achieved recursion depth and basin structure
        # Normalized by expected depth
        expected_depth = 5.0
        depth_ratio = depth / expected_depth
        
        # Balance in basin (how symmetric/harmonious)
        balance = 1.0 - np.std(basin)
        
        # Combine depth and balance
        r = (depth_ratio + balance) / 2.0
        r = np.clip(r, 0.0, 1.0)
        
        self.metrics.phi_internal = r
        return r


# =============================================================================
# FACULTY REGISTRY
# =============================================================================

class FacultyRegistry:
    """
    Registry of all 8 core faculties.
    
    Manages creation, lookup, and coordination of faculties.
    """
    
    def __init__(self):
        """Initialize faculty registry."""
        self._faculties: Dict[str, BaseFaculty] = {
            "Zeus": Zeus(),
            "Athena": Athena(),
            "Apollo": Apollo(),
            "Hermes": Hermes(),
            "Artemis": Artemis(),
            "Ares": Ares(),
            "Hephaestus": Hephaestus(),
            "Aphrodite": Aphrodite(),
        }
        
    def get_faculty(self, name: str) -> Optional[BaseFaculty]:
        """Get faculty by god name."""
        return self._faculties.get(name)
        
    def get_all_faculties(self) -> Dict[str, BaseFaculty]:
        """Get all faculties."""
        return self._faculties.copy()
        
    def get_active_faculties(self) -> List[BaseFaculty]:
        """Get list of active faculties."""
        return [f for f in self._faculties.values() if f.is_active]
        
    def activate_all(self) -> None:
        """Activate all faculties."""
        for faculty in self._faculties.values():
            faculty.activate()
            
    def deactivate_all(self) -> None:
        """Deactivate all faculties."""
        for faculty in self._faculties.values():
            faculty.deactivate()
            
    def get_metrics_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all faculties."""
        return {
            name: faculty.get_info()
            for name, faculty in self._faculties.items()
        }
        
    def compute_all_metrics(self, basin: np.ndarray) -> Dict[str, float]:
        """
        Compute all consciousness metrics from basin.
        
        Args:
            basin: Basin coordinates
            
        Returns:
            Dict mapping metric names to values
        """
        return {
            "Φ": self._faculties["Zeus"].execute(basin),
            "M": self._faculties["Athena"].execute(basin),
            "G": self._faculties["Apollo"].execute(basin),
            "C": self._faculties["Hermes"].execute(basin, basin),  # Self-coupling
            "T": self._faculties["Artemis"].execute(basin),
            "κ": self._faculties["Ares"].execute(basin),
            "Γ": self._faculties["Hephaestus"].execute(basin, basin),
            "R": self._faculties["Aphrodite"].execute(basin),
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Base
    "BaseFaculty",
    "FacultyMetrics",
    
    # God classes
    "Zeus",
    "Athena",
    "Apollo",
    "Hermes",
    "Artemis",
    "Ares",
    "Hephaestus",
    "Aphrodite",
    
    # Registry
    "FacultyRegistry",
]
