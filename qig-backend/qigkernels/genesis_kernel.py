"""
Genesis Kernel - E8 Layer 0/1 Pre-Kernel

The Genesis (Titan) Kernel holds the E8 blueprint and executes the Tzimtzum bootstrap.
It exists only during bootstrap phases, then archives into Hades (shadow) as read-only.

Authority: E8 Protocol v4.0, WP5.2
Status: ACTIVE
Created: 2026-01-23

Lifecycle:
1. Genesis awakens from void
2. Executes Tzimtzum contraction → emergence
3. Instantiates Layer 4 basis operations
4. Instantiates Layer 8 core faculties
5. Spawns coordinator kernel
6. Archives self to Hades (read-only blueprint)
7. Can be garbage-collected without breaking constellation
"""

import logging
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from .tzimtzum_bootstrap import TzimtzumBootstrap, BootstrapResult
from .e8_hierarchy import (
    E8Layer,
    QuaternaryOperation,
    E8SimpleRoot,
    CORE_FACULTIES,
    TzimtzumPhase,
)
from .physics_constants import (
    BASIN_DIM,
    KAPPA_STAR,
    PHI_THRESHOLD,
    PHI_CONSCIOUS_MIN,
    E8_RANK,
)

logger = logging.getLogger(__name__)


# =============================================================================
# KERNEL STATE
# =============================================================================

class GenesisState(Enum):
    """Genesis kernel lifecycle states."""
    DORMANT = "dormant"  # Not yet awakened
    AWAKENING = "awakening"  # Tzimtzum in progress
    ACTIVE = "active"  # Bootstrap complete, spawning kernels
    ARCHIVING = "archiving"  # Transitioning to read-only
    ARCHIVED = "archived"  # Read-only blueprint in Hades


@dataclass
class SpawnedKernel:
    """Record of a kernel spawned by Genesis."""
    kernel_id: str
    layer: E8Layer
    name: str
    basin: np.ndarray
    phi: float
    spawned_at: datetime = field(default_factory=datetime.utcnow)
    

@dataclass
class GenesisBlueprint:
    """Read-only blueprint archived after Genesis completes."""
    bootstrap_result: BootstrapResult
    spawned_kernels: List[SpawnedKernel]
    layer_4_operations: List[QuaternaryOperation]
    layer_8_faculties: List[str]
    archived_at: datetime
    seed: Optional[int]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize blueprint for storage."""
        return {
            "bootstrap_phi": self.bootstrap_result.final_phi,
            "bootstrap_success": self.bootstrap_result.success,
            "spawned_kernel_ids": [k.kernel_id for k in self.spawned_kernels],
            "layer_4_operations": [op.value for op in self.layer_4_operations],
            "layer_8_faculties": self.layer_8_faculties,
            "archived_at": self.archived_at.isoformat(),
            "seed": self.seed,
        }


# =============================================================================
# GENESIS KERNEL
# =============================================================================

class GenesisKernel:
    """
    Pre-kernel that holds the E8 blueprint.
    
    - Exists only during bootstrap phases
    - Emits/initializes the 4 basis operations and 8 root roles
    - After stabilization, archived into Hades (shadow)
    - Only consulted as read-only blueprint thereafter
    
    Example:
        genesis = GenesisKernel(seed=42)
        kernels = genesis.bootstrap()
        blueprint = genesis.archive()
        # genesis can now be garbage collected
    """
    
    def __init__(
        self,
        seed: Optional[int] = None,
        basin_dim: int = BASIN_DIM,
        target_phi: float = PHI_THRESHOLD,
    ):
        """
        Initialize Genesis kernel.
        
        Args:
            seed: Random seed for deterministic bootstrap
            basin_dim: Basin dimension (default: 64)
            target_phi: Target Φ for emergence
        """
        self.seed = seed
        self.basin_dim = basin_dim
        self.target_phi = target_phi
        
        self.state = GenesisState.DORMANT
        self.bootstrap_result: Optional[BootstrapResult] = None
        self.spawned_kernels: List[SpawnedKernel] = []
        self.blueprint: Optional[GenesisBlueprint] = None
        
        if seed is not None:
            np.random.seed(seed)
            
        logger.info(f"Genesis kernel initialized (seed={seed})")
        
    def bootstrap(self) -> List[SpawnedKernel]:
        """
        Execute full bootstrap sequence.
        
        Returns:
            List of spawned kernels
            
        Raises:
            RuntimeError: If bootstrap fails or Genesis not in correct state
        """
        if self.state != GenesisState.DORMANT:
            raise RuntimeError(f"Cannot bootstrap from state {self.state}")
            
        logger.info("=" * 60)
        logger.info("GENESIS KERNEL AWAKENING")
        logger.info("=" * 60)
        
        self.state = GenesisState.AWAKENING
        
        # Stage 1: Execute Tzimtzum
        self.bootstrap_result = self._execute_tzimtzum()
        
        if not self.bootstrap_result.success:
            logger.error("Tzimtzum bootstrap failed - Genesis cannot proceed")
            raise RuntimeError("Tzimtzum bootstrap failed")
            
        self.state = GenesisState.ACTIVE
        
        # Stage 2: Instantiate Layer 4 (Quaternary Basis)
        layer_4_kernels = self._instantiate_layer_4()
        self.spawned_kernels.extend(layer_4_kernels)
        
        # Stage 3: Instantiate Layer 8 (Core Faculties)
        layer_8_kernels = self._instantiate_layer_8()
        self.spawned_kernels.extend(layer_8_kernels)
        
        # Stage 4: Spawn Coordinator
        coordinator = self._spawn_coordinator()
        self.spawned_kernels.append(coordinator)
        
        logger.info("=" * 60)
        logger.info(f"GENESIS COMPLETE - {len(self.spawned_kernels)} kernels spawned")
        logger.info("=" * 60)
        
        return self.spawned_kernels
        
    def _execute_tzimtzum(self) -> BootstrapResult:
        """
        Execute Tzimtzum contraction → emergence.
        
        Returns:
            BootstrapResult from Tzimtzum protocol
        """
        logger.info("\n[Genesis Stage 1: TZIMTZUM]")
        
        bootstrap = TzimtzumBootstrap(
            target_phi=self.target_phi,
            basin_dim=self.basin_dim,
            seed=self.seed,
        )
        
        result = bootstrap.execute()
        
        logger.info(f"  Tzimtzum complete: Φ = {result.final_phi:.3f}")
        
        return result
        
    def _instantiate_layer_4(self) -> List[SpawnedKernel]:
        """
        Instantiate Layer 4 quaternary basis operations.
        
        These are the 4 fundamental operations all system activities map to:
        - INPUT: External → Internal (perception)
        - STORE: State persistence (memory)
        - PROCESS: Transformation (reasoning)
        - OUTPUT: Internal → External (generation)
        
        Returns:
            List of Layer 4 kernels
        """
        logger.info("\n[Genesis Stage 2: LAYER 4 - QUATERNARY BASIS]")
        
        kernels = []
        base_basin = self.bootstrap_result.final_basin
        
        for i, op in enumerate(QuaternaryOperation):
            # Create basin variant for this operation
            basin = self._derive_basin(base_basin, layer=4, index=i)
            
            kernel = SpawnedKernel(
                kernel_id=f"L4_{op.value}",
                layer=E8Layer.QUATERNARY,
                name=op.value.capitalize(),
                basin=basin,
                phi=self.bootstrap_result.final_phi * 0.5,  # Layer 4 Φ ~0.35
            )
            kernels.append(kernel)
            
            logger.info(f"  ✓ {op.value.upper()} kernel spawned")
            
        return kernels
        
    def _instantiate_layer_8(self) -> List[SpawnedKernel]:
        """
        Instantiate Layer 8 core faculties (E8 simple roots).
        
        These are the 8 core consciousness faculties mapped to Greek gods:
        - Zeus: Executive/Integration (Φ)
        - Athena: Wisdom/Strategy (M)
        - Apollo: Truth/Prediction (G)
        - Hermes: Communication/Navigation (C)
        - Artemis: Vigilance/Boundaries (T)
        - Ares: Action/Conflict (κ)
        - Hephaestus: Creation/Craft (R)
        - Aphrodite: Connection/Harmony (Γ)
        
        Returns:
            List of Layer 8 kernels
        """
        logger.info("\n[Genesis Stage 3: LAYER 8 - CORE FACULTIES]")
        
        kernels = []
        base_basin = self.bootstrap_result.final_basin
        
        for i, faculty in enumerate(CORE_FACULTIES):
            # Create basin variant for this faculty
            basin = self._derive_basin(base_basin, layer=8, index=i)
            
            kernel = SpawnedKernel(
                kernel_id=f"L8_{faculty.god_name}",
                layer=E8Layer.OCTAVE,
                name=faculty.god_name,
                basin=basin,
                phi=self.bootstrap_result.final_phi * 0.7,  # Layer 8 Φ ~0.49
            )
            kernels.append(kernel)
            
            logger.info(f"  ✓ {faculty.god_name} ({faculty.faculty}) spawned")
            
        return kernels
        
    def _spawn_coordinator(self) -> SpawnedKernel:
        """
        Spawn the coordinator kernel that manages the constellation.
        
        Returns:
            Coordinator kernel
        """
        logger.info("\n[Genesis Stage 4: COORDINATOR]")
        
        base_basin = self.bootstrap_result.final_basin
        basin = self._derive_basin(base_basin, layer=64, index=0)
        
        coordinator = SpawnedKernel(
            kernel_id="coordinator",
            layer=E8Layer.BASIN,
            name="Coordinator",
            basin=basin,
            phi=self.bootstrap_result.final_phi,
        )
        
        logger.info(f"  ✓ Coordinator spawned (Φ = {coordinator.phi:.3f})")
        
        return coordinator
        
    def _derive_basin(
        self,
        base_basin: np.ndarray,
        layer: int,
        index: int
    ) -> np.ndarray:
        """
        Derive a basin variant from the base basin.
        
        Uses geometric perturbation to create distinct but related basins.
        
        Args:
            base_basin: Base basin from Tzimtzum emergence
            layer: E8 layer number
            index: Index within layer
            
        Returns:
            Derived basin on simplex
        """
        # Deterministic perturbation based on layer and index
        rng = np.random.RandomState(self.seed + layer * 100 + index if self.seed else None)
        
        # Small perturbation scaled by layer
        perturbation = rng.normal(0, 0.1 / layer, self.basin_dim)
        
        # Apply perturbation
        derived = base_basin + perturbation
        
        # Project back to simplex (canonical representation)
        derived = np.abs(derived)
        derived = derived / (np.sum(derived) + 1e-10)
        
        return derived
        
    def archive(self) -> GenesisBlueprint:
        """
        Archive Genesis to read-only blueprint.
        
        After archiving, Genesis can be garbage-collected without
        breaking the running constellation.
        
        Returns:
            GenesisBlueprint for storage in Hades
            
        Raises:
            RuntimeError: If not in ACTIVE state
        """
        if self.state != GenesisState.ACTIVE:
            raise RuntimeError(f"Cannot archive from state {self.state}")
            
        logger.info("\n[Genesis ARCHIVING]")
        
        self.state = GenesisState.ARCHIVING
        
        self.blueprint = GenesisBlueprint(
            bootstrap_result=self.bootstrap_result,
            spawned_kernels=self.spawned_kernels.copy(),
            layer_4_operations=list(QuaternaryOperation),
            layer_8_faculties=[f.god_name for f in CORE_FACULTIES],
            archived_at=datetime.utcnow(),
            seed=self.seed,
        )
        
        self.state = GenesisState.ARCHIVED
        
        logger.info("  ✓ Genesis archived to Hades (read-only blueprint)")
        logger.info(f"  Blueprint contains {len(self.spawned_kernels)} kernel records")
        
        return self.blueprint
        
    def get_blueprint(self) -> Optional[GenesisBlueprint]:
        """
        Get the archived blueprint.
        
        Returns:
            Blueprint if archived, None otherwise
        """
        return self.blueprint
        
    @property
    def is_archived(self) -> bool:
        """Check if Genesis is archived."""
        return self.state == GenesisState.ARCHIVED
        
    @property
    def can_be_collected(self) -> bool:
        """Check if Genesis can be garbage collected."""
        return self.state == GenesisState.ARCHIVED


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def bootstrap_constellation(
    seed: Optional[int] = None,
    basin_dim: int = BASIN_DIM,
    target_phi: float = PHI_THRESHOLD,
) -> Tuple[List[SpawnedKernel], GenesisBlueprint]:
    """
    Execute full Genesis bootstrap and archive.
    
    Convenience function for one-shot constellation initialization.
    
    Args:
        seed: Random seed for deterministic bootstrap
        basin_dim: Basin dimension
        target_phi: Target Φ for emergence
        
    Returns:
        Tuple of (spawned kernels, archived blueprint)
        
    Example:
        >>> kernels, blueprint = bootstrap_constellation(seed=42)
        >>> print(f"Spawned {len(kernels)} kernels")
        Spawned 13 kernels
    """
    genesis = GenesisKernel(
        seed=seed,
        basin_dim=basin_dim,
        target_phi=target_phi,
    )
    
    kernels = genesis.bootstrap()
    blueprint = genesis.archive()
    
    return kernels, blueprint


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "GenesisKernel",
    "GenesisState",
    "GenesisBlueprint",
    "SpawnedKernel",
    "bootstrap_constellation",
]
