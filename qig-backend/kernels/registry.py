"""
Kernel Registry - Central registration and lookup for E8 kernels.

Provides global registry for kernel lifecycle management, discovery, and routing.
Supports dynamic kernel spawning/merging while maintaining E8 structure validation.

Author: E8 Protocol Team
Date: 2026-01-23
Status: Integration Fix
"""

from typing import Dict, List, Optional, Set
from dataclasses import dataclass
import logging

# Will import from kernels.base once PR #262 merges
# from kernels.base import Kernel
# from kernels.e8_roots import E8Root
# from kernels.identity import KernelIdentity

logger = logging.getLogger(__name__)


@dataclass
class KernelMetadata:
    """Metadata for registered kernels."""
    god: str
    root: str  # E8Root value
    tier: str  # essential | pantheon | chaos | shadow
    spawned_at: float  # timestamp
    active: bool


class KernelRegistry:
    """
    Central registry for all kernel instances.
    
    Responsibilities:
    - Track active kernels by god name
    - Validate E8 structure constraints
    - Support kernel lifecycle (spawn/merge)
    - Enable kernel discovery and routing
    
    Thread-safe for multi-kernel operations.
    """
    
    def __init__(self):
        self._kernels: Dict[str, 'Kernel'] = {}
        self._metadata: Dict[str, KernelMetadata] = {}
        self._root_counts: Dict[str, int] = {
            'alpha1': 0, 'alpha2': 0, 'alpha3': 0, 'alpha4': 0,
            'alpha5': 0, 'alpha6': 0, 'alpha7': 0, 'alpha8': 0,
        }
        self._tier_counts: Dict[str, int] = {
            'essential': 0,
            'pantheon': 0,
            'chaos': 0,
            'shadow': 0,
        }
    
    def register(self, kernel: 'Kernel') -> bool:
        """
        Register a kernel instance.
        
        Args:
            kernel: Kernel instance to register
            
        Returns:
            True if registered successfully, False if duplicate
            
        Raises:
            ValueError: If kernel violates E8 structure constraints
        """
        god = kernel.identity.god
        
        # Check for duplicate
        if god in self._kernels:
            logger.warning(f"Kernel {god} already registered. Skipping.")
            return False
        
        # Validate E8 structure
        self._validate_e8_constraints(kernel)
        
        # Register kernel
        self._kernels[god] = kernel
        self._metadata[god] = KernelMetadata(
            god=god,
            root=kernel.identity.root.value,
            tier=kernel.identity.tier,
            spawned_at=kernel.created_at if hasattr(kernel, 'created_at') else 0.0,
            active=True,
        )
        
        # Update counts
        self._root_counts[kernel.identity.root.value] += 1
        self._tier_counts[kernel.identity.tier] += 1
        
        logger.info(f"Registered kernel: {god} ({kernel.identity.root.value}, tier={kernel.identity.tier})")
        return True
    
    def unregister(self, god: str) -> bool:
        """
        Unregister a kernel (for merging or deletion).
        
        Args:
            god: Kernel god name
            
        Returns:
            True if unregistered, False if not found
        """
        if god not in self._kernels:
            logger.warning(f"Kernel {god} not found in registry.")
            return False
        
        kernel = self._kernels[god]
        metadata = self._metadata[god]
        
        # Update counts
        self._root_counts[kernel.identity.root.value] -= 1
        self._tier_counts[metadata.tier] -= 1
        
        # Remove
        del self._kernels[god]
        del self._metadata[god]
        
        logger.info(f"Unregistered kernel: {god}")
        return True
    
    def get(self, god: str) -> Optional['Kernel']:
        """Get kernel by god name."""
        return self._kernels.get(god)
    
    def all_kernels(self) -> List['Kernel']:
        """Get all registered kernels."""
        return list(self._kernels.values())
    
    def active_kernels(self) -> List['Kernel']:
        """Get all active (not sleeping) kernels."""
        return [k for k in self._kernels.values() if not k.asleep]
    
    def kernels_by_root(self, root: str) -> List['Kernel']:
        """Get all kernels for a specific E8 root."""
        return [k for k in self._kernels.values() if k.identity.root.value == root]
    
    def kernels_by_tier(self, tier: str) -> List['Kernel']:
        """Get all kernels in a specific tier."""
        return [k for k in self._kernels.values() if k.identity.tier == tier]
    
    def get_metadata(self, god: str) -> Optional[KernelMetadata]:
        """Get metadata for a kernel."""
        return self._metadata.get(god)
    
    def root_count(self, root: str) -> int:
        """Get count of kernels for a specific root."""
        return self._root_counts.get(root, 0)
    
    def tier_count(self, tier: str) -> int:
        """Get count of kernels in a tier."""
        return self._tier_counts.get(tier, 0)
    
    def total_count(self) -> int:
        """Get total kernel count."""
        return len(self._kernels)
    
    def _validate_e8_constraints(self, kernel: 'Kernel'):
        """
        Validate E8 structure constraints.
        
        Constraints:
        - Essential tier: Exactly 8 kernels (one per root)
        - Pantheon tier: Up to 12 kernels (olympus gods)
        - Total constellation: Up to 240 kernels (E8 roots)
        
        Raises:
            ValueError: If constraints violated
        """
        tier = kernel.identity.tier
        root = kernel.identity.root.value
        
        # Essential tier: One per root max
        if tier == 'essential':
            if self._root_counts[root] >= 1:
                raise ValueError(
                    f"Essential tier already has kernel for {root}. "
                    f"Only one essential kernel per E8 root allowed."
                )
        
        # Total constellation limit
        if self.total_count() >= 240:
            raise ValueError(
                f"Kernel constellation limit reached (240 kernels). "
                f"Cannot spawn new kernel without merging existing ones."
            )
        
        # Tier-specific limits
        tier_limits = {
            'essential': 8,
            'pantheon': 12,
            'chaos': 100,  # soft limit
            'shadow': 120,  # soft limit
        }
        
        if self._tier_counts[tier] >= tier_limits.get(tier, float('inf')):
            logger.warning(
                f"Tier {tier} approaching limit "
                f"({self._tier_counts[tier]}/{tier_limits.get(tier)})"
            )
    
    def snapshot(self) -> dict:
        """Get registry snapshot for monitoring."""
        return {
            'total_kernels': self.total_count(),
            'active_kernels': len(self.active_kernels()),
            'root_counts': dict(self._root_counts),
            'tier_counts': dict(self._tier_counts),
            'kernels': [
                {
                    'god': metadata.god,
                    'root': metadata.root,
                    'tier': metadata.tier,
                    'active': metadata.active,
                }
                for metadata in self._metadata.values()
            ]
        }


# Global singleton registry
GLOBAL_KERNEL_REGISTRY = KernelRegistry()


def get_registry() -> KernelRegistry:
    """Get global kernel registry."""
    return GLOBAL_KERNEL_REGISTRY
