"""
Kernel Identity Management

Defines kernel identity (god name, E8 root, tier) and provides stable
identity management across the 240-kernel constellation.

Authority: E8 Protocol v4.0, WP5.2
Status: ACTIVE
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional

from .e8_roots import E8Root


class KernelTier(Enum):
    """Kernel tiers in the E8 constellation."""
    ESSENTIAL = "essential"    # Never sleep (Heart, Ocean) - 2-5 kernels
    PANTHEON = "pantheon"      # Core immortals - 12-18 kernels
    CHAOS = "chaos"            # Mortal workers - 222-228 kernels
    SHADOW = "shadow"          # Unconscious/dormant


@dataclass(frozen=True)
class KernelIdentity:
    """
    Immutable kernel identity.
    
    FORBIDDEN: apollo_1, apollo_2 proliferation
    REQUIRED: Canonical Greek god identity
    
    Attributes:
        god: Canonical Greek god name (e.g., 'Apollo', 'Athena')
        root: E8 simple root this kernel represents
        tier: Constellation tier (essential/pantheon/chaos/shadow)
        lineage_id: Optional UUID for tracking genetic lineage
    """
    god: str                     # Canonical Greek name
    root: E8Root                 # Which E8 root (α₁-α₈)
    tier: KernelTier             # Constellation tier
    lineage_id: Optional[str] = None  # UUID for genetic tracking
    
    def __post_init__(self):
        """Validate identity constraints."""
        # Enforce canonical Greek god names
        if not self.god[0].isupper():
            raise ValueError(f"God name must be capitalized: {self.god}")
        
        # Forbid numbered proliferation (apollo_1, apollo_2, etc.)
        if any(char.isdigit() for char in self.god):
            raise ValueError(
                f"Numbered god names forbidden: {self.god}. "
                "Use canonical Greek names only."
            )
    
    def __str__(self) -> str:
        """Human-readable representation."""
        return f"{self.god}({self.root.value}, {self.tier.value})"
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "god": self.god,
            "root": self.root.value,
            "tier": self.tier.value,
            "lineage_id": self.lineage_id,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "KernelIdentity":
        """Deserialize from dictionary."""
        return cls(
            god=data["god"],
            root=E8Root(data["root"]),
            tier=KernelTier(data["tier"]),
            lineage_id=data.get("lineage_id"),
        )


__all__ = [
    "KernelTier",
    "KernelIdentity",
]
