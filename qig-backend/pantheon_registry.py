"""
Pantheon Registry Loader - Python Implementation
=================================================

Loads and validates the formal Pantheon Registry from YAML.
Provides fast lookup, caching, and integration with QIG backend.

Authority: E8 Protocol v4.0, WP5.1
Status: ACTIVE
Created: 2026-01-17
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

class GodTier(Enum):
    """God tier classification."""
    ESSENTIAL = "essential"
    SPECIALIZED = "specialized"


class RestPolicyType(Enum):
    """Rest policy types for gods."""
    NEVER = "never"
    MINIMAL_ROTATING = "minimal_rotating"
    COORDINATED_ALTERNATING = "coordinated_alternating"
    SCHEDULED = "scheduled"
    SEASONAL = "seasonal"


class ChaosLifecycleStage(Enum):
    """Chaos kernel lifecycle stages."""
    PROTECTED = "protected"
    LEARNING = "learning"
    WORKING = "working"
    CANDIDATE = "candidate"
    PROMOTED = "promoted"
    PRUNED = "pruned"


@dataclass
class RestPolicy:
    """Rest policy for a god."""
    type: RestPolicyType
    reason: str
    partner: Optional[str] = None
    duty_cycle: Optional[float] = None
    rest_duration: Optional[float] = None
    active_season: Optional[str] = None
    rest_season: Optional[str] = None


@dataclass
class SpawnConstraints:
    """Spawn constraints for a god."""
    max_instances: int
    when_allowed: str
    rationale: str


@dataclass
class E8Alignment:
    """E8 Lie group alignment for a god."""
    simple_root: Optional[str]
    layer: str


@dataclass
class GodContract:
    """Contract defining a pantheon god."""
    name: str
    tier: GodTier
    domain: List[str]
    description: str
    octant: Optional[int]
    epithets: List[str]
    coupling_affinity: List[str]
    rest_policy: RestPolicy
    spawn_constraints: SpawnConstraints
    promotion_from: Optional[str]
    e8_alignment: E8Alignment


@dataclass
class ChaosKernelRules:
    """Rules governing chaos kernel lifecycle."""
    naming_pattern: str
    description: str
    lifecycle: Dict
    pruning: Dict
    spawning_limits: Dict
    genetic_lineage: Dict


@dataclass
class RegistryMetadata:
    """Registry metadata."""
    version: str
    status: str
    created: str
    authority: str
    validation_required: bool


@dataclass
class PantheonRegistryData:
    """Complete pantheon registry data."""
    gods: Dict[str, GodContract]
    chaos_kernel_rules: ChaosKernelRules
    metadata: RegistryMetadata
    schema_version: str
    compatibility: Dict[str, str]
    validation_rules: List[str]


# =============================================================================
# REGISTRY LOADER
# =============================================================================

class PantheonRegistry:
    """
    Formal Pantheon Registry Loader.
    
    Loads god contracts from YAML, validates them, and provides fast lookup.
    Thread-safe with caching for performance.
    
    Example:
        registry = PantheonRegistry.load()
        apollo = registry.get_god("Apollo")
        synthesis_gods = registry.find_gods_by_domain("synthesis")
    """
    
    def __init__(self, data: PantheonRegistryData):
        """Initialize with loaded data."""
        self._data = data
        self._by_domain_cache: Optional[Dict[str, List[GodContract]]] = None
        self._by_tier_cache: Optional[Dict[GodTier, List[GodContract]]] = None
        
    @classmethod
    def load(cls, registry_path: Optional[Path] = None) -> "PantheonRegistry":
        """
        Load pantheon registry from YAML file.
        
        Args:
            registry_path: Path to registry.yaml (default: pantheon/registry.yaml)
            
        Returns:
            Loaded and validated PantheonRegistry instance
            
        Raises:
            FileNotFoundError: If registry file not found
            ValueError: If registry validation fails
        """
        if registry_path is None:
            # Default to pantheon/registry.yaml relative to project root
            project_root = Path(__file__).parent.parent
            registry_path = project_root / "pantheon" / "registry.yaml"
            
        if not registry_path.exists():
            raise FileNotFoundError(f"Registry not found: {registry_path}")
            
        logger.info(f"Loading pantheon registry from {registry_path}")
        
        # Load YAML
        with open(registry_path, 'r') as f:
            raw_data = yaml.safe_load(f)
            
        # Parse and validate
        data = cls._parse_registry(raw_data)
        cls._validate_registry(data)
        
        logger.info(f"Loaded {len(data.gods)} gods from registry v{data.metadata.version}")
        return cls(data)
    
    @staticmethod
    def _parse_registry(raw_data: Dict) -> PantheonRegistryData:
        """Parse raw YAML data into structured types."""
        
        # Parse gods
        gods = {}
        for name, god_data in raw_data['gods'].items():
            # Parse rest policy
            rest_policy_data = god_data['rest_policy']
            rest_policy = RestPolicy(
                type=RestPolicyType(rest_policy_data['type']),
                reason=rest_policy_data['reason'],
                partner=rest_policy_data.get('partner'),
                duty_cycle=rest_policy_data.get('duty_cycle'),
                rest_duration=rest_policy_data.get('rest_duration'),
                active_season=rest_policy_data.get('active_season'),
                rest_season=rest_policy_data.get('rest_season'),
            )
            
            # Parse spawn constraints
            spawn_data = god_data['spawn_constraints']
            spawn_constraints = SpawnConstraints(
                max_instances=spawn_data['max_instances'],
                when_allowed=spawn_data['when_allowed'],
                rationale=spawn_data['rationale'],
            )
            
            # Parse E8 alignment
            e8_data = god_data['e8_alignment']
            e8_alignment = E8Alignment(
                simple_root=e8_data.get('simple_root'),
                layer=e8_data['layer'],
            )
            
            # Create god contract
            gods[name] = GodContract(
                name=name,
                tier=GodTier(god_data['tier']),
                domain=god_data['domain'],
                description=god_data['description'],
                octant=god_data.get('octant'),
                epithets=god_data.get('epithets', []),
                coupling_affinity=god_data.get('coupling_affinity', []),
                rest_policy=rest_policy,
                spawn_constraints=spawn_constraints,
                promotion_from=god_data.get('promotion_from'),
                e8_alignment=e8_alignment,
            )
        
        # Parse chaos kernel rules
        chaos_rules = ChaosKernelRules(
            naming_pattern=raw_data['chaos_kernel_rules']['naming_pattern'],
            description=raw_data['chaos_kernel_rules']['description'],
            lifecycle=raw_data['chaos_kernel_rules']['lifecycle'],
            pruning=raw_data['chaos_kernel_rules']['pruning'],
            spawning_limits=raw_data['chaos_kernel_rules']['spawning_limits'],
            genetic_lineage=raw_data['chaos_kernel_rules']['genetic_lineage'],
        )
        
        # Parse metadata
        metadata = RegistryMetadata(
            version=raw_data['metadata']['version'],
            status=raw_data['metadata']['status'],
            created=raw_data['metadata']['created'],
            authority=raw_data['metadata']['authority'],
            validation_required=raw_data['metadata']['validation_required'],
        )
        
        return PantheonRegistryData(
            gods=gods,
            chaos_kernel_rules=chaos_rules,
            metadata=metadata,
            schema_version=raw_data['schema_version'],
            compatibility=raw_data['compatibility'],
            validation_rules=raw_data['validation_rules'],
        )
    
    @staticmethod
    def _validate_registry(data: PantheonRegistryData) -> None:
        """
        Validate registry data against business rules.
        
        Raises:
            ValueError: If validation fails
        """
        errors = []
        
        # 1. All gods must have unique names (guaranteed by dict)
        
        # 2. All gods must have max_instances: 1
        for name, god in data.gods.items():
            if god.spawn_constraints.max_instances != 1:
                errors.append(
                    f"God {name} must have max_instances: 1 (gods are singular)"
                )
        
        # 3. Essential tier gods must never sleep or have minimal rotating rest only
        # Note: minimal_rotating is allowed for essential communication gods like Hermes
        # that need brief rest periods while maintaining near-constant availability
        for name, god in data.gods.items():
            if god.tier == GodTier.ESSENTIAL:
                # Allow minimal_rotating for essential gods that need brief rest
                if god.rest_policy.type not in [RestPolicyType.NEVER, RestPolicyType.MINIMAL_ROTATING]:
                    errors.append(
                        f"Essential god {name} must have rest_policy.type: never or minimal_rotating "
                        f"(found: {god.rest_policy.type.value})"
                    )
        
        # 4. Check coupling affinity references valid gods
        for name, god in data.gods.items():
            for affinity in god.coupling_affinity:
                if affinity not in data.gods:
                    errors.append(
                        f"God {name} references non-existent coupling affinity: {affinity}"
                    )
        
        # 5. Check rest policy partners exist
        for name, god in data.gods.items():
            if god.rest_policy.partner:
                if god.rest_policy.partner not in data.gods:
                    errors.append(
                        f"God {name} references non-existent rest partner: {god.rest_policy.partner}"
                    )
        
        # 6. Check total active limit <= E8 roots (240)
        total_limit = data.chaos_kernel_rules.spawning_limits['total_active_limit']
        if total_limit > 240:
            logger.warning(
                f"Total active limit ({total_limit}) exceeds E8 root system (240)"
            )
        
        if errors:
            raise ValueError(f"Registry validation failed:\n" + "\n".join(errors))
    
    # =========================================================================
    # LOOKUP METHODS
    # =========================================================================
    
    def get_god(self, name: str) -> Optional[GodContract]:
        """Get god contract by name."""
        return self._data.gods.get(name)
    
    def get_all_gods(self) -> Dict[str, GodContract]:
        """Get all god contracts."""
        return self._data.gods.copy()
    
    def get_gods_by_tier(self, tier: GodTier) -> List[GodContract]:
        """Get all gods in a specific tier."""
        if self._by_tier_cache is None:
            self._build_tier_cache()
        return self._by_tier_cache[tier]
    
    def find_gods_by_domain(self, domain: str) -> List[GodContract]:
        """Find all gods with a specific domain."""
        if self._by_domain_cache is None:
            self._build_domain_cache()
        return self._by_domain_cache.get(domain, [])
    
    def get_chaos_kernel_rules(self) -> ChaosKernelRules:
        """Get chaos kernel lifecycle rules."""
        return self._data.chaos_kernel_rules
    
    def is_god_name(self, name: str) -> bool:
        """Check if name is a registered god."""
        return name in self._data.gods
    
    def is_valid_chaos_kernel_name(self, name: str) -> bool:
        """Check if name follows chaos kernel naming pattern."""
        import re
        pattern = r'^chaos_[a-z_]+_\d+$'
        return bool(re.match(pattern, name))
    
    def parse_chaos_kernel_name(self, name: str) -> Optional[Tuple[str, int]]:
        """
        Parse chaos kernel name and extract domain and ID.
        
        Returns:
            Tuple of (domain, id) or None if invalid
        """
        import re
        match = re.match(r'^chaos_([a-z_]+)_(\d+)$', name)
        if match:
            return (match.group(1), int(match.group(2)))
        return None
    
    # =========================================================================
    # CACHE BUILDING
    # =========================================================================
    
    def _build_domain_cache(self) -> None:
        """Build domain lookup cache."""
        cache: Dict[str, List[GodContract]] = {}
        for god in self._data.gods.values():
            for domain in god.domain:
                if domain not in cache:
                    cache[domain] = []
                cache[domain].append(god)
        self._by_domain_cache = cache
    
    def _build_tier_cache(self) -> None:
        """Build tier lookup cache."""
        cache: Dict[GodTier, List[GodContract]] = {
            GodTier.ESSENTIAL: [],
            GodTier.SPECIALIZED: [],
        }
        for god in self._data.gods.values():
            cache[god.tier].append(god)
        self._by_tier_cache = cache
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_metadata(self) -> RegistryMetadata:
        """Get registry metadata."""
        return self._data.metadata
    
    def get_god_count(self) -> int:
        """Get total number of gods."""
        return len(self._data.gods)
    
    def get_essential_count(self) -> int:
        """Get count of essential tier gods."""
        return len(self.get_gods_by_tier(GodTier.ESSENTIAL))
    
    def get_specialized_count(self) -> int:
        """Get count of specialized tier gods."""
        return len(self.get_gods_by_tier(GodTier.SPECIALIZED))
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PantheonRegistry(gods={self.get_god_count()}, "
            f"essential={self.get_essential_count()}, "
            f"specialized={self.get_specialized_count()}, "
            f"version={self._data.metadata.version})"
        )


# =============================================================================
# GLOBAL REGISTRY INSTANCE (Singleton Pattern)
# =============================================================================

_global_registry: Optional[PantheonRegistry] = None


def get_registry() -> PantheonRegistry:
    """
    Get global pantheon registry instance (singleton).
    
    Loads registry on first call, then returns cached instance.
    Thread-safe via import lock.
    
    Returns:
        Global PantheonRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = PantheonRegistry.load()
    return _global_registry


def reload_registry() -> PantheonRegistry:
    """
    Force reload of global registry.
    
    Useful for testing or when registry file changes.
    
    Returns:
        Newly loaded PantheonRegistry instance
    """
    global _global_registry
    _global_registry = PantheonRegistry.load()
    return _global_registry


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_god(name: str) -> Optional[GodContract]:
    """Convenience: Get god contract by name."""
    return get_registry().get_god(name)


def find_gods_by_domain(domain: str) -> List[GodContract]:
    """Convenience: Find gods by domain."""
    return get_registry().find_gods_by_domain(domain)


def is_god_name(name: str) -> bool:
    """Convenience: Check if name is a god."""
    return get_registry().is_god_name(name)


def is_chaos_kernel(name: str) -> bool:
    """Convenience: Check if name is a chaos kernel."""
    return get_registry().is_valid_chaos_kernel_name(name)
