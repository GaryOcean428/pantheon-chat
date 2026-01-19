#!/usr/bin/env python3
"""
Cross-Mythology God Mapping - Python Implementation
====================================================

Loads and provides lookup functions for cross-mythology god name mapping.
Maps external mythology names (Egyptian, Norse, Hindu, etc.) to Greek canonical archetypes.

Authority: E8 Protocol v4.0, WP5.5
Status: ACTIVE
Created: 2026-01-19

Philosophy:
- Greek names are CANONICAL in codebase
- External names are ALIASES (metadata only)
- Simple lookup table (NO runtime complexity)
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from functools import lru_cache

import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

@dataclass
class GodMapping:
    """Mapping from external god name to Greek archetype."""
    external_name: str
    mythology: str  # egyptian, norse, hindu, sumerian, mesoamerican
    greek_archetype: str
    domain: List[str]
    notes: str
    alternative_mapping: Optional[str] = None


@dataclass
class MythologyInfo:
    """Complete information about a god across mythologies."""
    greek_name: str
    external_equivalents: Dict[str, List[str]]  # mythology -> [god_names]
    all_domains: Set[str]
    mapping_count: int


# =============================================================================
# YAML LOADER
# =============================================================================

class CrossMythologyRegistry:
    """
    Registry for cross-mythology god name mappings.
    
    Provides fast lookup and reverse lookup for god names across mythologies.
    """
    
    def __init__(self, registry_path: Optional[Path] = None):
        """
        Initialize the cross-mythology registry.
        
        Args:
            registry_path: Path to myth_mappings.yaml (defaults to pantheon/myth_mappings.yaml)
        """
        if registry_path is None:
            # Default to pantheon/myth_mappings.yaml relative to this file
            registry_path = Path(__file__).parent / "myth_mappings.yaml"
        
        self.registry_path = registry_path
        self._raw_data: Optional[Dict] = None
        self._mappings: Dict[str, GodMapping] = {}
        self._greek_to_external: Dict[str, List[GodMapping]] = {}
        self._domain_index: Dict[str, List[str]] = {}  # domain -> [greek_names]
        
        self._load_registry()
    
    def _load_registry(self):
        """Load and index the registry from YAML."""
        try:
            with open(self.registry_path, 'r', encoding='utf-8') as f:
                self._raw_data = yaml.safe_load(f)
            
            if not self._raw_data:
                raise ValueError(f"Empty registry file: {self.registry_path}")
            
            # Index all mythologies
            for mythology in ['egyptian', 'norse', 'hindu', 'sumerian', 'mesoamerican']:
                if mythology not in self._raw_data:
                    logger.warning(f"Mythology '{mythology}' not found in registry")
                    continue
                
                myth_data = self._raw_data[mythology]
                for external_name, god_data in myth_data.items():
                    mapping = GodMapping(
                        external_name=external_name,
                        mythology=mythology,
                        greek_archetype=god_data['greek_archetype'],
                        domain=god_data.get('domain', []),
                        notes=god_data.get('notes', ''),
                        alternative_mapping=god_data.get('alternative_mapping'),
                    )
                    
                    # Index by external name (case-insensitive)
                    key = external_name.lower()
                    self._mappings[key] = mapping
                    
                    # Reverse index: greek -> external
                    greek_name = mapping.greek_archetype
                    if greek_name not in self._greek_to_external:
                        self._greek_to_external[greek_name] = []
                    self._greek_to_external[greek_name].append(mapping)
                    
                    # Domain index
                    for domain in mapping.domain:
                        domain_key = domain.lower()
                        if domain_key not in self._domain_index:
                            self._domain_index[domain_key] = []
                        if greek_name not in self._domain_index[domain_key]:
                            self._domain_index[domain_key].append(greek_name)
            
            logger.info(
                f"Loaded {len(self._mappings)} cross-mythology mappings "
                f"covering {len(self._greek_to_external)} Greek archetypes"
            )
            
        except FileNotFoundError:
            logger.error(f"Registry file not found: {self.registry_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML registry: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading registry: {e}")
            raise
    
    def resolve_god_name(self, external_name: str) -> str:
        """
        Resolve an external god name to its Greek archetype.
        
        Args:
            external_name: External god name (e.g., "Odin", "Thoth", "Shiva")
        
        Returns:
            Greek archetype name (e.g., "Zeus", "Hermes", "Dionysus")
        
        Raises:
            KeyError: If external name not found in registry
        
        Examples:
            >>> registry.resolve_god_name("Odin")
            'Zeus'
            >>> registry.resolve_god_name("Thoth")
            'Hermes'
        """
        key = external_name.lower()
        if key not in self._mappings:
            raise KeyError(
                f"God '{external_name}' not found in cross-mythology registry. "
                f"Available: {sorted(self._mappings.keys())}"
            )
        
        return self._mappings[key].greek_archetype
    
    def find_similar_gods(self, domain: List[str]) -> List[Tuple[str, int]]:
        """
        Find Greek gods by domain keywords.
        
        Args:
            domain: List of domain keywords (e.g., ["wisdom", "war", "strategy"])
        
        Returns:
            List of (greek_name, match_count) sorted by match count descending
        
        Examples:
            >>> registry.find_similar_gods(["wisdom", "strategy"])
            [('Athena', 2), ('Zeus', 1), ...]
        """
        matches: Dict[str, int] = {}
        
        for domain_word in domain:
            domain_key = domain_word.lower()
            if domain_key in self._domain_index:
                for greek_name in self._domain_index[domain_key]:
                    matches[greek_name] = matches.get(greek_name, 0) + 1
        
        # Sort by match count descending
        sorted_matches = sorted(matches.items(), key=lambda x: x[1], reverse=True)
        return sorted_matches
    
    def get_mythology_info(self, god_name: str) -> Dict:
        """
        Get complete information about a god (Greek or external).
        
        Args:
            god_name: God name (Greek or external)
        
        Returns:
            Dictionary with:
            - is_greek: bool
            - canonical_name: str (Greek archetype)
            - mythology: str (if external) or None
            - domain: List[str]
            - notes: str
            - alternative_mapping: Optional[str]
            - external_equivalents: Dict[str, List[str]] (if Greek)
        
        Examples:
            >>> registry.get_mythology_info("Thoth")
            {'is_greek': False, 'canonical_name': 'Hermes', 'mythology': 'egyptian', ...}
            >>> registry.get_mythology_info("Zeus")
            {'is_greek': True, 'canonical_name': 'Zeus', 'external_equivalents': {...}, ...}
        """
        # Check if it's an external name
        key = god_name.lower()
        if key in self._mappings:
            mapping = self._mappings[key]
            return {
                'is_greek': False,
                'canonical_name': mapping.greek_archetype,
                'mythology': mapping.mythology,
                'external_name': mapping.external_name,
                'domain': mapping.domain,
                'notes': mapping.notes,
                'alternative_mapping': mapping.alternative_mapping,
            }
        
        # Check if it's a Greek name
        if god_name in self._greek_to_external:
            equivalents: Dict[str, List[str]] = {}
            all_domains: Set[str] = set()
            
            for mapping in self._greek_to_external[god_name]:
                mythology = mapping.mythology
                if mythology not in equivalents:
                    equivalents[mythology] = []
                equivalents[mythology].append(mapping.external_name)
                all_domains.update(mapping.domain)
            
            return {
                'is_greek': True,
                'canonical_name': god_name,
                'external_equivalents': equivalents,
                'all_domains': sorted(all_domains),
                'mapping_count': len(self._greek_to_external[god_name]),
            }
        
        raise KeyError(
            f"God '{god_name}' not found in registry (neither Greek nor external)"
        )
    
    def get_external_equivalents(self, greek_name: str) -> Dict[str, List[str]]:
        """
        Get all external mythology equivalents for a Greek god.
        
        Args:
            greek_name: Greek god name (e.g., "Zeus", "Hermes")
        
        Returns:
            Dictionary mapping mythology -> list of external god names
            Example: {'norse': ['Odin'], 'hindu': ['Vishnu'], ...}
        
        Examples:
            >>> registry.get_external_equivalents("Zeus")
            {'egyptian': ['Ra'], 'norse': ['Odin', 'Thor'], 'hindu': ['Vishnu'], ...}
        """
        if greek_name not in self._greek_to_external:
            return {}
        
        equivalents: Dict[str, List[str]] = {}
        for mapping in self._greek_to_external[greek_name]:
            mythology = mapping.mythology
            if mythology not in equivalents:
                equivalents[mythology] = []
            equivalents[mythology].append(mapping.external_name)
        
        return equivalents
    
    def list_all_greek_archetypes(self) -> List[str]:
        """Get list of all Greek archetypes that have external mappings."""
        return sorted(self._greek_to_external.keys())
    
    def list_all_external_names(self, mythology: Optional[str] = None) -> List[str]:
        """
        Get list of all external god names.
        
        Args:
            mythology: Optional filter by mythology (egyptian, norse, etc.)
        
        Returns:
            Sorted list of external god names
        """
        if mythology:
            return sorted([
                m.external_name for m in self._mappings.values()
                if m.mythology == mythology
            ])
        return sorted([m.external_name for m in self._mappings.values()])
    
    def get_number_symbolism(self) -> Dict:
        """
        Get number symbolism information (informational only).
        
        Returns:
            Dictionary of number symbolism data from registry
        """
        return self._raw_data.get('number_symbolism', {})
    
    def get_metadata(self) -> Dict:
        """Get registry metadata."""
        return self._raw_data.get('metadata', {})
    
    def get_philosophy(self) -> Dict:
        """Get mapping philosophy documentation."""
        return self._raw_data.get('philosophy', {})


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_default_registry: Optional[CrossMythologyRegistry] = None


@lru_cache(maxsize=1)
def get_cross_mythology_registry() -> CrossMythologyRegistry:
    """
    Get or create the default cross-mythology registry singleton.
    
    Returns:
        CrossMythologyRegistry instance
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = CrossMythologyRegistry()
    return _default_registry


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def resolve_god_name(external_name: str) -> str:
    """
    Convenience function: Resolve external god name to Greek archetype.
    
    Args:
        external_name: External god name
    
    Returns:
        Greek archetype name
    """
    registry = get_cross_mythology_registry()
    return registry.resolve_god_name(external_name)


def find_similar_gods(domain: List[str]) -> List[Tuple[str, int]]:
    """
    Convenience function: Find gods by domain keywords.
    
    Args:
        domain: List of domain keywords
    
    Returns:
        List of (greek_name, match_count) tuples
    """
    registry = get_cross_mythology_registry()
    return registry.find_similar_gods(domain)


def get_mythology_info(god_name: str) -> Dict:
    """
    Convenience function: Get complete mythology info for a god.
    
    Args:
        god_name: God name (Greek or external)
    
    Returns:
        Dictionary with god information
    """
    registry = get_cross_mythology_registry()
    return registry.get_mythology_info(god_name)


def get_external_equivalents(greek_name: str) -> Dict[str, List[str]]:
    """
    Convenience function: Get external equivalents for a Greek god.
    
    Args:
        greek_name: Greek god name
    
    Returns:
        Dictionary mapping mythology -> external god names
    """
    registry = get_cross_mythology_registry()
    return registry.get_external_equivalents(greek_name)


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    # Example usage
    registry = get_cross_mythology_registry()
    
    print("=== Cross-Mythology God Mapping Examples ===\n")
    
    # Resolve external names
    print("1. Resolving external names to Greek archetypes:")
    for external in ["Odin", "Thoth", "Shiva", "Quetzalcoatl"]:
        greek = registry.resolve_god_name(external)
        print(f"   {external} → {greek}")
    
    print("\n2. Finding gods by domain:")
    matches = registry.find_similar_gods(["wisdom", "strategy", "war"])
    for greek_name, count in matches[:5]:
        print(f"   {greek_name}: {count} domain matches")
    
    print("\n3. Getting mythology info:")
    info = registry.get_mythology_info("Thoth")
    print(f"   Thoth → {info['canonical_name']}")
    print(f"   Mythology: {info['mythology']}")
    print(f"   Domain: {', '.join(info['domain'][:3])}...")
    
    print("\n4. Getting external equivalents:")
    equiv = registry.get_external_equivalents("Zeus")
    print(f"   Zeus equivalents:")
    for myth, names in equiv.items():
        print(f"   - {myth}: {', '.join(names)}")
    
    print("\n5. List all Greek archetypes with mappings:")
    archetypes = registry.list_all_greek_archetypes()
    print(f"   {len(archetypes)} Greek archetypes: {', '.join(archetypes[:10])}...")
    
    print("\n6. Philosophy:")
    philosophy = registry.get_philosophy()
    print(f"   Canonical naming: {philosophy.get('canonical_naming')}")
    print(f"   Purpose: {registry.get_metadata().get('purpose')}")
