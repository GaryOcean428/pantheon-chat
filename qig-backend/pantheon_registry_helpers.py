"""
Pantheon Registry Helper Functions
===================================

Helper functions extracted from api_pantheon_registry.py to maintain
the 500-line hard limit per AGENTS.md guidelines.

Authority: E8 Protocol v4.0, WP5.1
Status: ACTIVE
Created: 2026-01-20
"""

from typing import Dict, Any
import re


def god_to_dict(god: Any) -> Dict[str, Any]:
    """
    Convert GodContract to dictionary for JSON serialization.
    
    Args:
        god: GodContract instance
        
    Returns:
        Dictionary representation of god contract
    """
    return {
        "name": god.name,
        "domain": god.domain,
        "tier": god.tier,
        "epithets": god.epithets,
        "octant": god.octant,
        "coupling_affinity": god.coupling_affinity,
        "spawn_constraints": {
            "max_instances": god.spawn_constraints.max_instances,
            "when_allowed": god.spawn_constraints.when_allowed,
        },
        "rest_policy": {
            "type": god.rest_policy.type,
            "partner": god.rest_policy.partner,
            "duty_cycle": god.rest_policy.duty_cycle,
        },
        "promotion_from": god.promotion_from,
    }


def is_valid_chaos_kernel_name(name: str) -> bool:
    """
    Validate chaos kernel name format.
    
    Pattern: chaos_{domain}_{sequential_id}
    Example: chaos_synthesis_001
    
    Args:
        name: Kernel name to validate
        
    Returns:
        True if valid chaos kernel name format
    """
    return bool(re.match(r'^chaos_[a-z_]+_\d+$', name))
