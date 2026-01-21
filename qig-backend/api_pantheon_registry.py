"""
Pantheon Registry API - Python Backend
======================================

Flask API endpoints for formal Pantheon Registry operations.
All functional logic lives here; TypeScript is a thin API wrapper.

Authority: E8 Protocol v4.0, WP5.1
Status: ACTIVE
Created: 2026-01-20
"""

import logging
from typing import Dict
from flask import Blueprint, jsonify, request

from pantheon_registry import (
    get_registry,
    get_god,
    find_gods_by_domain,
)
from kernel_spawner import (
    KernelSpawner,
    RoleSpec,
)
from pantheon_registry_helpers import god_to_dict, is_valid_chaos_kernel_name

logger = logging.getLogger(__name__)

# Create Blueprint
pantheon_registry_api = Blueprint('pantheon_registry', __name__, url_prefix='/api/pantheon')


# =============================================================================
# REGISTRY ENDPOINTS
# =============================================================================

@pantheon_registry_api.route('/registry', methods=['GET'])
def get_full_registry():
    """
    GET /api/pantheon/registry
    Get full pantheon registry.
    """
    try:
        registry = get_registry()
        
        # Convert to dict for JSON serialization
        gods_dict = {}
        for name, god in registry.get_all_gods().items():
            gods_dict[name] = {
                'name': god.name,
                'tier': god.tier.value,
                'domain': god.domain,
                'description': god.description,
                'octant': god.octant,
                'epithets': god.epithets,
                'coupling_affinity': god.coupling_affinity,
                'rest_policy': {
                    'type': god.rest_policy.type.value,
                    'reason': god.rest_policy.reason,
                    'partner': god.rest_policy.partner,
                    'duty_cycle': god.rest_policy.duty_cycle,
                    'rest_duration': god.rest_policy.rest_duration,
                    'active_season': god.rest_policy.active_season,
                    'rest_season': god.rest_policy.rest_season,
                },
                'spawn_constraints': {
                    'max_instances': god.spawn_constraints.max_instances,
                    'when_allowed': god.spawn_constraints.when_allowed,
                    'rationale': god.spawn_constraints.rationale,
                },
                'promotion_from': god.promotion_from,
                'e8_alignment': {
                    'simple_root': god.e8_alignment.simple_root,
                    'layer': god.e8_alignment.layer,
                },
            }
        
        metadata = registry.get_metadata()
        chaos_rules = registry.get_chaos_kernel_rules()
        
        return jsonify({
            'success': True,
            'data': {
                'gods': gods_dict,
                'chaos_kernel_rules': {
                    'naming_pattern': chaos_rules.naming_pattern,
                    'description': chaos_rules.description,
                    'lifecycle': chaos_rules.lifecycle,
                    'pruning': chaos_rules.pruning,
                    'spawning_limits': chaos_rules.spawning_limits,
                    'genetic_lineage': chaos_rules.genetic_lineage,
                },
                'metadata': {
                    'version': metadata.version,
                    'status': metadata.status,
                    'created': metadata.created,
                    'authority': metadata.authority,
                    'validation_required': metadata.validation_required,
                },
                'schema_version': '1.0',
            }
        })
    except Exception as e:
        logger.error(f"Error getting registry: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@pantheon_registry_api.route('/registry/metadata', methods=['GET'])
def get_registry_metadata():
    """
    GET /api/pantheon/registry/metadata
    Get registry metadata.
    """
    try:
        registry = get_registry()
        metadata = registry.get_metadata()
        
        return jsonify({
            'success': True,
            'data': {
                'version': metadata.version,
                'status': metadata.status,
                'created': metadata.created,
                'authority': metadata.authority,
                'validation_required': metadata.validation_required,
            }
        })
    except Exception as e:
        logger.error(f"Error getting metadata: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@pantheon_registry_api.route('/registry/gods', methods=['GET'])
def get_all_gods():
    """
    GET /api/pantheon/registry/gods
    Get all god contracts.
    """
    try:
        registry = get_registry()
        gods_dict = {}
        
        for name, god in registry.get_all_gods().items():
            gods_dict[name] = god_to_dict(god)
        
        return jsonify({
            'success': True,
            'data': gods_dict,
            'count': len(gods_dict)
        })
    except Exception as e:
        logger.error(f"Error getting all gods: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@pantheon_registry_api.route('/registry/gods/<name>', methods=['GET'])
def get_god_by_name(name: str):
    """
    GET /api/pantheon/registry/gods/<name>
    Get specific god contract.
    """
    try:
        god = get_god(name)
        
        if not god:
            return jsonify({
                'success': False,
                'error': f'God {name} not found'
            }), 404
        
        return jsonify({
            'success': True,
            'data': god_to_dict(god)
        })
    except Exception as e:
        logger.error(f"Error getting god {name}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@pantheon_registry_api.route('/registry/gods/by-tier/<tier>', methods=['GET'])
def get_gods_by_tier(tier: str):
    """
    GET /api/pantheon/registry/gods/by-tier/<tier>
    Get gods by tier (essential or specialized).
    """
    try:
        if tier not in ['essential', 'specialized']:
            return jsonify({
                'success': False,
                'error': 'Invalid tier - must be "essential" or "specialized"'
            }), 400
        
        registry = get_registry()
        from pantheon_registry import GodTier
        tier_enum = GodTier.ESSENTIAL if tier == 'essential' else GodTier.SPECIALIZED
        
        gods = registry.get_gods_by_tier(tier_enum)
        gods_list = [god_to_dict(god) for god in gods]
        
        return jsonify({
            'success': True,
            'data': gods_list,
            'count': len(gods_list)
        })
    except Exception as e:
        logger.error(f"Error getting gods by tier {tier}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@pantheon_registry_api.route('/registry/gods/by-domain/<domain>', methods=['GET'])
def get_gods_by_domain(domain: str):
    """
    GET /api/pantheon/registry/gods/by-domain/<domain>
    Find gods by domain.
    """
    try:
        gods = find_gods_by_domain(domain)
        gods_list = [{'name': god.name, 'contract': god_to_dict(god)} for god in gods]
        
        return jsonify({
            'success': True,
            'data': gods_list,
            'count': len(gods_list)
        })
    except Exception as e:
        logger.error(f"Error finding gods by domain {domain}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@pantheon_registry_api.route('/registry/chaos-rules', methods=['GET'])
def get_chaos_rules():
    """
    GET /api/pantheon/registry/chaos-rules
    Get chaos kernel lifecycle rules.
    """
    try:
        registry = get_registry()
        rules = registry.get_chaos_kernel_rules()
        
        return jsonify({
            'success': True,
            'data': {
                'naming_pattern': rules.naming_pattern,
                'description': rules.description,
                'lifecycle': rules.lifecycle,
                'pruning': rules.pruning,
                'spawning_limits': rules.spawning_limits,
                'genetic_lineage': rules.genetic_lineage,
            }
        })
    except Exception as e:
        logger.error(f"Error getting chaos rules: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# =============================================================================
# KERNEL SPAWNER ENDPOINTS
# =============================================================================

@pantheon_registry_api.route('/spawner/select', methods=['POST'])
def select_kernel():
    """
    POST /api/pantheon/spawner/select
    Select god or chaos kernel for a role.
    
    Body: RoleSpec
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'Invalid request - JSON body required'
            }), 400
        
        # Validate role spec
        domains = data.get('domain', [])
        capabilities = data.get('required_capabilities', [])
        
        if not domains or not isinstance(domains, list):
            return jsonify({
                'success': False,
                'error': 'Invalid role spec - domain array required'
            }), 400
        
        if not capabilities or not isinstance(capabilities, list):
            return jsonify({
                'success': False,
                'error': 'Invalid role spec - required_capabilities array required'
            }), 400
        
        # Create role spec
        role = RoleSpec(
            domains=domains,
            required_capabilities=capabilities,
            preferred_god=data.get('preferred_god'),
            allow_chaos_spawn=data.get('allow_chaos_spawn', True),
            urgency=data.get('urgency', 'normal')
        )
        
        # Select kernel
        spawner = KernelSpawner()
        selection = spawner.select_god(role)
        
        return jsonify({
            'success': True,
            'data': {
                'selected_type': selection.selected_type,
                'god_name': selection.god_name,
                'epithet': selection.epithet,
                'chaos_name': selection.chaos_name,
                'rationale': selection.rationale,
                'spawn_approved': selection.spawn_approved,
                'requires_pantheon_vote': selection.requires_pantheon_vote,
            }
        })
    except Exception as e:
        logger.error(f"Error selecting kernel: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@pantheon_registry_api.route('/spawner/validate', methods=['POST'])
def validate_spawn():
    """
    POST /api/pantheon/spawner/validate
    Validate spawn request.
    
    Body: { name: string }
    """
    try:
        data = request.get_json()
        
        if not data or 'name' not in data:
            return jsonify({
                'success': False,
                'error': 'Invalid request - name required'
            }), 400
        
        name = data['name']
        spawner = KernelSpawner()
        valid, reason = spawner.validate_spawn_request(name)
        
        return jsonify({
            'success': True,
            'data': {
                'valid': valid,
                'reason': reason
            }
        })
    except Exception as e:
        logger.error(f"Error validating spawn: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@pantheon_registry_api.route('/spawner/chaos/parse/<name>', methods=['GET'])
def parse_chaos_name(name: str):
    """
    GET /api/pantheon/spawner/chaos/parse/<name>
    Parse chaos kernel name.
    """
    try:
        registry = get_registry()
        parsed = registry.parse_chaos_kernel_name(name)
        
        if not parsed:
            return jsonify({
                'success': False,
                'error': f'Invalid chaos kernel name: {name}'
            }), 400
        
        domain, id_num = parsed
        return jsonify({
            'success': True,
            'data': {
                'id': name,
                'name': name,
                'domain': domain,
                'sequential_id': id_num
            }
        })
    except Exception as e:
        logger.error(f"Error parsing chaos name {name}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@pantheon_registry_api.route('/spawner/status', methods=['GET'])
def get_spawner_status():
    """
    GET /api/pantheon/spawner/status
    Get spawner status (active counts, limits).
    """
    try:
        registry = get_registry()
        spawner = KernelSpawner()
        rules = registry.get_chaos_kernel_rules()
        
        return jsonify({
            'success': True,
            'data': {
                'total_chaos_spawned': spawner.get_total_chaos_count(),
                'active_chaos_count': spawner.get_active_count('_total_chaos'),  # Mock for now
                'limits': rules.spawning_limits,
            }
        })
    except Exception as e:
        logger.error(f"Error getting spawner status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@pantheon_registry_api.route('/spawner/validate/chaos/<name>', methods=['GET'])
def validate_chaos_kernel_name(name: str):
    """
    GET /api/pantheon/spawner/validate/chaos/<name>
    Validate chaos kernel name format.
    """
    try:
        is_valid = is_valid_chaos_kernel_name(name)
        return jsonify({
            'success': True,
            'data': {
                'valid': is_valid,
                'name': name
            }
        })
    except Exception as e:
        logger.error(f"Error validating chaos kernel name {name}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# =============================================================================
# HEALTH CHECK
# =============================================================================

@pantheon_registry_api.route('/health', methods=['GET'])
def health_check():
    """
    GET /api/pantheon/health
    Health check for registry service.
    """
    try:
        registry = get_registry()
        metadata = registry.get_metadata()
        god_count = registry.get_god_count()
        
        return jsonify({
            'success': True,
            'data': {
                'status': 'healthy',
                'registry_version': metadata.version,
                'registry_status': metadata.status,
                'god_count': god_count,
                'loaded_at': metadata.created,
            }
        })
    except Exception as e:
        logger.error(f"Registry health check failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'data': {
                'status': 'unhealthy'
            }
        }), 500


# =============================================================================
# REGISTRATION FUNCTION
# =============================================================================

def register_pantheon_registry_routes(app):
    """
    Register pantheon registry routes with Flask app.
    
    Args:
        app: Flask application instance
    """
    app.register_blueprint(pantheon_registry_api)
    logger.info("[PantheonRegistry] API routes registered at /api/pantheon/*")
