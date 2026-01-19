"""
Kernel Lifecycle Flask API Routes
=================================

Flask endpoints for kernel lifecycle operations.

Authority: E8 Protocol v4.0, WP5.3
Status: ACTIVE
Created: 2026-01-18
"""

import logging
from flask import Blueprint, jsonify, request
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Create blueprint
lifecycle_bp = Blueprint('lifecycle', __name__, url_prefix='/lifecycle')


def register_lifecycle_routes(app):
    """
    Register lifecycle routes with Flask app.
    
    Endpoints:
    - POST /lifecycle/spawn - Spawn new kernel
    - POST /lifecycle/split - Split kernel into specialists
    - POST /lifecycle/merge - Merge redundant kernels
    - POST /lifecycle/prune - Prune kernel to shadow pantheon
    - POST /lifecycle/resurrect - Resurrect from shadow pantheon
    - POST /lifecycle/promote - Promote chaos kernel to god
    - GET /lifecycle/kernels - List all active kernels
    - GET /lifecycle/shadows - List shadow pantheon
    - GET /lifecycle/events - Get lifecycle event history
    - GET /lifecycle/stats - Get lifecycle statistics
    - POST /lifecycle/evaluate - Evaluate policies for all kernels
    - POST /lifecycle/execute-action - Execute a triggered action
    - GET /lifecycle/policies - List all policies
    - POST /lifecycle/policies/enable - Enable a policy
    - POST /lifecycle/policies/disable - Disable a policy
    """
    
    app.register_blueprint(lifecycle_bp)
    logger.info("[LifecycleAPI] Registered lifecycle routes at /lifecycle/*")


# =============================================================================
# LIFECYCLE OPERATIONS
# =============================================================================

@lifecycle_bp.route('/spawn', methods=['POST'])
def spawn_kernel():
    """
    Spawn a new kernel.
    
    Request body:
    {
        "domains": ["synthesis", "foresight"],
        "required_capabilities": ["prediction"],
        "preferred_god": "Apollo",  # optional
        "allow_chaos_spawn": true,  # optional
        "mentor": "kernel_abc123",  # optional, for chaos kernels
        "initial_basin": [0.015625, ...]  # optional, 64D array
    }
    
    Response:
    {
        "success": true,
        "kernel": {...},
        "message": "Spawned god kernel: Apollo Pythios"
    }
    """
    try:
        from kernel_lifecycle import get_lifecycle_manager
        from kernel_spawner import RoleSpec
        import numpy as np
        
        data = request.json or {}
        
        # Create role specification
        role = RoleSpec(
            domains=data.get('domains', []),
            required_capabilities=data.get('required_capabilities', []),
            preferred_god=data.get('preferred_god'),
            allow_chaos_spawn=data.get('allow_chaos_spawn', True),
        )
        
        # Get optional parameters
        mentor = data.get('mentor')
        initial_basin = data.get('initial_basin')
        if initial_basin:
            initial_basin = np.array(initial_basin)
        
        # Spawn kernel
        manager = get_lifecycle_manager()
        kernel = manager.spawn(role, mentor=mentor, initial_basin=initial_basin)
        
        return jsonify({
            'success': True,
            'kernel': kernel.to_dict(),
            'message': f"Spawned {kernel.kernel_type} kernel: {kernel.name}",
        })
    
    except Exception as e:
        logger.error(f"[LifecycleAPI] Spawn error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
        }), 400


@lifecycle_bp.route('/split', methods=['POST'])
def split_kernel():
    """
    Split a kernel into specialized sub-kernels.
    
    Request body:
    {
        "kernel_id": "kernel_abc123",
        "split_criterion": "domain"  # or "skill", "random"
    }
    
    Response:
    {
        "success": true,
        "kernel1": {...},
        "kernel2": {...},
        "message": "Split kernel into 2 specialists"
    }
    """
    try:
        from kernel_lifecycle import get_lifecycle_manager
        
        data = request.json or {}
        kernel_id = data.get('kernel_id')
        split_criterion = data.get('split_criterion', 'domain')
        
        if not kernel_id:
            return jsonify({
                'success': False,
                'error': 'kernel_id required',
            }), 400
        
        manager = get_lifecycle_manager()
        kernel = manager.get_kernel(kernel_id)
        
        if not kernel:
            return jsonify({
                'success': False,
                'error': f'Kernel {kernel_id} not found',
            }), 404
        
        # Split kernel
        k1, k2 = manager.split(kernel, split_criterion)
        
        return jsonify({
            'success': True,
            'kernel1': k1.to_dict(),
            'kernel2': k2.to_dict(),
            'message': f"Split {kernel.name} into {k1.name} and {k2.name}",
        })
    
    except Exception as e:
        logger.error(f"[LifecycleAPI] Split error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
        }), 400


@lifecycle_bp.route('/merge', methods=['POST'])
def merge_kernels():
    """
    Merge two kernels into one.
    
    Request body:
    {
        "kernel1_id": "kernel_abc123",
        "kernel2_id": "kernel_def456",
        "merge_reason": "redundant_capabilities"
    }
    
    Response:
    {
        "success": true,
        "merged_kernel": {...},
        "message": "Merged 2 kernels into unified kernel"
    }
    """
    try:
        from kernel_lifecycle import get_lifecycle_manager
        
        data = request.json or {}
        kernel1_id = data.get('kernel1_id')
        kernel2_id = data.get('kernel2_id')
        merge_reason = data.get('merge_reason', 'manual_merge')
        
        if not kernel1_id or not kernel2_id:
            return jsonify({
                'success': False,
                'error': 'kernel1_id and kernel2_id required',
            }), 400
        
        manager = get_lifecycle_manager()
        kernel1 = manager.get_kernel(kernel1_id)
        kernel2 = manager.get_kernel(kernel2_id)
        
        if not kernel1 or not kernel2:
            return jsonify({
                'success': False,
                'error': 'One or both kernels not found',
            }), 404
        
        # Merge kernels
        merged = manager.merge(kernel1, kernel2, merge_reason)
        
        return jsonify({
            'success': True,
            'merged_kernel': merged.to_dict(),
            'message': f"Merged {kernel1.name} and {kernel2.name} into {merged.name}",
        })
    
    except Exception as e:
        logger.error(f"[LifecycleAPI] Merge error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
        }), 400


@lifecycle_bp.route('/prune', methods=['POST'])
def prune_kernel():
    """
    Prune a kernel to shadow pantheon.
    
    Request body:
    {
        "kernel_id": "kernel_abc123",
        "reason": "persistent_low_phi"
    }
    
    Response:
    {
        "success": true,
        "shadow_id": "shadow_xyz789",
        "message": "Pruned kernel to shadow pantheon"
    }
    """
    try:
        from kernel_lifecycle import get_lifecycle_manager
        
        data = request.json or {}
        kernel_id = data.get('kernel_id')
        reason = data.get('reason', 'manual_prune')
        
        if not kernel_id:
            return jsonify({
                'success': False,
                'error': 'kernel_id required',
            }), 400
        
        manager = get_lifecycle_manager()
        kernel = manager.get_kernel(kernel_id)
        
        if not kernel:
            return jsonify({
                'success': False,
                'error': f'Kernel {kernel_id} not found',
            }), 404
        
        # Prune kernel
        shadow = manager.prune(kernel, reason)
        
        return jsonify({
            'success': True,
            'shadow_id': shadow.shadow_id,
            'message': f"Pruned {kernel.name} to shadow pantheon",
        })
    
    except Exception as e:
        logger.error(f"[LifecycleAPI] Prune error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
        }), 400


@lifecycle_bp.route('/resurrect', methods=['POST'])
def resurrect_kernel():
    """
    Resurrect a kernel from shadow pantheon.
    
    Request body:
    {
        "shadow_id": "shadow_xyz789",
        "reason": "capability_needed",
        "mentor": "kernel_abc123"  # optional
    }
    
    Response:
    {
        "success": true,
        "kernel": {...},
        "message": "Resurrected kernel from shadow pantheon"
    }
    """
    try:
        from kernel_lifecycle import get_lifecycle_manager
        
        data = request.json or {}
        shadow_id = data.get('shadow_id')
        reason = data.get('reason', 'manual_resurrection')
        mentor = data.get('mentor')
        
        if not shadow_id:
            return jsonify({
                'success': False,
                'error': 'shadow_id required',
            }), 400
        
        manager = get_lifecycle_manager()
        shadow = manager.get_shadow(shadow_id)
        
        if not shadow:
            return jsonify({
                'success': False,
                'error': f'Shadow kernel {shadow_id} not found',
            }), 404
        
        # Resurrect kernel
        kernel = manager.resurrect(shadow, reason, mentor)
        
        return jsonify({
            'success': True,
            'kernel': kernel.to_dict(),
            'message': f"Resurrected {kernel.name} from shadow pantheon",
        })
    
    except Exception as e:
        logger.error(f"[LifecycleAPI] Resurrect error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
        }), 400


@lifecycle_bp.route('/promote', methods=['POST'])
def promote_kernel():
    """
    Promote a chaos kernel to god status.
    
    Request body:
    {
        "kernel_id": "kernel_abc123",
        "god_name": "Prometheus"
    }
    
    Response:
    {
        "success": true,
        "god_kernel": {...},
        "message": "Promoted chaos kernel to god status"
    }
    """
    try:
        from kernel_lifecycle import get_lifecycle_manager
        
        data = request.json or {}
        kernel_id = data.get('kernel_id')
        god_name = data.get('god_name')
        
        if not kernel_id or not god_name:
            return jsonify({
                'success': False,
                'error': 'kernel_id and god_name required',
            }), 400
        
        manager = get_lifecycle_manager()
        kernel = manager.get_kernel(kernel_id)
        
        if not kernel:
            return jsonify({
                'success': False,
                'error': f'Kernel {kernel_id} not found',
            }), 404
        
        # Promote kernel
        god_kernel = manager.promote(kernel, god_name)
        
        return jsonify({
            'success': True,
            'god_kernel': god_kernel.to_dict(),
            'message': f"Promoted {kernel.name} to god {god_name}",
        })
    
    except Exception as e:
        logger.error(f"[LifecycleAPI] Promote error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
        }), 400


# =============================================================================
# QUERY & STATS
# =============================================================================

@lifecycle_bp.route('/kernels', methods=['GET'])
def list_kernels():
    """
    List all active kernels.
    
    Response:
    {
        "success": true,
        "kernels": [...],
        "count": 5
    }
    """
    try:
        from kernel_lifecycle import get_lifecycle_manager
        
        manager = get_lifecycle_manager()
        kernels = manager.list_active_kernels()
        
        return jsonify({
            'success': True,
            'kernels': [k.to_dict() for k in kernels],
            'count': len(kernels),
        })
    
    except Exception as e:
        logger.error(f"[LifecycleAPI] List kernels error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
        }), 500


@lifecycle_bp.route('/shadows', methods=['GET'])
def list_shadows():
    """
    List all shadow pantheon kernels.
    
    Response:
    {
        "success": true,
        "shadows": [...],
        "count": 3
    }
    """
    try:
        from kernel_lifecycle import get_lifecycle_manager
        from dataclasses import asdict
        
        manager = get_lifecycle_manager()
        shadows = manager.list_shadow_kernels()
        
        # Convert to dict (handle numpy arrays)
        shadows_dict = []
        for shadow in shadows:
            shadow_dict = asdict(shadow)
            # Convert numpy array to list
            if hasattr(shadow.final_basin, 'tolist'):
                shadow_dict['final_basin'] = shadow.final_basin.tolist()
            # Convert datetime to ISO string
            shadow_dict['prune_timestamp'] = shadow.prune_timestamp.isoformat()
            if shadow.last_resurrection:
                shadow_dict['last_resurrection'] = shadow.last_resurrection.isoformat()
            shadows_dict.append(shadow_dict)
        
        return jsonify({
            'success': True,
            'shadows': shadows_dict,
            'count': len(shadows),
        })
    
    except Exception as e:
        logger.error(f"[LifecycleAPI] List shadows error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
        }), 500


@lifecycle_bp.route('/events', methods=['GET'])
def get_events():
    """
    Get lifecycle event history.
    
    Query params:
    - limit: Maximum number of events (default: 100)
    - event_type: Filter by event type
    
    Response:
    {
        "success": true,
        "events": [...],
        "count": 25
    }
    """
    try:
        from kernel_lifecycle import get_lifecycle_manager
        from dataclasses import asdict
        
        manager = get_lifecycle_manager()
        
        # Get query params
        limit = int(request.args.get('limit', 100))
        event_type = request.args.get('event_type')
        
        # Get events
        events = manager.event_log[-limit:]
        
        # Filter by type if specified
        if event_type:
            events = [e for e in events if e.event_type.value == event_type]
        
        # Convert to dict
        events_dict = []
        for event in events:
            event_dict = asdict(event)
            event_dict['event_type'] = event.event_type.value
            event_dict['timestamp'] = event.timestamp.isoformat()
            events_dict.append(event_dict)
        
        return jsonify({
            'success': True,
            'events': events_dict,
            'count': len(events_dict),
        })
    
    except Exception as e:
        logger.error(f"[LifecycleAPI] Get events error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
        }), 500


@lifecycle_bp.route('/stats', methods=['GET'])
def get_stats():
    """
    Get lifecycle statistics.
    
    Response:
    {
        "success": true,
        "stats": {
            "active_kernels": 5,
            "shadow_kernels": 3,
            "total_events": 25,
            "event_counts": {...},
            ...
        }
    }
    """
    try:
        from kernel_lifecycle import get_lifecycle_manager
        
        manager = get_lifecycle_manager()
        stats = manager.get_lifecycle_stats()
        
        return jsonify({
            'success': True,
            'stats': stats,
        })
    
    except Exception as e:
        logger.error(f"[LifecycleAPI] Get stats error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
        }), 500


# =============================================================================
# POLICY ENGINE
# =============================================================================

@lifecycle_bp.route('/evaluate', methods=['POST'])
def evaluate_policies():
    """
    Evaluate policies for all kernels.
    
    Request body (optional):
    {
        "context": {...}  # Additional context for evaluation
    }
    
    Response:
    {
        "success": true,
        "triggered_actions": [...],
        "count": 3
    }
    """
    try:
        from lifecycle_policy import get_policy_engine
        
        data = request.json or {}
        context = data.get('context')
        
        engine = get_policy_engine()
        actions = engine.evaluate_all_kernels(context)
        
        return jsonify({
            'success': True,
            'triggered_actions': actions,
            'count': len(actions),
        })
    
    except Exception as e:
        logger.error(f"[LifecycleAPI] Evaluate policies error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
        }), 500


@lifecycle_bp.route('/execute-action', methods=['POST'])
def execute_action():
    """
    Execute a triggered lifecycle action.
    
    Request body:
    {
        "action": {...}  # Action from evaluate_policies
    }
    
    Response:
    {
        "success": true,
        "message": "Action executed successfully"
    }
    """
    try:
        from lifecycle_policy import get_policy_engine
        
        data = request.json or {}
        action = data.get('action')
        
        if not action:
            return jsonify({
                'success': False,
                'error': 'action required',
            }), 400
        
        engine = get_policy_engine()
        success = engine.execute_action(action)
        
        if success:
            # policy_type is already a LifecycleEvent enum in the action dict
            policy_type_str = action['policy_type'].value if hasattr(action['policy_type'], 'value') else str(action['policy_type'])
            return jsonify({
                'success': True,
                'message': f"Executed {policy_type_str} action",
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Action execution failed',
            }), 500
    
    except Exception as e:
        logger.error(f"[LifecycleAPI] Execute action error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
        }), 500


@lifecycle_bp.route('/policies', methods=['GET'])
def list_policies():
    """
    List all lifecycle policies.
    
    Response:
    {
        "success": true,
        "policies": [...],
        "stats": {...}
    }
    """
    try:
        from lifecycle_policy import get_policy_engine
        
        engine = get_policy_engine()
        
        policies = []
        for policy in engine.policies:
            policies.append({
                'name': policy.policy_name,
                'type': policy.policy_type.value,
                'enabled': policy.enabled,
                'priority': policy.priority,
                'description': policy.description,
                'cooldown_cycles': policy.cooldown_cycles,
            })
        
        stats = engine.get_policy_stats()
        
        return jsonify({
            'success': True,
            'policies': policies,
            'stats': stats,
        })
    
    except Exception as e:
        logger.error(f"[LifecycleAPI] List policies error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
        }), 500


@lifecycle_bp.route('/policies/enable', methods=['POST'])
def enable_policy():
    """
    Enable a lifecycle policy.
    
    Request body:
    {
        "policy_name": "prune_low_phi_persistent"
    }
    
    Response:
    {
        "success": true,
        "message": "Policy enabled"
    }
    """
    try:
        from lifecycle_policy import get_policy_engine
        
        data = request.json or {}
        policy_name = data.get('policy_name')
        
        if not policy_name:
            return jsonify({
                'success': False,
                'error': 'policy_name required',
            }), 400
        
        engine = get_policy_engine()
        success = engine.enable_policy(policy_name)
        
        if success:
            return jsonify({
                'success': True,
                'message': f"Enabled policy: {policy_name}",
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Policy {policy_name} not found',
            }), 404
    
    except Exception as e:
        logger.error(f"[LifecycleAPI] Enable policy error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
        }), 500


@lifecycle_bp.route('/policies/disable', methods=['POST'])
def disable_policy():
    """
    Disable a lifecycle policy.
    
    Request body:
    {
        "policy_name": "prune_low_phi_persistent"
    }
    
    Response:
    {
        "success": true,
        "message": "Policy disabled"
    }
    """
    try:
        from lifecycle_policy import get_policy_engine
        
        data = request.json or {}
        policy_name = data.get('policy_name')
        
        if not policy_name:
            return jsonify({
                'success': False,
                'error': 'policy_name required',
            }), 400
        
        engine = get_policy_engine()
        success = engine.disable_policy(policy_name)
        
        if success:
            return jsonify({
                'success': True,
                'message': f"Disabled policy: {policy_name}",
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Policy {policy_name} not found',
            }), 404
    
    except Exception as e:
        logger.error(f"[LifecycleAPI] Disable policy error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
        }), 500
