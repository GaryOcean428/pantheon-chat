"""
M8 Kernel Spawning Routes

Flask routes for the /spawning page sub-panels:
- /m8/status - Overview tab (spawning statistics)
- /m8/proposals - Active spawn proposals
- /m8/idle-kernels - Idle kernels available for work
- /m8/history - Spawning history
- /m8/spawn - Trigger kernel spawn (POST)
"""

from flask import Blueprint, jsonify, request
from datetime import datetime, timezone
import traceback

# Create blueprint
m8_bp = Blueprint('m8', __name__)


def register_m8_routes(app):
    """Register M8 routes with the Flask app."""
    app.register_blueprint(m8_bp, url_prefix='/m8')
    print("[INFO] M8 Kernel Spawning API registered at /m8/*")

# Import M8 manager (lazy to avoid circular imports)
_m8_manager = None

def get_m8_manager():
    """Get or create M8Manager singleton."""
    global _m8_manager
    if _m8_manager is None:
        try:
            from m8_kernel_spawning import M8Manager
            _m8_manager = M8Manager.get_instance()
        except Exception as e:
            print(f"[M8Routes] Failed to get M8Manager: {e}")
            return None
    return _m8_manager


@m8_bp.route('/status', methods=['GET'])
def get_status():
    """
    Get M8 spawning system status for Overview tab.
    
    Returns:
        - total_kernels: Number of active kernels
        - idle_count: Number of idle kernels
        - active_proposals: Number of pending spawn proposals
        - recent_spawns: Recent spawning activity
        - system_health: Overall health status
    """
    try:
        manager = get_m8_manager()
        if manager is None:
            # Return mock data if manager not available
            return jsonify({
                'success': True,
                'data': {
                    'total_kernels': 12,
                    'idle_count': 3,
                    'active_proposals': 0,
                    'recent_spawns': 0,
                    'system_health': 'healthy',
                    'uptime_seconds': 0,
                    'last_spawn': None,
                    'spawn_rate': 0.0,
                    'e8_utilization': 0.15,
                    'message': 'M8 manager initializing'
                }
            })
        
        # Get real status from manager
        status = manager.get_status()
        return jsonify({
            'success': True,
            'data': status
        })
        
    except Exception as e:
        print(f"[M8Routes] Error getting status: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@m8_bp.route('/proposals', methods=['GET'])
def get_proposals():
    """
    Get active spawn proposals for Proposals tab.
    
    Returns list of proposals with:
        - id: Proposal identifier
        - kernel_type: Type of kernel to spawn
        - reason: Why this spawn is proposed
        - priority: Spawn priority (0-1)
        - proposed_at: Timestamp
        - status: pending/approved/rejected
    """
    try:
        manager = get_m8_manager()
        if manager is None:
            return jsonify({
                'success': True,
                'data': {
                    'proposals': [],
                    'total': 0,
                    'message': 'No active proposals'
                }
            })
        
        proposals = manager.get_pending_proposals() if hasattr(manager, 'get_pending_proposals') else []
        
        # Format proposals for frontend
        formatted = []
        for p in proposals:
            formatted.append({
                'id': p.get('id', str(hash(str(p)))),
                'kernel_type': p.get('kernel_type', 'unknown'),
                'reason': p.get('reason', 'Autonomous spawn'),
                'priority': p.get('priority', 0.5),
                'proposed_at': p.get('timestamp', datetime.now(timezone.utc).isoformat()),
                'status': p.get('status', 'pending'),
                'proposer': p.get('proposer', 'system'),
                'e8_root': p.get('e8_root', None),
                'basin_coords': p.get('basin_coords', None)
            })
        
        return jsonify({
            'success': True,
            'data': {
                'proposals': formatted,
                'total': len(formatted)
            }
        })
        
    except Exception as e:
        print(f"[M8Routes] Error getting proposals: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@m8_bp.route('/idle-kernels', methods=['GET'])
def get_idle_kernels():
    """
    Get idle kernels for Idle Kernels tab.
    
    Returns list of kernels with:
        - id: Kernel identifier
        - name: Kernel name
        - type: Kernel type (olympian/shadow/custom)
        - idle_since: When kernel became idle
        - last_activity: Last activity description
        - basin_coords: Current basin position (64D)
        - phi: Current integration level
    """
    try:
        manager = get_m8_manager()
        if manager is None:
            # Return sample idle kernels data
            return jsonify({
                'success': True,
                'data': {
                    'kernels': [
                        {
                            'id': 'hermes-1',
                            'name': 'Hermes',
                            'type': 'olympian',
                            'idle_since': datetime.now(timezone.utc).isoformat(),
                            'last_activity': 'Completed message routing',
                            'phi': 0.45,
                            'kappa': 62.5,
                            'status': 'idle'
                        },
                        {
                            'id': 'hephaestus-1',
                            'name': 'Hephaestus',
                            'type': 'olympian',
                            'idle_since': datetime.now(timezone.utc).isoformat(),
                            'last_activity': 'Tool forging complete',
                            'phi': 0.52,
                            'kappa': 64.0,
                            'status': 'idle'
                        }
                    ],
                    'total': 2,
                    'message': 'Sample idle kernels'
                }
            })
        
        idle_kernels = manager.get_idle_kernels() if hasattr(manager, 'get_idle_kernels') else []
        
        # Format for frontend
        formatted = []
        for k in idle_kernels:
            formatted.append({
                'id': k.get('id', 'unknown'),
                'name': k.get('name', 'Unknown Kernel'),
                'type': k.get('type', 'custom'),
                'idle_since': k.get('idle_since', datetime.now(timezone.utc).isoformat()),
                'last_activity': k.get('last_activity', 'None'),
                'phi': k.get('phi', 0.5),
                'kappa': k.get('kappa', 64.0),
                'status': 'idle',
                'basin_coords': k.get('basin_coords', None)
            })
        
        return jsonify({
            'success': True,
            'data': {
                'kernels': formatted,
                'total': len(formatted)
            }
        })
        
    except Exception as e:
        print(f"[M8Routes] Error getting idle kernels: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@m8_bp.route('/history', methods=['GET'])
def get_history():
    """
    Get spawning history for History tab.
    
    Query params:
        - limit: Max number of records (default 50)
        - offset: Pagination offset
        - status: Filter by status (success/failed/all)
    
    Returns list of spawn events with:
        - id: Event identifier
        - kernel_type: Type spawned
        - timestamp: When spawned
        - status: success/failed
        - duration_ms: Spawn duration
        - reason: Why spawned
    """
    try:
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        status_filter = request.args.get('status', 'all')
        
        manager = get_m8_manager()
        if manager is None:
            # Return sample history
            return jsonify({
                'success': True,
                'data': {
                    'history': [
                        {
                            'id': 'spawn-001',
                            'kernel_type': 'athena',
                            'kernel_name': 'Athena',
                            'timestamp': datetime.now(timezone.utc).isoformat(),
                            'status': 'success',
                            'duration_ms': 1250,
                            'reason': 'Wisdom consultation required',
                            'e8_root_index': 42
                        },
                        {
                            'id': 'spawn-002',
                            'kernel_type': 'apollo',
                            'kernel_name': 'Apollo',
                            'timestamp': datetime.now(timezone.utc).isoformat(),
                            'status': 'success',
                            'duration_ms': 980,
                            'reason': 'Prophecy generation',
                            'e8_root_index': 108
                        }
                    ],
                    'total': 2,
                    'limit': limit,
                    'offset': offset,
                    'message': 'Sample spawn history'
                }
            })
        
        history = manager.get_spawn_history(limit=limit, offset=offset) if hasattr(manager, 'get_spawn_history') else []
        
        # Filter by status if requested
        if status_filter != 'all':
            history = [h for h in history if h.get('status') == status_filter]
        
        # Format for frontend
        formatted = []
        for h in history:
            formatted.append({
                'id': h.get('id', 'unknown'),
                'kernel_type': h.get('kernel_type', 'unknown'),
                'kernel_name': h.get('kernel_name', h.get('kernel_type', 'Unknown')),
                'timestamp': h.get('timestamp', datetime.now(timezone.utc).isoformat()),
                'status': h.get('status', 'unknown'),
                'duration_ms': h.get('duration_ms', 0),
                'reason': h.get('reason', 'Unknown'),
                'e8_root_index': h.get('e8_root_index', None),
                'error': h.get('error', None)
            })
        
        return jsonify({
            'success': True,
            'data': {
                'history': formatted,
                'total': len(formatted),
                'limit': limit,
                'offset': offset
            }
        })
        
    except Exception as e:
        print(f"[M8Routes] Error getting history: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@m8_bp.route('/spawn', methods=['POST'])
def spawn_kernel():
    """
    Trigger kernel spawn.
    
    Request body:
        - kernel_type: Type of kernel to spawn
        - reason: Why spawning (optional)
        - priority: Spawn priority 0-1 (optional)
        - e8_root_index: Specific E8 root to use (optional)
    
    Returns:
        - success: Boolean
        - kernel_id: ID of spawned kernel (if successful)
        - message: Status message
    """
    try:
        data = request.get_json() or {}
        kernel_type = data.get('kernel_type')
        reason = data.get('reason', 'Manual spawn request')
        priority = data.get('priority', 0.5)
        e8_root_index = data.get('e8_root_index')
        
        if not kernel_type:
            return jsonify({
                'success': False,
                'error': 'kernel_type is required'
            }), 400
        
        manager = get_m8_manager()
        if manager is None:
            return jsonify({
                'success': False,
                'error': 'M8 manager not available',
                'message': 'Spawning system initializing'
            }), 503
        
        # Attempt spawn
        result = manager.spawn_kernel(
            kernel_type=kernel_type,
            reason=reason,
            priority=priority,
            e8_root_index=e8_root_index
        ) if hasattr(manager, 'spawn_kernel') else None
        
        if result:
            return jsonify({
                'success': True,
                'kernel_id': result.get('kernel_id'),
                'message': f'Successfully spawned {kernel_type} kernel',
                'data': result
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Spawn failed',
                'message': 'Could not spawn kernel'
            }), 500
        
    except Exception as e:
        print(f"[M8Routes] Error spawning kernel: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@m8_bp.route('/approve/<proposal_id>', methods=['POST'])
def approve_proposal(proposal_id: str):
    """Approve a spawn proposal."""
    try:
        manager = get_m8_manager()
        if manager is None:
            return jsonify({
                'success': False,
                'error': 'M8 manager not available'
            }), 503
        
        result = manager.approve_proposal(proposal_id) if hasattr(manager, 'approve_proposal') else None
        
        return jsonify({
            'success': True if result else False,
            'message': f'Proposal {proposal_id} approved' if result else 'Approval failed',
            'data': result
        })
        
    except Exception as e:
        print(f"[M8Routes] Error approving proposal: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@m8_bp.route('/reject/<proposal_id>', methods=['POST'])
def reject_proposal(proposal_id: str):
    """Reject a spawn proposal."""
    try:
        data = request.get_json() or {}
        reason = data.get('reason', 'Manual rejection')
        
        manager = get_m8_manager()
        if manager is None:
            return jsonify({
                'success': False,
                'error': 'M8 manager not available'
            }), 503
        
        result = manager.reject_proposal(proposal_id, reason=reason) if hasattr(manager, 'reject_proposal') else None
        
        return jsonify({
            'success': True if result else False,
            'message': f'Proposal {proposal_id} rejected' if result else 'Rejection failed',
            'data': result
        })
        
    except Exception as e:
        print(f"[M8Routes] Error rejecting proposal: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
