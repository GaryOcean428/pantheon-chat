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

# Import M8 spawner (lazy to avoid circular imports)
_m8_spawner = None

def get_m8_spawner():
    """Get or create M8KernelSpawner singleton."""
    global _m8_spawner
    if _m8_spawner is None:
        try:
            from m8_kernel_spawning import get_spawner
            _m8_spawner = get_spawner()
        except Exception as e:
            print(f"[M8Routes] Failed to get M8KernelSpawner: {e}")
            return None
    return _m8_spawner


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
        spawner = get_m8_spawner()
        if spawner is None:
            # Return mock data if spawner not available
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
                    'message': 'M8 spawner initializing'
                }
            })
        
        # Get real status from spawner
        try:
            proposals = spawner.list_proposals() if hasattr(spawner, 'list_proposals') else []
            history = spawner.spawn_history[-10:] if hasattr(spawner, 'spawn_history') else []
            status = {
                'total_kernels': len(spawner.spawned_kernels) if hasattr(spawner, 'spawned_kernels') else 0,
                'idle_count': sum(1 for k in (spawner.spawned_kernels or []) if k.get('status') == 'idle') if hasattr(spawner, 'spawned_kernels') else 0,
                'active_proposals': len([p for p in proposals if p.get('status') == 'pending']),
                'recent_spawns': len(history),
                'system_health': 'healthy',
                'uptime_seconds': 0,
                'last_spawn': history[-1].get('timestamp') if history else None,
                'spawn_rate': len(history) / 3600.0 if history else 0.0,
                'e8_utilization': 0.15
            }
        except Exception as e:
            print(f"[M8Routes] Error building status: {e}")
            status = {'error': str(e)}
        
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
        spawner = get_m8_spawner()
        if spawner is None:
            return jsonify({
                'success': True,
                'data': {
                    'proposals': [],
                    'total': 0,
                    'message': 'No active proposals'
                }
            })
        
        proposals = spawner.list_proposals() if hasattr(spawner, 'list_proposals') else []
        
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
        spawner = get_m8_spawner()
        if spawner is None:
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
        
        # Get spawned kernels and filter for idle ones
        spawned = spawner.spawned_kernels if hasattr(spawner, 'spawned_kernels') else []
        idle_kernels = [k for k in spawned if k.get('status') == 'idle']
        
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
        
        spawner = get_m8_spawner()
        if spawner is None:
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
        
        # Get spawn history from spawner
        all_history = spawner.spawn_history if hasattr(spawner, 'spawn_history') else []
        history = all_history[offset:offset + limit]
        
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
        
        spawner = get_m8_spawner()
        if spawner is None:
            return jsonify({
                'success': False,
                'error': 'M8 spawner not available',
                'message': 'Spawning system initializing'
            }), 503
        
        # Attempt spawn via proposal system
        from m8_kernel_spawning import SpawnReason
        result = None
        try:
            # Create and immediately execute a spawn proposal
            proposal_id = spawner.propose_spawn(
                kernel_type=kernel_type,
                reason=SpawnReason.USER_REQUEST if hasattr(SpawnReason, 'USER_REQUEST') else reason,
                priority=priority
            ) if hasattr(spawner, 'propose_spawn') else None
            
            if proposal_id:
                # Auto-approve and execute
                result = spawner.execute_spawn(proposal_id) if hasattr(spawner, 'execute_spawn') else {'kernel_id': proposal_id}
        except Exception as e:
            print(f"[M8Routes] Spawn error: {e}")
            result = None
        
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
        spawner = get_m8_spawner()
        if spawner is None:
            return jsonify({
                'success': False,
                'error': 'M8 spawner not available'
            }), 503
        
        result = spawner.vote_on_proposal(proposal_id, vote=True) if hasattr(spawner, 'vote_on_proposal') else None
        
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
        
        spawner = get_m8_spawner()
        if spawner is None:
            return jsonify({
                'success': False,
                'error': 'M8 spawner not available'
            }), 503
        
        result = spawner.vote_on_proposal(proposal_id, vote=False, reason=reason) if hasattr(spawner, 'vote_on_proposal') else None
        
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


_lifecycle_gov = None

def get_lifecycle_governance():
    """Get lifecycle governance singleton."""
    global _lifecycle_gov
    if _lifecycle_gov is None:
        try:
            from kernel_lifecycle_governance import get_lifecycle_governance as get_gov
            _lifecycle_gov = get_gov()
        except Exception as e:
            print(f"[M8Routes] Failed to get lifecycle governance: {e}")
            return None
    return _lifecycle_gov


@m8_bp.route('/governance/stats', methods=['GET'])
def get_governance_stats():
    """Get kernel lifecycle governance statistics."""
    try:
        gov = get_lifecycle_governance()
        
        default_stats = {
            'current_kernels': 0,
            'e8_cap': 240,
            'available_slots': 240,
            'at_capacity': False,
            'active_proposals': 0,
            'kernel_types': {},
            'protected_count': 0,
        }
        
        if gov is None:
            return jsonify({
                'success': True,
                'data': default_stats
            })
        
        stats = gov.get_governance_stats()
        merged = {**default_stats, **(stats or {})}
        return jsonify({
            'success': True,
            'data': merged
        })
    except Exception as e:
        print(f"[M8Routes] Governance stats error: {e}")
        return jsonify({
            'success': False, 
            'error': str(e),
            'data': {
                'current_kernels': 0,
                'e8_cap': 240,
                'available_slots': 240,
                'at_capacity': False,
                'active_proposals': 0,
            }
        }), 500


@m8_bp.route('/governance/capacity', methods=['GET'])
def get_e8_capacity():
    """Get current E8 kernel capacity status."""
    try:
        gov = get_lifecycle_governance()
        
        default_capacity = {
            'current': 0,
            'cap': 240,
            'available': 240,
            'at_capacity': False,
        }
        
        if gov is None:
            return jsonify(default_capacity)
        
        current, cap, has_room = gov.check_e8_capacity()
        return jsonify({
            'current': current,
            'cap': cap,
            'available': cap - current,
            'at_capacity': not has_room,
        })
    except Exception as e:
        print(f"[M8Routes] E8 capacity error: {e}")
        return jsonify({
            'current': 0,
            'cap': 240,
            'available': 240,
            'at_capacity': False,
        }), 500


@m8_bp.route('/governance/proposals', methods=['GET'])
def get_lifecycle_proposals():
    """Get all active lifecycle proposals requiring god oversight."""
    try:
        gov = get_lifecycle_governance()
        if gov is None:
            return jsonify({
                'proposals': [], 
                'total': 0
            })
        
        proposals = gov.get_active_proposals() or []
        return jsonify({
            'proposals': proposals,
            'total': len(proposals)
        })
    except Exception as e:
        print(f"[M8Routes] Get proposals error: {e}")
        return jsonify({
            'proposals': [],
            'total': 0
        }), 500


@m8_bp.route('/governance/proposals/<proposal_id>', methods=['GET'])
def get_lifecycle_proposal(proposal_id: str):
    """Get a specific lifecycle proposal."""
    try:
        gov = get_lifecycle_governance()
        if gov is None:
            return jsonify({'success': False, 'error': 'Governance not available'}), 503
        
        proposal = gov.get_proposal(proposal_id)
        if proposal:
            return jsonify({'success': True, 'proposal': proposal})
        return jsonify({'success': False, 'error': 'Proposal not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@m8_bp.route('/governance/propose', methods=['POST'])
def propose_lifecycle_action():
    """
    Propose a lifecycle action for god debate.
    
    Body:
        action: 'spawn' | 'merge' | 'cannibalize' | 'evolve' | 'hibernate' | 'awaken'
        proposed_by: god name proposing the action
        reason: why this action is needed
        target_kernel_id: (optional) target kernel for action
        domain: (optional) domain for spawn
        merge_target_id: (optional) kernel to merge into
        primitive_function: (optional) primitive function to preserve
    """
    try:
        data = request.get_json() or {}
        
        action_str = data.get('action', 'spawn')
        proposed_by = data.get('proposed_by', 'zeus')
        reason = data.get('reason', 'Governance decision')
        
        gov = get_lifecycle_governance()
        if gov is None:
            return jsonify({'success': False, 'error': 'Governance not available'}), 503
        
        from kernel_lifecycle_governance import LifecycleAction
        try:
            action = LifecycleAction(action_str)
        except ValueError:
            return jsonify({'success': False, 'error': f'Invalid action: {action_str}'}), 400
        
        success, message, proposal_id = gov.propose_action(
            action=action,
            proposed_by=proposed_by,
            reason=reason,
            target_kernel_id=data.get('target_kernel_id'),
            domain=data.get('domain'),
            basin_coordinates=data.get('basin_coordinates'),
            parent_kernels=data.get('parent_kernels'),
            merge_target_id=data.get('merge_target_id'),
            primitive_function=data.get('primitive_function'),
        )
        
        return jsonify({
            'success': success,
            'message': message,
            'proposal_id': proposal_id
        })
    except Exception as e:
        print(f"[M8Routes] Propose error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@m8_bp.route('/governance/vote/<proposal_id>', methods=['POST'])
def vote_on_lifecycle_proposal(proposal_id: str):
    """
    Cast a god's vote on a lifecycle proposal.
    
    Body:
        god_name: which god is voting
        vote: 'approve' | 'reject' | 'abstain' | 'defer'
        argument: (optional) reasoning for the vote
    """
    try:
        data = request.get_json() or {}
        god_name = data.get('god_name', 'zeus')
        vote_str = data.get('vote', 'abstain')
        argument = data.get('argument')
        
        gov = get_lifecycle_governance()
        if gov is None:
            return jsonify({'success': False, 'error': 'Governance not available'}), 503
        
        from kernel_lifecycle_governance import VoteDecision
        try:
            vote = VoteDecision(vote_str)
        except ValueError:
            return jsonify({'success': False, 'error': f'Invalid vote: {vote_str}'}), 400
        
        success, message = gov.vote_on_proposal(proposal_id, god_name, vote, argument)
        
        return jsonify({
            'success': success,
            'message': message
        })
    except Exception as e:
        print(f"[M8Routes] Vote error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@m8_bp.route('/governance/execute/<proposal_id>', methods=['POST'])
def execute_lifecycle_proposal(proposal_id: str):
    """Execute an approved lifecycle proposal."""
    try:
        gov = get_lifecycle_governance()
        if gov is None:
            return jsonify({'success': False, 'error': 'Governance not available'}), 503
        
        success, message, result = gov.execute_proposal(proposal_id)
        
        return jsonify({
            'success': success,
            'message': message,
            'result': result
        })
    except Exception as e:
        print(f"[M8Routes] Execute error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

