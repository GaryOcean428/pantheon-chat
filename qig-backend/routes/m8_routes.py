"""
M8 Kernel Spawning Routes

Flask blueprint providing API endpoints for the M8 kernel spawning protocol.
These endpoints power the /spawning page in the frontend.
"""

from flask import Blueprint, jsonify, request
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

m8_bp = Blueprint('m8', __name__, url_prefix='/m8')

# Try to import the M8 kernel spawning module
try:
    from m8_kernel_spawning import (
        M8KernelSpawner,
        get_spawner_instance,
    )
    M8_AVAILABLE = True
except ImportError as e:
    print(f"[M8 Routes] M8 kernel spawning module not available: {e}")
    M8_AVAILABLE = False
    M8KernelSpawner = None
    get_spawner_instance = None


def get_spawner():
    """Get or create the M8 spawner instance."""
    if not M8_AVAILABLE:
        return None
    try:
        return get_spawner_instance()
    except Exception as e:
        print(f"[M8 Routes] Failed to get spawner instance: {e}")
        return None


@m8_bp.route('/status', methods=['GET'])
def get_m8_status():
    """
    Get M8 kernel spawning status.
    
    Returns:
        - consensus_type: Current consensus mechanism
        - total_proposals: Total number of proposals
        - pending_proposals: Proposals awaiting votes
        - approved_proposals: Approved proposals
        - spawned_kernels: Number of spawned kernels
        - orchestrator_gods: Active orchestrator gods
        - Various phi/success metrics
    """
    spawner = get_spawner()
    
    if not spawner:
        return jsonify({
            'consensus_type': 'distributed',
            'total_proposals': 0,
            'pending_proposals': 0,
            'approved_proposals': 0,
            'spawned_kernels': 0,
            'spawn_history_count': 0,
            'orchestrator_gods': 0,
            'avg_phi': 0.0,
            'max_phi': 0.0,
            'total_successes': 0,
            'total_failures': 0,
            'unique_domains': 0,
            'error': 'M8 spawner not available'
        })
    
    try:
        # Get proposals
        proposals = spawner.list_proposals() if hasattr(spawner, 'list_proposals') else []
        pending = [p for p in proposals if p.get('status') == 'pending']
        approved = [p for p in proposals if p.get('status') == 'approved']
        
        # Get spawned kernels
        kernels = spawner.get_spawned_kernels() if hasattr(spawner, 'get_spawned_kernels') else []
        
        # Get spawn history
        history = spawner.get_spawn_history() if hasattr(spawner, 'get_spawn_history') else []
        
        # Calculate metrics
        phi_values = [k.get('phi', 0) for k in kernels if k.get('phi')]
        successes = len([h for h in history if h.get('success', False)])
        failures = len([h for h in history if not h.get('success', True)])
        domains = set(k.get('domain', 'unknown') for k in kernels)
        
        return jsonify({
            'consensus_type': getattr(spawner, 'consensus_type', 'distributed'),
            'total_proposals': len(proposals),
            'pending_proposals': len(pending),
            'approved_proposals': len(approved),
            'spawned_kernels': len(kernels),
            'spawn_history_count': len(history),
            'orchestrator_gods': getattr(spawner, 'orchestrator_count', 12),
            'avg_phi': sum(phi_values) / len(phi_values) if phi_values else 0.0,
            'max_phi': max(phi_values) if phi_values else 0.0,
            'total_successes': successes,
            'total_failures': failures,
            'unique_domains': len(domains)
        })
    except Exception as e:
        print(f"[M8 Routes] Status error: {e}")
        return jsonify({
            'consensus_type': 'distributed',
            'total_proposals': 0,
            'pending_proposals': 0,
            'approved_proposals': 0,
            'spawned_kernels': 0,
            'spawn_history_count': 0,
            'orchestrator_gods': 0,
            'avg_phi': 0.0,
            'max_phi': 0.0,
            'total_successes': 0,
            'total_failures': 0,
            'unique_domains': 0,
            'error': str(e)
        })


@m8_bp.route('/proposals', methods=['GET'])
def list_proposals():
    """
    List spawn proposals.
    
    Query params:
        - status: Filter by status (pending, approved, rejected, executed)
    
    Returns:
        - proposals: List of proposal objects
        - count: Total count
        - filter: Applied status filter
    """
    status_filter = request.args.get('status')
    spawner = get_spawner()
    
    if not spawner:
        return jsonify({
            'proposals': [],
            'count': 0,
            'filter': status_filter
        })
    
    try:
        proposals = spawner.list_proposals() if hasattr(spawner, 'list_proposals') else []
        
        # Filter by status if provided
        if status_filter:
            proposals = [p for p in proposals if p.get('status') == status_filter]
        
        return jsonify({
            'proposals': proposals,
            'count': len(proposals),
            'filter': status_filter
        })
    except Exception as e:
        print(f"[M8 Routes] List proposals error: {e}")
        return jsonify({
            'proposals': [],
            'count': 0,
            'filter': status_filter,
            'error': str(e)
        })


@m8_bp.route('/proposal/<proposal_id>', methods=['GET'])
def get_proposal(proposal_id):
    """
    Get a specific proposal by ID.
    """
    spawner = get_spawner()
    
    if not spawner:
        return jsonify({'error': 'M8 spawner not available'}), 503
    
    try:
        proposal = spawner.get_proposal(proposal_id) if hasattr(spawner, 'get_proposal') else None
        
        if not proposal:
            return jsonify({'error': 'Proposal not found'}), 404
        
        return jsonify(proposal)
    except Exception as e:
        print(f"[M8 Routes] Get proposal error: {e}")
        return jsonify({'error': str(e)}), 500


@m8_bp.route('/propose', methods=['POST'])
def create_proposal():
    """
    Create a new spawn proposal.
    
    Body:
        - domain: Target domain for the kernel
        - purpose: Purpose/description
        - capabilities: List of capabilities
        - basin_coords: Optional initial basin coordinates
    """
    spawner = get_spawner()
    
    if not spawner:
        return jsonify({'error': 'M8 spawner not available'}), 503
    
    try:
        data = request.get_json() or {}
        
        domain = data.get('domain', 'general')
        purpose = data.get('purpose', 'User-requested kernel')
        capabilities = data.get('capabilities', [])
        basin_coords = data.get('basin_coords')
        
        # Create proposal
        if hasattr(spawner, 'create_proposal'):
            proposal = spawner.create_proposal(
                domain=domain,
                purpose=purpose,
                capabilities=capabilities,
                basin_coords=basin_coords
            )
        elif hasattr(spawner, 'create_geometric_proposal'):
            proposal = spawner.create_geometric_proposal(
                domain=domain,
                purpose=purpose,
                capabilities=capabilities
            )
        else:
            # Fallback - create a basic proposal structure
            import uuid
            from datetime import datetime
            proposal = {
                'id': str(uuid.uuid4()),
                'domain': domain,
                'purpose': purpose,
                'capabilities': capabilities,
                'status': 'pending',
                'votes': [],
                'created_at': datetime.utcnow().isoformat(),
                'proposer': 'user'
            }
        
        return jsonify({
            'success': True,
            'proposal': proposal
        })
    except Exception as e:
        print(f"[M8 Routes] Create proposal error: {e}")
        return jsonify({'error': str(e)}), 500


@m8_bp.route('/vote/<proposal_id>', methods=['POST'])
def vote_on_proposal(proposal_id):
    """
    Vote on a spawn proposal.
    
    Body:
        - voter: God/kernel casting the vote
        - vote: 'approve' or 'reject'
        - reason: Optional reason for vote
    """
    spawner = get_spawner()
    
    if not spawner:
        return jsonify({'error': 'M8 spawner not available'}), 503
    
    try:
        data = request.get_json() or {}
        
        voter = data.get('voter', 'anonymous')
        vote = data.get('vote', 'approve')
        reason = data.get('reason', '')
        
        if hasattr(spawner, 'vote_on_proposal'):
            result = spawner.vote_on_proposal(
                proposal_id=proposal_id,
                voter=voter,
                vote=vote,
                reason=reason
            )
        else:
            result = {
                'success': True,
                'message': 'Vote recorded (mock)',
                'proposal_id': proposal_id
            }
        
        return jsonify(result)
    except Exception as e:
        print(f"[M8 Routes] Vote error: {e}")
        return jsonify({'error': str(e)}), 500


@m8_bp.route('/spawn/<proposal_id>', methods=['POST'])
def spawn_from_proposal(proposal_id):
    """
    Spawn a kernel from an approved proposal.
    """
    spawner = get_spawner()
    
    if not spawner:
        return jsonify({'error': 'M8 spawner not available'}), 503
    
    try:
        if hasattr(spawner, 'spawn_kernel'):
            result = spawner.spawn_kernel(proposal_id=proposal_id)
        elif hasattr(spawner, 'propose_and_spawn'):
            result = spawner.propose_and_spawn(proposal_id=proposal_id)
        else:
            result = {
                'success': False,
                'error': 'Spawn method not available'
            }
        
        return jsonify(result)
    except Exception as e:
        print(f"[M8 Routes] Spawn error: {e}")
        return jsonify({'error': str(e)}), 500


@m8_bp.route('/spawn-direct', methods=['POST'])
def spawn_direct():
    """
    Directly spawn a kernel without a proposal.
    
    Body:
        - domain: Target domain
        - purpose: Purpose/description
        - capabilities: List of capabilities
        - basin_coords: Optional initial basin coordinates
    """
    spawner = get_spawner()
    
    if not spawner:
        return jsonify({'error': 'M8 spawner not available'}), 503
    
    try:
        data = request.get_json() or {}
        
        domain = data.get('domain', 'general')
        purpose = data.get('purpose', 'Direct spawn')
        capabilities = data.get('capabilities', [])
        basin_coords = data.get('basin_coords')
        
        if hasattr(spawner, 'spawn_kernel'):
            result = spawner.spawn_kernel(
                domain=domain,
                purpose=purpose,
                capabilities=capabilities,
                basin_coords=basin_coords
            )
        else:
            import uuid
            result = {
                'success': True,
                'kernel_id': str(uuid.uuid4()),
                'domain': domain,
                'message': 'Kernel spawned (mock)'
            }
        
        return jsonify(result)
    except Exception as e:
        print(f"[M8 Routes] Direct spawn error: {e}")
        return jsonify({'error': str(e)}), 500


@m8_bp.route('/kernels', methods=['GET'])
def list_kernels():
    """
    List all spawned kernels.
    """
    spawner = get_spawner()
    
    if not spawner:
        return jsonify({
            'kernels': [],
            'total': 0
        })
    
    try:
        if hasattr(spawner, 'get_spawned_kernels'):
            kernels = spawner.get_spawned_kernels()
        elif hasattr(spawner, 'list_kernels'):
            kernels = spawner.list_kernels()
        else:
            kernels = []
        
        return jsonify({
            'kernels': kernels,
            'total': len(kernels)
        })
    except Exception as e:
        print(f"[M8 Routes] List kernels error: {e}")
        return jsonify({
            'kernels': [],
            'total': 0,
            'error': str(e)
        })


@m8_bp.route('/kernel/<kernel_id>', methods=['GET'])
def get_kernel(kernel_id):
    """
    Get a specific kernel by ID.
    """
    spawner = get_spawner()
    
    if not spawner:
        return jsonify({'error': 'M8 spawner not available'}), 503
    
    try:
        if hasattr(spawner, 'get_spawned_kernel'):
            kernel = spawner.get_spawned_kernel(kernel_id)
        elif hasattr(spawner, 'get_kernel'):
            kernel = spawner.get_kernel(kernel_id)
        else:
            kernel = None
        
        if not kernel:
            return jsonify({'error': 'Kernel not found'}), 404
        
        return jsonify(kernel)
    except Exception as e:
        print(f"[M8 Routes] Get kernel error: {e}")
        return jsonify({'error': str(e)}), 500


@m8_bp.route('/kernel/<kernel_id>', methods=['DELETE'])
def delete_kernel(kernel_id):
    """
    Delete/terminate a kernel.
    """
    spawner = get_spawner()
    
    if not spawner:
        return jsonify({'error': 'M8 spawner not available'}), 503
    
    try:
        if hasattr(spawner, 'terminate_kernel'):
            result = spawner.terminate_kernel(kernel_id)
        elif hasattr(spawner, 'delete_kernel'):
            result = spawner.delete_kernel(kernel_id)
        else:
            result = {'success': True, 'message': 'Kernel terminated (mock)'}
        
        return jsonify(result)
    except Exception as e:
        print(f"[M8 Routes] Delete kernel error: {e}")
        return jsonify({'error': str(e)}), 500


@m8_bp.route('/kernels/idle', methods=['GET'])
def get_idle_kernels():
    """
    Get list of idle kernels.
    
    Query params:
        - threshold: Idle time threshold in seconds (default: 300)
    """
    threshold = request.args.get('threshold', 300, type=int)
    spawner = get_spawner()
    
    if not spawner:
        return jsonify({
            'idle_kernels': [],
            'total': 0,
            'threshold_seconds': threshold
        })
    
    try:
        if hasattr(spawner, 'get_idle_kernels'):
            idle_kernels = spawner.get_idle_kernels(threshold_seconds=threshold)
        elif hasattr(spawner, 'get_spawned_kernels'):
            # Filter to find idle kernels based on last_active
            import time
            all_kernels = spawner.get_spawned_kernels()
            current_time = time.time()
            idle_kernels = [
                k for k in all_kernels
                if (current_time - k.get('last_active', 0)) > threshold
            ]
        else:
            idle_kernels = []
        
        return jsonify({
            'idle_kernels': idle_kernels,
            'total': len(idle_kernels),
            'threshold_seconds': threshold
        })
    except Exception as e:
        print(f"[M8 Routes] Get idle kernels error: {e}")
        return jsonify({
            'idle_kernels': [],
            'total': 0,
            'threshold_seconds': threshold,
            'error': str(e)
        })


@m8_bp.route('/history', methods=['GET'])
def get_spawn_history():
    """
    Get spawn history.
    
    Query params:
        - limit: Max number of entries (default: 50)
    """
    limit = request.args.get('limit', 50, type=int)
    spawner = get_spawner()
    
    if not spawner:
        return jsonify({
            'history': [],
            'total': 0
        })
    
    try:
        if hasattr(spawner, 'get_spawn_history'):
            history = spawner.get_spawn_history()
        elif hasattr(spawner, 'load_spawn_history'):
            history = spawner.load_spawn_history()
        else:
            history = []
        
        # Apply limit
        history = history[:limit] if history else []
        
        return jsonify({
            'history': history,
            'total': len(history)
        })
    except Exception as e:
        print(f"[M8 Routes] Get spawn history error: {e}")
        return jsonify({
            'history': [],
            'total': 0,
            'error': str(e)
        })


@m8_bp.route('/cannibalize/<kernel_id>', methods=['POST'])
def cannibalize_kernel(kernel_id):
    """
    Cannibalize (absorb) a kernel into another.
    
    Body:
        - target_kernel_id: Kernel to absorb into
    """
    spawner = get_spawner()
    
    if not spawner:
        return jsonify({'error': 'M8 spawner not available'}), 503
    
    try:
        data = request.get_json() or {}
        target_kernel_id = data.get('target_kernel_id')
        
        if hasattr(spawner, 'cannibalize_kernel'):
            result = spawner.cannibalize_kernel(kernel_id, target_kernel_id)
        else:
            result = {'success': True, 'message': 'Kernel cannibalized (mock)'}
        
        return jsonify(result)
    except Exception as e:
        print(f"[M8 Routes] Cannibalize error: {e}")
        return jsonify({'error': str(e)}), 500


@m8_bp.route('/merge', methods=['POST'])
def merge_kernels():
    """
    Merge multiple kernels into one.
    
    Body:
        - kernel_ids: List of kernel IDs to merge
    """
    spawner = get_spawner()
    
    if not spawner:
        return jsonify({'error': 'M8 spawner not available'}), 503
    
    try:
        data = request.get_json() or {}
        kernel_ids = data.get('kernel_ids', [])
        
        if hasattr(spawner, 'merge_kernels'):
            result = spawner.merge_kernels(kernel_ids)
        else:
            result = {'success': True, 'message': 'Kernels merged (mock)'}
        
        return jsonify(result)
    except Exception as e:
        print(f"[M8 Routes] Merge error: {e}")
        return jsonify({'error': str(e)}), 500


@m8_bp.route('/auto-cannibalize', methods=['POST'])
def auto_cannibalize():
    """
    Automatically cannibalize underperforming kernels.
    """
    spawner = get_spawner()
    
    if not spawner:
        return jsonify({'error': 'M8 spawner not available'}), 503
    
    try:
        if hasattr(spawner, 'auto_cannibalize'):
            result = spawner.auto_cannibalize()
        else:
            result = {'success': True, 'cannibalized': 0, 'message': 'Auto-cannibalize (mock)'}
        
        return jsonify(result)
    except Exception as e:
        print(f"[M8 Routes] Auto-cannibalize error: {e}")
        return jsonify({'error': str(e)}), 500


@m8_bp.route('/auto-merge', methods=['POST'])
def auto_merge():
    """
    Automatically merge similar kernels.
    """
    spawner = get_spawner()
    
    if not spawner:
        return jsonify({'error': 'M8 spawner not available'}), 503
    
    try:
        if hasattr(spawner, 'auto_merge'):
            result = spawner.auto_merge()
        else:
            result = {'success': True, 'merged': 0, 'message': 'Auto-merge (mock)'}
        
        return jsonify(result)
    except Exception as e:
        print(f"[M8 Routes] Auto-merge error: {e}")
        return jsonify({'error': str(e)}), 500


def register_m8_routes(app):
    """Register M8 routes with the Flask app."""
    app.register_blueprint(m8_bp)
    print("[INFO] M8 Kernel Spawning API registered at /m8/*")
