"""
Pantheon Health Governance Routes

Flask routes for kernel health governance:
- /governance/health - Fleet health report
- /governance/kernels - List all monitored kernels
- /governance/kernel/<id> - Get specific kernel vitals
- /governance/care-plans - Active care plans
- /governance/proposals - Spawn proposals
- /governance/vote - Cast vote on spawn proposal
"""

from flask import Blueprint, jsonify, request
from datetime import datetime
import traceback

governance_bp = Blueprint('governance', __name__)


def register_governance_routes(app):
    """Register Governance routes with the Flask app."""
    app.register_blueprint(governance_bp, url_prefix='/governance')
    print("[INFO] Pantheon Health Governance API registered at /governance/*")


_governance_instance = None


def get_governance():
    """Get or create governance singleton."""
    global _governance_instance
    if _governance_instance is None:
        try:
            from olympus.pantheon_health_governance import get_health_governance
            _governance_instance = get_health_governance()
        except Exception as e:
            print(f"[GovernanceRoutes] Failed to get governance: {e}")
            return None
    return _governance_instance


@governance_bp.route('/health', methods=['GET'])
def get_fleet_health():
    """
    Get fleet health report from Pantheon governance.
    
    Returns comprehensive health statistics including:
    - Population count
    - Average vitality
    - Status distribution
    - Active care plans
    - Governance statistics
    """
    try:
        governance = get_governance()
        if governance is None:
            return jsonify({
                'success': False,
                'error': 'Governance system not initialized',
                'data': {
                    'population': 0,
                    'average_vitality': 0,
                    'status_distribution': {},
                    'active_care_plans': 0,
                    'governance_stats': {}
                }
            })
        
        report = governance.get_fleet_health_report()
        return jsonify({
            'success': True,
            'data': report,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@governance_bp.route('/kernels', methods=['GET'])
def list_monitored_kernels():
    """List all kernels under governance monitoring."""
    try:
        governance = get_governance()
        if governance is None:
            return jsonify({'success': False, 'error': 'Governance not available', 'kernels': []})
        
        kernels = []
        for kernel_id, vitals in governance.kernel_vitals.items():
            kernels.append({
                'kernel_id': kernel_id,
                'status': vitals.get_status().value,
                'vitality': vitals.compute_vitality_score(),
                'phi': vitals.phi,
                'stress': vitals.stress,
                'consecutive_failures': vitals.consecutive_failures,
                'generation': vitals.generation,
                'has_care_plan': kernel_id in governance.care_plans
            })
        
        return jsonify({
            'success': True,
            'kernels': kernels,
            'count': len(kernels)
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@governance_bp.route('/kernel/<kernel_id>', methods=['GET'])
def get_kernel_vitals(kernel_id):
    """Get detailed vitals for a specific kernel."""
    try:
        governance = get_governance()
        if governance is None:
            return jsonify({'success': False, 'error': 'Governance not available'})
        
        if kernel_id not in governance.kernel_vitals:
            return jsonify({'success': False, 'error': f'Kernel {kernel_id} not found'}), 404
        
        vitals = governance.kernel_vitals[kernel_id]
        care_plan = governance.care_plans.get(kernel_id)
        
        data = {
            'kernel_id': kernel_id,
            'status': vitals.get_status().value,
            'vitality': vitals.compute_vitality_score(),
            'phi': vitals.phi,
            'kappa': vitals.kappa,
            'kappa_drift': vitals.kappa_drift,
            'stress': vitals.stress,
            'consecutive_failures': vitals.consecutive_failures,
            'recovery_debt': vitals.recovery_debt,
            'basin_stability': vitals.basin_stability,
            'generation': vitals.generation,
            'age_cycles': vitals.age_cycles,
            'care_plan': None
        }
        
        if care_plan:
            data['care_plan'] = {
                'stage': care_plan.stage,
                'max_stages': care_plan.max_stages,
                'interventions': [i.value for i in care_plan.interventions],
                'assigned_healer': care_plan.assigned_healer,
                'escalation_count': care_plan.escalation_count,
                'notes': care_plan.notes
            }
        
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@governance_bp.route('/care-plans', methods=['GET'])
def list_care_plans():
    """List all active care plans."""
    try:
        governance = get_governance()
        if governance is None:
            return jsonify({'success': False, 'error': 'Governance not available', 'plans': []})
        
        plans = []
        for kernel_id, plan in governance.care_plans.items():
            plans.append({
                'kernel_id': kernel_id,
                'stage': plan.stage,
                'max_stages': plan.max_stages,
                'interventions': [i.value for i in plan.interventions],
                'assigned_healer': plan.assigned_healer,
                'escalation_count': plan.escalation_count,
                'created_at': plan.created_at
            })
        
        return jsonify({
            'success': True,
            'plans': plans,
            'count': len(plans)
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@governance_bp.route('/proposals', methods=['GET'])
def list_spawn_proposals():
    """List all spawn proposals and their vote status."""
    try:
        governance = get_governance()
        if governance is None:
            return jsonify({'success': False, 'error': 'Governance not available', 'proposals': []})
        
        proposals = []
        for proposal_id, proposal in governance.spawn_proposals.items():
            votes_summary = {}
            for god, vote in proposal.votes.items():
                votes_summary[god] = vote.value
            
            proposals.append({
                'proposal_id': proposal_id,
                'reason': proposal.reason,
                'proposed_by': proposal.proposed_by,
                'timestamp': proposal.timestamp,
                'votes': votes_summary,
                'consensus_reached': proposal.consensus_reached,
                'approved': proposal.approved
            })
        
        return jsonify({
            'success': True,
            'proposals': proposals,
            'count': len(proposals)
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@governance_bp.route('/propose-spawn', methods=['POST'])
def propose_spawn():
    """Create a new spawn proposal for god voting."""
    try:
        governance = get_governance()
        if governance is None:
            return jsonify({'success': False, 'error': 'Governance not available'}), 503
        
        data = request.get_json() or {}
        reason = data.get('reason', 'manual_request')
        proposed_by = data.get('proposed_by', 'Zeus')
        workload_context = data.get('workload', {})
        population_context = data.get('population', {})
        
        proposal = governance.propose_spawn(
            reason=reason,
            proposed_by=proposed_by,
            workload_context=workload_context,
            population_context=population_context
        )
        
        return jsonify({
            'success': True,
            'proposal_id': proposal.proposal_id,
            'message': f'Spawn proposal created by {proposed_by}'
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@governance_bp.route('/vote', methods=['POST'])
def vote_on_proposal():
    """Cast a vote on a spawn proposal."""
    try:
        governance = get_governance()
        if governance is None:
            return jsonify({'success': False, 'error': 'Governance not available'}), 503
        
        data = request.get_json() or {}
        proposal_id = data.get('proposal_id')
        god_name = data.get('god_name')
        vote_value = data.get('vote', 'abstain').lower()
        
        if not proposal_id or not god_name:
            return jsonify({
                'success': False,
                'error': 'Missing proposal_id or god_name'
            }), 400
        
        from olympus.pantheon_health_governance import SpawnProposalVote
        vote_map = {
            'approve': SpawnProposalVote.APPROVE,
            'reject': SpawnProposalVote.REJECT,
            'abstain': SpawnProposalVote.ABSTAIN,
            'defer': SpawnProposalVote.DEFER
        }
        
        vote = vote_map.get(vote_value, SpawnProposalVote.ABSTAIN)
        success = governance.vote_on_spawn(proposal_id, god_name, vote)
        
        if success:
            proposal = governance.spawn_proposals.get(proposal_id)
            return jsonify({
                'success': True,
                'message': f'{god_name} voted {vote_value} on {proposal_id}',
                'consensus_reached': proposal.consensus_reached if proposal else False,
                'approved': proposal.approved if proposal else False
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Vote failed - check proposal_id and god_name'
            }), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@governance_bp.route('/stats', methods=['GET'])
def get_governance_stats():
    """Get governance statistics."""
    try:
        governance = get_governance()
        if governance is None:
            return jsonify({
                'success': False,
                'error': 'Governance not available',
                'stats': {}
            })
        
        return jsonify({
            'success': True,
            'stats': {
                'total_kernels_monitored': len(governance.kernel_vitals),
                'active_care_plans': len(governance.care_plans),
                'pending_proposals': sum(1 for p in governance.spawn_proposals.values() if not p.consensus_reached),
                'spawns_blocked': governance.total_spawns_blocked,
                'deaths_prevented': governance.total_deaths_prevented,
                'total_interventions': governance.total_interventions,
                'voting_gods': governance.VOTING_GODS,
                'consensus_threshold': governance.CONSENSUS_THRESHOLD
            }
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500
