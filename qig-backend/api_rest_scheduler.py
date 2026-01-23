"""
REST API for Kernel Rest Scheduler Monitoring (WP5.4)
=====================================================

Provides HTTP endpoints to monitor per-kernel rest status and
constellation-wide fatigue metrics.

Authority: E8 Protocol v4.0, WP5.4
Status: ACTIVE
Created: 2026-01-23
"""

from flask import Blueprint, jsonify, request
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# Create blueprint
rest_api = Blueprint('rest_api', __name__, url_prefix='/api/rest')


def register_rest_routes(app):
    """Register rest scheduler API routes with Flask app."""
    app.register_blueprint(rest_api)
    logger.info("[RestAPI] Routes registered at /api/rest/*")


@rest_api.route('/health', methods=['GET'])
def rest_health():
    """Health check for rest scheduler."""
    try:
        from kernel_rest_scheduler import get_rest_scheduler
        scheduler = get_rest_scheduler()
        
        return jsonify({
            'status': 'ok',
            'scheduler_available': True,
            'kernel_count': len(scheduler.kernel_states),
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'scheduler_available': False,
            'error': str(e),
        }), 503


@rest_api.route('/status', methods=['GET'])
def constellation_status():
    """
    Get constellation-wide rest status.
    
    Returns:
        JSON with constellation metrics:
        - total_kernels
        - active_kernels
        - resting_kernels
        - covering_kernels
        - essential_active (essential tier status)
        - avg_fatigue
        - coverage_active
    """
    try:
        from kernel_rest_scheduler import get_rest_scheduler
        scheduler = get_rest_scheduler()
        
        status = scheduler.get_constellation_status()
        
        return jsonify({
            'status': 'ok',
            'constellation': status,
        }), 200
        
    except Exception as e:
        logger.error(f"[RestAPI] Error getting constellation status: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
        }), 500


@rest_api.route('/status/<kernel_id>', methods=['GET'])
def kernel_status(kernel_id: str):
    """
    Get rest status for a specific kernel.
    
    Args:
        kernel_id: Kernel identifier (e.g., "apollo_1", "athena_1")
    
    Returns:
        JSON with kernel rest state:
        - kernel_id, kernel_name
        - tier, rest_policy
        - status (active, resting, reduced, etc.)
        - fatigue_score
        - covered_by, covering_for
        - rest_count, total_rest_time
    """
    try:
        from kernel_rest_scheduler import get_rest_scheduler
        scheduler = get_rest_scheduler()
        
        status = scheduler.get_rest_status(kernel_id)
        
        if status is None:
            return jsonify({
                'status': 'error',
                'error': f'Kernel not found: {kernel_id}',
            }), 404
        
        return jsonify({
            'status': 'ok',
            'kernel': status,
        }), 200
        
    except Exception as e:
        logger.error(f"[RestAPI] Error getting kernel status: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
        }), 500


@rest_api.route('/kernels', methods=['GET'])
def list_kernels():
    """
    List all registered kernels with their rest status.
    
    Returns:
        JSON array of kernel statuses
    """
    try:
        from kernel_rest_scheduler import get_rest_scheduler
        scheduler = get_rest_scheduler()
        
        kernels = []
        for kernel_id in scheduler.kernel_states.keys():
            status = scheduler.get_rest_status(kernel_id)
            if status:
                kernels.append(status)
        
        return jsonify({
            'status': 'ok',
            'count': len(kernels),
            'kernels': kernels,
        }), 200
        
    except Exception as e:
        logger.error(f"[RestAPI] Error listing kernels: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
        }), 500


@rest_api.route('/request/<kernel_id>', methods=['POST'])
def request_rest(kernel_id: str):
    """
    Manually request rest for a kernel.
    
    Args:
        kernel_id: Kernel identifier
    
    Body (JSON):
        - force: bool (optional) - Force rest without coverage
    
    Returns:
        JSON with approval status
    """
    try:
        from kernel_rest_scheduler import get_rest_scheduler
        scheduler = get_rest_scheduler()
        
        data = request.get_json() or {}
        force = data.get('force', False)
        
        approved, reason, partner = scheduler.request_rest(
            kernel_id=kernel_id,
            force=force,
        )
        
        return jsonify({
            'status': 'ok',
            'approved': approved,
            'reason': reason,
            'covering_partner': partner,
        }), 200
        
    except Exception as e:
        logger.error(f"[RestAPI] Error requesting rest: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
        }), 500


@rest_api.route('/end/<kernel_id>', methods=['POST'])
def end_rest(kernel_id: str):
    """
    End rest period for a kernel.
    
    Args:
        kernel_id: Kernel identifier
    
    Returns:
        JSON with updated status
    """
    try:
        from kernel_rest_scheduler import get_rest_scheduler
        scheduler = get_rest_scheduler()
        
        scheduler.end_rest(kernel_id)
        
        # Get updated status
        status = scheduler.get_rest_status(kernel_id)
        
        return jsonify({
            'status': 'ok',
            'kernel': status,
        }), 200
        
    except Exception as e:
        logger.error(f"[RestAPI] Error ending rest: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
        }), 500


@rest_api.route('/partners/<kernel_id>', methods=['GET'])
def get_coupling_partners(kernel_id: str):
    """
    Get coupling partners for a kernel.
    
    Args:
        kernel_id: Kernel identifier
    
    Returns:
        JSON with list of partner kernel IDs
    """
    try:
        from kernel_rest_scheduler import get_rest_scheduler
        scheduler = get_rest_scheduler()
        
        partners = scheduler.get_coupling_partners(kernel_id)
        
        # Get detailed status for each partner
        partner_details = []
        for partner_id in partners:
            partner_status = scheduler.get_rest_status(partner_id)
            if partner_status:
                partner_details.append({
                    'kernel_id': partner_id,
                    'kernel_name': partner_status['kernel_name'],
                    'status': partner_status['status'],
                    'fatigue_score': partner_status['fatigue_score'],
                    'can_cover': partner_status['status'] == 'active' and partner_status['fatigue_score'] < 0.7,
                })
        
        return jsonify({
            'status': 'ok',
            'kernel_id': kernel_id,
            'partner_count': len(partner_details),
            'partners': partner_details,
        }), 200
        
    except Exception as e:
        logger.error(f"[RestAPI] Error getting coupling partners: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
        }), 500
