"""
Search Budget API Routes

Endpoints for managing search provider budgets and preferences:
- GET /api/search/budget/status - Get current budget status
- POST /api/search/budget/toggle - Enable/disable providers
- POST /api/search/budget/limits - Set daily limits
- POST /api/search/budget/overage - Allow/disallow overage
- GET /api/search/budget/learning - Get learning metrics
"""

import logging
from flask import Blueprint, request, jsonify

logger = logging.getLogger(__name__)

search_budget_bp = Blueprint('search_budget', __name__, url_prefix='/api/search/budget')


def get_orchestrator():
    """Get the budget orchestrator lazily."""
    from search.search_budget_orchestrator import get_budget_orchestrator
    return get_budget_orchestrator()


@search_budget_bp.route('/status', methods=['GET'])
def get_budget_status():
    """Get current budget status for all providers."""
    try:
        orchestrator = get_orchestrator()
        status = orchestrator.get_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"[SearchBudget] Status error: {e}")
        return jsonify({'error': str(e)}), 500


@search_budget_bp.route('/context', methods=['GET'])
def get_budget_context():
    """Get budget context for kernel decisions."""
    try:
        orchestrator = get_orchestrator()
        context = orchestrator.get_budget_context()
        return jsonify(context.to_dict())
    except Exception as e:
        logger.error(f"[SearchBudget] Context error: {e}")
        return jsonify({'error': str(e)}), 500


@search_budget_bp.route('/toggle', methods=['POST'])
def toggle_provider():
    """Enable or disable a search provider."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No JSON body'}), 400
        
        provider = data.get('provider')
        enabled = data.get('enabled')
        
        if not provider:
            return jsonify({'success': False, 'error': 'Missing provider'}), 400
        
        if enabled is None:
            return jsonify({'success': False, 'error': 'Missing enabled flag'}), 400
        
        orchestrator = get_orchestrator()
        success = orchestrator.set_provider_enabled(provider, bool(enabled))
        
        if success:
            return jsonify({
                'success': True,
                'provider': provider,
                'enabled': enabled,
                'status': orchestrator.get_status()
            })
        else:
            return jsonify({'success': False, 'error': f'Unknown provider: {provider}'}), 400
    
    except Exception as e:
        logger.error(f"[SearchBudget] Toggle error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@search_budget_bp.route('/limits', methods=['POST'])
def set_daily_limits():
    """Set daily limits for providers."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No JSON body'}), 400
        
        orchestrator = get_orchestrator()
        updated = []
        
        for provider, limit in data.items():
            if provider in ['google', 'perplexity', 'tavily', 'duckduckgo']:
                if orchestrator.set_daily_limit(provider, int(limit)):
                    updated.append(provider)
        
        return jsonify({
            'success': True,
            'updated': updated,
            'status': orchestrator.get_status()
        })
    
    except Exception as e:
        logger.error(f"[SearchBudget] Limits error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@search_budget_bp.route('/overage', methods=['POST'])
def set_overage():
    """Allow or disallow exceeding daily limits."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No JSON body'}), 400
        
        allow = data.get('allow')
        if allow is None:
            return jsonify({'success': False, 'error': 'Missing allow flag'}), 400
        
        orchestrator = get_orchestrator()
        orchestrator.set_allow_overage(bool(allow))
        
        return jsonify({
            'success': True,
            'allow_overage': bool(allow),
            'status': orchestrator.get_status()
        })
    
    except Exception as e:
        logger.error(f"[SearchBudget] Overage error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@search_budget_bp.route('/learning', methods=['GET'])
def get_learning_metrics():
    """Get learning metrics for kernel evolution."""
    try:
        orchestrator = get_orchestrator()
        metrics = orchestrator.get_learning_metrics()
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"[SearchBudget] Learning metrics error: {e}")
        return jsonify({'error': str(e)}), 500


@search_budget_bp.route('/reset', methods=['POST'])
def reset_daily():
    """Manually reset daily counters (admin only)."""
    try:
        orchestrator = get_orchestrator()
        orchestrator.reset_daily()
        return jsonify({
            'success': True,
            'message': 'Daily counters reset',
            'status': orchestrator.get_status()
        })
    except Exception as e:
        logger.error(f"[SearchBudget] Reset error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


def register_search_budget_routes(app):
    """Register search budget routes with Flask app."""
    app.register_blueprint(search_budget_bp)
    logger.info("[SearchBudget] Registered routes at /api/search/budget")
    return 1
