"""
Training Monitoring API Routes
===============================

Exposes training progress and coherence metrics via HTTP API.

Provides endpoints for:
- Progress tracking (steps, topics, curriculum)
- Coherence evaluation (perplexity, degeneracy detection)
- Training status and health

Uses dependency injection to avoid circular imports.
"""

from flask import Blueprint, jsonify, request
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Create blueprint
training_monitor_bp = Blueprint('training_monitor', __name__, url_prefix='/api/training')

# Store reference to training integrator (dependency injection)
_training_integrator = None


def set_training_integrator(integrator):
    """
    Set the training integrator instance (dependency injection).
    
    This should be called during app initialization to inject the dependency.
    """
    global _training_integrator
    _training_integrator = integrator


def get_training_integrator():
    """Get the injected training integrator."""
    return _training_integrator


@training_monitor_bp.route('/status', methods=['GET'])
def get_training_status():
    """
    Get comprehensive training status.
    
    Returns:
        - Training active status
        - Outcome count
        - Progress metrics (steps, topics, curriculum)
        - Coherence statistics
        - Component connection status
    """
    integrator = get_training_integrator()
    if not integrator:
        return jsonify({
            'status': 'error',
            'error': 'Training integrator not available'
        }), 503
    
    try:
        status = integrator.get_training_status()
        return jsonify({
            'status': 'success',
            'data': status
        })
    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@training_monitor_bp.route('/progress', methods=['GET'])
def get_progress_metrics():
    """
    Get progress metrics for all training sessions.
    
    Query params:
        - session_id: Optional session ID to get specific session metrics
    
    Returns:
        - Global metrics (aggregated across all sessions)
        - Per-session metrics
    """
    integrator = get_training_integrator()
    if not integrator:
        return jsonify({
            'status': 'error',
            'error': 'Training integrator not available'
        }), 503
    
    try:
        session_id = request.args.get('session_id')
        
        if session_id:
            # Get specific session
            session_metrics = integrator.progress_tracker.get_or_create_session(session_id)
            return jsonify({
                'status': 'success',
                'data': session_metrics.to_dict()
            })
        else:
            # Get all sessions and global
            summary = integrator.progress_tracker.get_summary()
            return jsonify({
                'status': 'success',
                'data': summary
            })
    except Exception as e:
        logger.error(f"Error getting progress metrics: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@training_monitor_bp.route('/coherence/stats', methods=['GET'])
def get_coherence_stats():
    """
    Get coherence evaluation statistics.
    
    Returns:
        - Total samples
        - Average coherence
        - Min/max coherence
        - Standard deviation
    """
    integrator = get_training_integrator()
    if not integrator:
        return jsonify({
            'status': 'error',
            'error': 'Training integrator not available'
        }), 503
    
    try:
        stats = integrator.coherence_evaluator.get_stats()
        return jsonify({
            'status': 'success',
            'data': stats
        })
    except Exception as e:
        logger.error(f"Error getting coherence stats: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@training_monitor_bp.route('/coherence/trend', methods=['GET'])
def get_coherence_trend():
    """
    Get coherence trend analysis.
    
    Query params:
        - window: Number of recent samples to analyze (default: 50)
    
    Returns:
        - Current coherence
        - Average coherence
        - Coherence trend (slope)
        - Perplexity trend
        - Repetition trend
        - Entropy collapse trend
        - Degradation detected flag
    """
    integrator = get_training_integrator()
    if not integrator:
        return jsonify({
            'status': 'error',
            'error': 'Training integrator not available'
        }), 503
    
    try:
        window = int(request.args.get('window', 50))
        trend = integrator.coherence_evaluator.get_coherence_trend(window=window)
        return jsonify({
            'status': 'success',
            'data': trend
        })
    except Exception as e:
        logger.error(f"Error getting coherence trend: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@training_monitor_bp.route('/coherence/evaluate', methods=['POST'])
def evaluate_text_coherence():
    """
    Evaluate coherence of provided text.
    
    Request body:
        - text: Text to evaluate
        - basin_trajectory: Optional list of basin coordinates
    
    Returns:
        - Perplexity
        - Self-consistency
        - Long-range coherence
        - Repetition score
        - Entropy collapse score
        - Overall coherence
    """
    integrator = get_training_integrator()
    if not integrator:
        return jsonify({
            'status': 'error',
            'error': 'Training integrator not available'
        }), 503
    
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'status': 'error',
                'error': 'Missing required field: text'
            }), 400
        
        text = data['text']
        basin_trajectory = data.get('basin_trajectory')
        
        metrics = integrator.coherence_evaluator.evaluate(
            text=text,
            basin_trajectory=basin_trajectory
        )
        
        return jsonify({
            'status': 'success',
            'data': {
                'perplexity': metrics.perplexity,
                'self_consistency': metrics.self_consistency,
                'long_range_coherence': metrics.long_range_coherence,
                'repetition_score': metrics.repetition_score,
                'entropy_collapse_score': metrics.entropy_collapse_score,
                'overall_coherence': metrics.overall_coherence,
            }
        })
    except Exception as e:
        logger.error(f"Error evaluating coherence: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@training_monitor_bp.route('/health', methods=['GET'])
def get_training_health():
    """
    Get training system health check.
    
    Returns:
        - Component status (orchestrator, progress, coherence)
        - Training active status
        - Recent error flags
    """
    integrator = get_training_integrator()
    if not integrator:
        return jsonify({
            'status': 'error',
            'error': 'Training integrator not available',
            'healthy': False
        }), 503
    
    try:
        # Check coherence for degradation
        coherence_trend = integrator.coherence_evaluator.get_coherence_trend(window=50)
        degradation_detected = coherence_trend.get('degradation_detected', False)
        
        # Get progress stats
        progress_summary = integrator.progress_tracker.get_summary()
        
        health = {
            'status': 'healthy' if not degradation_detected else 'degraded',
            'healthy': not degradation_detected,
            'training_active': integrator._training_active,
            'components': {
                'orchestrator': integrator.orchestrator is not None,
                'progress_tracker': True,
                'coherence_evaluator': True,
                'curiosity_engine': integrator.curiosity_engine is not None,
                'shadow_loop': integrator.shadow_loop is not None,
                'feedback_system': integrator.feedback_system is not None,
            },
            'metrics': {
                'total_training_steps': progress_summary['global']['train_steps_completed'],
                'unique_topics': progress_summary['global']['unique_topics_seen'],
                'curriculum_progress': progress_summary['global']['curriculum_progress_index'],
                'avg_coherence': coherence_trend.get('avg_coherence', 0),
                'degradation_detected': degradation_detected,
            }
        }
        
        return jsonify({
            'status': 'success',
            'data': health
        })
    except Exception as e:
        logger.error(f"Error getting training health: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'healthy': False
        }), 500


__all__ = ['training_monitor_bp']
