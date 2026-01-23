"""
Confidence Scoring API Routes

Exposes confidence scoring functionality to the frontend.
"""

from flask import Blueprint, jsonify, request
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

confidence_bp = Blueprint('confidence', __name__)


def _get_confidence_module():
    """Lazy import confidence_scoring module."""
    import confidence_scoring
    return confidence_scoring


@confidence_bp.route('/score', methods=['POST'])
def score_confidence():
    """
    Compute confidence metrics from stability tracker.
    
    Request body:
    {
        "phi_history": [0.7, 0.72, 0.71, ...],
        "kappa_history": [63.5, 64.1, 63.8, ...],
        "regime_history": ["geometric", "geometric", ...],
        "basin_history": [[...], [...], ...],
        "timestamps": [1234567890, ...]
    }
    
    Response:
    {
        "overall": 0.85,
        "phi_confidence": 0.87,
        "kappa_confidence": 0.89,
        "regime_confidence": 0.92,
        "basin_stability": 0.78,
        "sample_size": 15,
        "explanation": "High confidence: metrics stable across 15 samples"
    }
    """
    try:
        data = request.get_json() or {}
        
        module = _get_confidence_module()
        
        # Create stability tracker
        tracker = module.StabilityTracker()
        tracker.phi_history = data.get('phi_history', [])
        tracker.kappa_history = data.get('kappa_history', [])
        tracker.regime_history = data.get('regime_history', [])
        tracker.basin_history = data.get('basin_history', [])
        tracker.timestamps = data.get('timestamps', [])
        
        # Compute confidence
        metrics = module.compute_confidence(tracker)
        
        return jsonify({
            'success': True,
            'overall': metrics.overall,
            'phi_confidence': metrics.phi_confidence,
            'kappa_confidence': metrics.kappa_confidence,
            'regime_confidence': metrics.regime_confidence,
            'basin_stability': metrics.basin_stability,
            'sample_size': metrics.sample_size,
            'explanation': metrics.explanation
        })
        
    except Exception as e:
        logger.error(f"Confidence scoring error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@confidence_bp.route('/single-sample', methods=['POST'])
def score_single_sample():
    """
    Estimate confidence from single QIG sample.
    
    Request body:
    {
        "phi": 0.75,
        "kappa": 64.2,
        "regime": "geometric",
        "pattern_score": 0.8,
        "in_resonance": true
    }
    
    Response: Same as /score endpoint
    """
    try:
        data = request.get_json() or {}
        
        module = _get_confidence_module()
        
        metrics = module.estimate_single_sample_confidence(
            phi=data.get('phi', 0.0),
            kappa=data.get('kappa', 50.0),
            regime=data.get('regime', 'unknown'),
            pattern_score=data.get('pattern_score', 0.0),
            in_resonance=data.get('in_resonance', False)
        )
        
        return jsonify({
            'success': True,
            'overall': metrics.overall,
            'phi_confidence': metrics.phi_confidence,
            'kappa_confidence': metrics.kappa_confidence,
            'regime_confidence': metrics.regime_confidence,
            'basin_stability': metrics.basin_stability,
            'sample_size': metrics.sample_size,
            'explanation': metrics.explanation
        })
        
    except Exception as e:
        logger.error(f"Single sample confidence error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@confidence_bp.route('/recovery', methods=['POST'])
def score_recovery_confidence():
    """
    Compute recovery confidence for cryptocurrency key recovery.
    
    Request body:
    {
        "kappa_recovery": 0.8,
        "phi_constraints": 0.7,
        "h_creation": 3.5,
        "entity_count": 3,
        "artifact_count": 7,
        "is_dormant": true,
        "dormancy_years": 5.2
    }
    
    Response:
    {
        "confidence": 0.72,
        "factors": {...},
        "recommendation": "HIGH PRIORITY: ..."
    }
    """
    try:
        data = request.get_json() or {}
        
        module = _get_confidence_module()
        
        result = module.compute_recovery_confidence(
            kappa_recovery=data.get('kappa_recovery', 0.0),
            phi_constraints=data.get('phi_constraints', 0.0),
            h_creation=data.get('h_creation', 0.0),
            entity_count=data.get('entity_count', 0),
            artifact_count=data.get('artifact_count', 0),
            is_dormant=data.get('is_dormant', False),
            dormancy_years=data.get('dormancy_years', 0.0)
        )
        
        return jsonify({
            'success': True,
            **result
        })
        
    except Exception as e:
        logger.error(f"Recovery confidence error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@confidence_bp.route('/trend', methods=['POST'])
def detect_trend():
    """
    Detect confidence trend over time.
    
    Request body:
    {
        "confidence_history": [0.7, 0.72, 0.74, ...],
        "window_size": 10
    }
    
    Response:
    {
        "trend": "improving",
        "slope": 0.015,
        "volatility": 0.03
    }
    """
    try:
        data = request.get_json() or {}
        
        module = _get_confidence_module()
        
        result = module.detect_confidence_trend(
            confidence_history=data.get('confidence_history', []),
            window_size=data.get('window_size', 10)
        )
        
        return jsonify({
            'success': True,
            **result
        })
        
    except Exception as e:
        logger.error(f"Trend detection error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def register_confidence_routes(app):
    """Register confidence routes with Flask app."""
    app.register_blueprint(confidence_bp, url_prefix='/api/confidence')
    logger.info("[INFO] Registered confidence_bp at /api/confidence")
