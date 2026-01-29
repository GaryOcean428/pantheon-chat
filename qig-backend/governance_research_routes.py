#!/usr/bin/env python3
"""
Flask API Routes for Governance and Research Modules
====================================================

Provides monitoring and testing endpoints for wired modules.

Authority: E8 Protocol v4.0
Status: ACTIVE
Created: 2026-01-23
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flask import Flask

try:
    from flask import jsonify, request
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

logger = logging.getLogger(__name__)


def register_governance_research_routes(app: Flask):
    """
    Register API routes for governance and research modules.
    
    Args:
        app: Flask application instance
    """
    if not FLASK_AVAILABLE:
        logger.warning("[GovernanceRoutes] Flask not available, skipping route registration")
        return
    
    from governance_research_wiring import (
        PANTHEON_GOVERNANCE_AVAILABLE,
        GOD_DEBATES_ETHICAL_AVAILABLE,
        SLEEP_PACKET_ETHICAL_AVAILABLE,
        GEOMETRIC_DEEP_RESEARCH_AVAILABLE,
        VOCABULARY_VALIDATOR_AVAILABLE,
        _governance_integration,
        _ethical_debate_manager,
        _sleep_packet_validator,
        _deep_research_engine,
        _vocab_validator,
        get_ethical_debate_manager_instance,
        get_deep_research_engine_instance,
    )
    
    @app.route('/api/governance/status', methods=['GET'])
    def governance_status():
        """Get status of all governance and research modules."""
        return jsonify({
            'modules': {
                'pantheon_governance': PANTHEON_GOVERNANCE_AVAILABLE,
                'god_debates_ethical': GOD_DEBATES_ETHICAL_AVAILABLE,
                'sleep_packet_ethical': SLEEP_PACKET_ETHICAL_AVAILABLE,
                'geometric_deep_research': GEOMETRIC_DEEP_RESEARCH_AVAILABLE,
                'vocabulary_validator': VOCABULARY_VALIDATOR_AVAILABLE,
            },
            'singletons': {
                'governance_integration': _governance_integration is not None,
                'ethical_debate_manager': _ethical_debate_manager is not None,
                'sleep_packet_validator': _sleep_packet_validator is not None,
                'deep_research_engine': _deep_research_engine is not None,
                'vocab_validator': _vocab_validator is not None,
            }
        })
    
    @app.route('/api/governance/validate_kernel', methods=['POST'])
    def validate_kernel_endpoint():
        """Validate a kernel name against registry rules."""
        if not PANTHEON_GOVERNANCE_AVAILABLE:
            return jsonify({'error': 'Governance not available'}), 503
        
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json(silent=True) or {}
        name = data.get('name')
        
        if not name:
            return jsonify({'error': 'Missing name parameter'}), 400
        
        from pantheon_governance_integration import validate_kernel_name
        valid, reason = validate_kernel_name(name)
        
        return jsonify({
            'valid': valid,
            'reason': reason,
            'name': name
        })
    
    @app.route('/api/debates/ethics_report', methods=['GET'])
    def debates_ethics_report():
        """Get ethics report for all debates."""
        if not GOD_DEBATES_ETHICAL_AVAILABLE:
            return jsonify({'error': 'Ethical debates not available'}), 503
        
        manager = get_ethical_debate_manager_instance()
        if manager is None:
            return jsonify({'error': 'Manager not initialized'}), 500
        
        report = manager.get_debate_ethics_report()
        return jsonify(report)
    
    @app.route('/api/research/deep_research', methods=['POST'])
    def deep_research_endpoint():
        """
        Execute phi-driven deep research.
        
        Note: This is a synchronous blocking endpoint for MVP.
        For production, consider using async Flask views or a task queue.
        """
        if not GEOMETRIC_DEEP_RESEARCH_AVAILABLE:
            return jsonify({'error': 'Deep research not available'}), 503
        
        engine = get_deep_research_engine_instance()
        if engine is None:
            return jsonify({'error': 'Research engine not initialized'}), 500
        
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json(silent=True) or {}
        query = data.get('query')
        phi = data.get('phi', 0.5)
        kappa = data.get('kappa', 50.0)
        
        if not query:
            return jsonify({'error': 'Missing query parameter'}), 400
        
        # For MVP: synchronous execution
        # TODO: Move to async Flask view or background task queue for production
        try:
            from geometric_deep_research import ResearchTelemetry
            import numpy as np
            import asyncio
            
            telemetry = ResearchTelemetry(
                phi=phi,
                kappa_eff=kappa,
                regime='normal',
                surprise=0.5
            )
            
            # Check if event loop is already running
            try:
                loop = asyncio.get_running_loop()
                # Already in an event loop - cannot use asyncio.run()
                return jsonify({
                    'error': 'Deep research requires async context',
                    'suggestion': 'Use async Flask view or background task queue'
                }), 501
            except RuntimeError:
                # No event loop running, safe to use asyncio.run()
                result = asyncio.run(engine.deep_research(query, telemetry))
                
                return jsonify({
                    'query': result.query,
                    'depth': result.depth,
                    'sources_count': len(result.sources),
                    'integration_level': result.integration_level,
                    'timestamp': result.timestamp.isoformat()
                })
        except Exception as e:
            logger.error(f"[GovernanceRoutes] Deep research failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    logger.info("[GovernanceRoutes] Governance/Research API routes registered")
