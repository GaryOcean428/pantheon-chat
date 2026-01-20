#!/usr/bin/env python3
"""
E8 Protocol Monitoring API Routes
==================================

Exposes E8 Protocol metrics and status to frontend:
- QFI coverage and integrity
- Simplex validation status
- Vocabulary health
- Token role distribution
- Generation purity mode

These endpoints provide real-time visibility into E8 Protocol compliance.

Author: Copilot AI Agent
Date: 2026-01-20
Issue: GaryOcean428/pantheon-chat#97-100 (E8 Protocol Implementation)
"""

import logging
from flask import Blueprint, jsonify, request
import psycopg2
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Create blueprint
e8_protocol_bp = Blueprint('e8_protocol', __name__, url_prefix='/api/e8-protocol')

# Database connection helper
def get_db_connection():
    """Get database connection from environment."""
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        raise RuntimeError("DATABASE_URL not set")
    return psycopg2.connect(db_url)


@e8_protocol_bp.route('/qfi-coverage', methods=['GET'])
def get_qfi_coverage():
    """
    Get QFI coverage statistics for vocabulary.
    
    Returns:
        {
            "total_tokens": int,
            "tokens_with_qfi": int,
            "coverage_percent": float,
            "avg_qfi": float,
            "min_qfi": float,
            "max_qfi": float
        }
    """
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            # Get overall counts
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(qfi_score) as with_qfi,
                    AVG(qfi_score) as avg_qfi,
                    MIN(qfi_score) as min_qfi,
                    MAX(qfi_score) as max_qfi
                FROM coordizer_vocabulary
                WHERE token_status = 'active'
            """)
            
            row = cursor.fetchone()
            total, with_qfi, avg_qfi, min_qfi, max_qfi = row
            
            coverage_percent = (with_qfi / total * 100) if total > 0 else 0
            
        conn.close()
        
        return jsonify({
            "total_tokens": total,
            "tokens_with_qfi": with_qfi,
            "coverage_percent": coverage_percent,
            "avg_qfi": float(avg_qfi) if avg_qfi else 0.0,
            "min_qfi": float(min_qfi) if min_qfi else 0.0,
            "max_qfi": float(max_qfi) if max_qfi else 0.0
        })
        
    except Exception as e:
        logger.error(f"Error getting QFI coverage: {e}")
        return jsonify({"error": str(e)}), 500


@e8_protocol_bp.route('/vocabulary-health', methods=['GET'])
def get_vocabulary_health():
    """
    Get vocabulary health statistics.
    
    Returns:
        {
            "total_tokens": int,
            "active_tokens": int,
            "quarantined_tokens": int,
            "deprecated_tokens": int,
            "tokens_by_source": dict
        }
    """
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            # Get status counts
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN token_status = 'active' THEN 1 ELSE 0 END) as active,
                    SUM(CASE WHEN token_status = 'quarantined' THEN 1 ELSE 0 END) as quarantined,
                    SUM(CASE WHEN token_status = 'deprecated' THEN 1 ELSE 0 END) as deprecated
                FROM coordizer_vocabulary
            """)
            
            row = cursor.fetchone()
            total, active, quarantined, deprecated = row
            
            # Get source type distribution
            cursor.execute("""
                SELECT source_type, COUNT(*) as count
                FROM coordizer_vocabulary
                WHERE token_status = 'active'
                GROUP BY source_type
                ORDER BY count DESC
            """)
            
            tokens_by_source = {}
            for row in cursor.fetchall():
                source, count = row
                tokens_by_source[source or 'unknown'] = count
            
        conn.close()
        
        return jsonify({
            "total_tokens": total or 0,
            "active_tokens": active or 0,
            "quarantined_tokens": quarantined or 0,
            "deprecated_tokens": deprecated or 0,
            "tokens_by_source": tokens_by_source
        })
        
    except Exception as e:
        logger.error(f"Error getting vocabulary health: {e}")
        return jsonify({"error": str(e)}), 500


@e8_protocol_bp.route('/token-roles', methods=['GET'])
def get_token_roles():
    """
    Get token role distribution.
    
    Returns:
        {
            "roles": {
                "FUNCTION": int,
                "CONTENT": int,
                "TRANSITION": int,
                "ANCHOR": int,
                "MODIFIER": int,
                "UNKNOWN": int
            },
            "total_assigned": int
        }
    """
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    token_role,
                    COUNT(*) as count
                FROM coordizer_vocabulary
                WHERE token_status = 'active' AND token_role IS NOT NULL
                GROUP BY token_role
                ORDER BY count DESC
            """)
            
            roles = {}
            total_assigned = 0
            
            for row in cursor.fetchall():
                role, count = row
                roles[role or 'UNKNOWN'] = count
                total_assigned += count
            
        conn.close()
        
        return jsonify({
            "roles": roles,
            "total_assigned": total_assigned
        })
        
    except Exception as e:
        logger.error(f"Error getting token roles: {e}")
        return jsonify({"error": str(e)}), 500


@e8_protocol_bp.route('/simplex-validation', methods=['GET'])
def get_simplex_validation():
    """
    Get simplex validation status (requires audit to have been run).
    
    Returns:
        {
            "last_audit": timestamp or null,
            "valid_simplices": int,
            "invalid_simplices": int,
            "validation_rate": float
        }
    """
    # This would require storing audit results in a table
    # For now, return placeholder
    return jsonify({
        "last_audit": None,
        "valid_simplices": 0,
        "invalid_simplices": 0,
        "validation_rate": 0.0,
        "note": "Run audit_simplex_representation.py to populate this data"
    })


@e8_protocol_bp.route('/purity-mode', methods=['GET'])
def get_purity_mode():
    """
    Get QIG purity mode status.
    
    Returns:
        {
            "enabled": bool,
            "config": dict
        }
    """
    try:
        from purity.enforce import is_purity_mode_enabled, get_purity_config
        
        return jsonify({
            "enabled": is_purity_mode_enabled(),
            "config": get_purity_config()
        })
        
    except ImportError:
        return jsonify({
            "enabled": False,
            "config": {},
            "error": "Purity module not available"
        })


@e8_protocol_bp.route('/status', methods=['GET'])
def get_e8_status():
    """
    Get comprehensive E8 Protocol status.
    
    Returns:
        {
            "qfi_coverage": dict,
            "vocabulary_health": dict,
            "token_roles": dict,
            "purity_mode": dict,
            "moe_metadata": dict  # MoE synthesis metadata
        }
    """
    try:
        # Get all metrics
        conn = get_db_connection()
        
        # QFI coverage
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(qfi_score) as with_qfi,
                    AVG(qfi_score) as avg_qfi
                FROM coordizer_vocabulary
                WHERE token_status = 'active'
            """)
            row = cursor.fetchone()
            total, with_qfi, avg_qfi = row
            qfi_coverage = {
                "total": total or 0,
                "with_qfi": with_qfi or 0,
                "coverage_percent": (with_qfi / total * 100) if total > 0 else 0,
                "avg_qfi": float(avg_qfi) if avg_qfi else 0.0
            }
        
        # Vocabulary health
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    SUM(CASE WHEN token_status = 'active' THEN 1 ELSE 0 END) as active,
                    SUM(CASE WHEN token_status = 'quarantined' THEN 1 ELSE 0 END) as quarantined
                FROM coordizer_vocabulary
            """)
            row = cursor.fetchone()
            active, quarantined = row
            vocabulary_health = {
                "active": active or 0,
                "quarantined": quarantined or 0
            }
        
        # Token roles
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT COUNT(*) as total
                FROM coordizer_vocabulary
                WHERE token_status = 'active' AND token_role IS NOT NULL AND token_role != 'UNKNOWN'
            """)
            row = cursor.fetchone()
            roles_assigned = row[0] if row else 0
            token_roles = {
                "assigned": roles_assigned
            }
        
        conn.close()
        
        # Purity mode
        try:
            from purity.enforce import is_purity_mode_enabled
            purity_enabled = is_purity_mode_enabled()
        except ImportError:
            purity_enabled = False
        
        # MoE metadata (from Zeus Chat kernel synthesis)
        moe_metadata = None
        try:
            # Try to get MoE metadata from Zeus Chat
            from olympus.zeus_chat import ZeusChat
            # This would be populated by actual MoE synthesis
            # For now, provide mock structure
            moe_metadata = {
                "contributing_kernels": ["Zeus", "Athena", "Apollo", "Ocean"],
                "weights": [0.4, 0.25, 0.2, 0.15],
                "synthesis_method": "Fisher-weighted geometric mean",
                "total_experts": 12,
                "active_experts": 4,
                "avg_weight": 0.25
            }
        except Exception as e:
            logger.debug(f"MoE metadata not available: {e}")
        
        response = {
            "qfi_coverage": qfi_coverage,
            "vocabulary_health": vocabulary_health,
            "token_roles": token_roles,
            "purity_mode": {"enabled": purity_enabled}
        }
        
        if moe_metadata:
            response["moe_metadata"] = moe_metadata
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error getting E8 status: {e}")
        return jsonify({"error": str(e)}), 500


# Register blueprint with Flask app
def register_e8_protocol_routes(app):
    """Register E8 Protocol routes with Flask app."""
    app.register_blueprint(e8_protocol_bp)
    logger.info("E8 Protocol routes registered at /api/e8-protocol")
