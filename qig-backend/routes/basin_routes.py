"""
Basin Matching API Routes

Exposes basin geometry matching and clustering functionality.
"""

from flask import Blueprint, jsonify, request
import logging

logger = logging.getLogger(__name__)

basin_bp = Blueprint('basin', __name__)


def _get_basin_module():
    """Lazy import basin_matching module."""
    import basin_matching
    return basin_matching


@basin_bp.route('/match', methods=['POST'])
def match_basins():
    """
    Find addresses with similar basin geometry.
    
    Request body:
    {
        "target": {
            "address": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
            "phi": 0.75,
            "kappa": 64.2,
            "beta": 0.44,
            "regime": "geometric",
            "pattern_score": 0.8,
            "basin_coordinates": [...]
        },
        "candidates": [
            {
                "address": "...",
                "phi": 0.73,
                ...
            },
            ...
        ],
        "top_k": 10
    }
    
    Response:
    {
        "matches": [
            {
                "candidate_address": "...",
                "target_address": "...",
                "similarity": 0.85,
                "kappa_distance": 2.3,
                "phi_distance": 0.05,
                "fisher_distance": 0.12,
                "pattern_similarity": 0.88,
                "regime_match": true,
                "confidence": 0.92,
                "explanation": "Very high geometric similarity..."
            },
            ...
        ]
    }
    """
    try:
        data = request.get_json() or {}
        
        module = _get_basin_module()
        
        # Parse target signature
        target_data = data.get('target', {})
        target_sig = module.BasinSignature(
            address=target_data.get('address', ''),
            phi=target_data.get('phi', 0.0),
            kappa=target_data.get('kappa', 50.0),
            beta=target_data.get('beta', 0.0),
            regime=target_data.get('regime', 'unknown'),
            pattern_score=target_data.get('pattern_score', 0.0),
            basin_coordinates=target_data.get('basin_coordinates', []),
            fisher_trace=target_data.get('fisher_trace', 0.0),
            ricci_scalar=target_data.get('ricci_scalar', 0.0)
        )
        
        # Parse candidate signatures
        candidates_data = data.get('candidates', [])
        candidate_sigs = []
        for cand in candidates_data:
            candidate_sigs.append(module.BasinSignature(
                address=cand.get('address', ''),
                phi=cand.get('phi', 0.0),
                kappa=cand.get('kappa', 50.0),
                beta=cand.get('beta', 0.0),
                regime=cand.get('regime', 'unknown'),
                pattern_score=cand.get('pattern_score', 0.0),
                basin_coordinates=cand.get('basin_coordinates', []),
                fisher_trace=cand.get('fisher_trace', 0.0),
                ricci_scalar=cand.get('ricci_scalar', 0.0)
            ))
        
        top_k = data.get('top_k', 10)
        
        # Find similar basins
        matches = module.find_similar_basins(target_sig, candidate_sigs, top_k)
        
        # Convert to JSON
        matches_json = []
        for match in matches:
            matches_json.append({
                'candidate_address': match.candidate_address,
                'target_address': match.target_address,
                'similarity': match.similarity,
                'kappa_distance': match.kappa_distance,
                'phi_distance': match.phi_distance,
                'fisher_distance': match.fisher_distance,
                'pattern_similarity': match.pattern_similarity,
                'regime_match': match.regime_match,
                'confidence': match.confidence,
                'explanation': match.explanation
            })
        
        return jsonify({
            'success': True,
            'matches': matches_json,
            'count': len(matches_json)
        })
        
    except Exception as e:
        logger.error(f"Basin matching error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@basin_bp.route('/cluster', methods=['POST'])
def cluster_basins():
    """
    Cluster addresses by basin geometry using DBSCAN-like algorithm.
    
    Request body:
    {
        "signatures": [
            {
                "address": "...",
                "phi": 0.75,
                "kappa": 64.2,
                ...
            },
            ...
        ],
        "epsilon": 0.3,
        "min_points": 2
    }
    
    Response:
    {
        "clusters": {
            "1": [
                {
                    "address": "...",
                    "phi": 0.75,
                    ...
                },
                ...
            ],
            ...
        },
        "stats": {
            "1": {
                "centroid_phi": 0.74,
                "centroid_kappa": 64.0,
                "phi_variance": 0.001,
                "kappa_variance": 2.5,
                "dominant_regime": "geometric",
                "avg_pattern_score": 0.82,
                "cohesion": 0.88
            },
            ...
        },
        "count": 3
    }
    """
    try:
        data = request.get_json() or {}
        
        module = _get_basin_module()
        
        # Parse signatures
        signatures_data = data.get('signatures', [])
        signatures = []
        for sig_data in signatures_data:
            signatures.append(module.BasinSignature(
                address=sig_data.get('address', ''),
                phi=sig_data.get('phi', 0.0),
                kappa=sig_data.get('kappa', 50.0),
                beta=sig_data.get('beta', 0.0),
                regime=sig_data.get('regime', 'unknown'),
                pattern_score=sig_data.get('pattern_score', 0.0),
                basin_coordinates=sig_data.get('basin_coordinates', []),
                fisher_trace=sig_data.get('fisher_trace', 0.0),
                ricci_scalar=sig_data.get('ricci_scalar', 0.0)
            ))
        
        epsilon = data.get('epsilon', 0.3)
        min_points = data.get('min_points', 2)
        
        # Cluster
        clusters = module.cluster_by_basin(signatures, epsilon, min_points)
        
        # Convert to JSON and compute stats
        clusters_json = {}
        stats_json = {}
        
        for cluster_id, cluster_sigs in clusters.items():
            # Convert signatures to dict
            clusters_json[str(cluster_id)] = [
                {
                    'address': sig.address,
                    'phi': sig.phi,
                    'kappa': sig.kappa,
                    'beta': sig.beta,
                    'regime': sig.regime,
                    'pattern_score': sig.pattern_score,
                    'basin_coordinates': sig.basin_coordinates,
                    'fisher_trace': sig.fisher_trace,
                    'ricci_scalar': sig.ricci_scalar
                }
                for sig in cluster_sigs
            ]
            
            # Compute stats
            stats_json[str(cluster_id)] = module.get_cluster_stats(cluster_sigs)
        
        return jsonify({
            'success': True,
            'clusters': clusters_json,
            'stats': stats_json,
            'count': len(clusters)
        })
        
    except Exception as e:
        logger.error(f"Basin clustering error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@basin_bp.route('/similar', methods=['POST'])
def check_similarity():
    """
    Check if two basin signatures are geometrically similar.
    
    Request body:
    {
        "sig1": {...},
        "sig2": {...},
        "strict_mode": false
    }
    
    Response:
    {
        "similar": true,
        "distances": {
            "fisher_dist": 0.12,
            "kappa_dist": 0.03,
            "phi_dist": 0.05,
            "pattern_dist": 0.08,
            "total_distance": 0.07
        }
    }
    """
    try:
        data = request.get_json() or {}
        
        module = _get_basin_module()
        
        # Parse signatures
        sig1_data = data.get('sig1', {})
        sig2_data = data.get('sig2', {})
        
        sig1 = module.BasinSignature(
            address=sig1_data.get('address', ''),
            phi=sig1_data.get('phi', 0.0),
            kappa=sig1_data.get('kappa', 50.0),
            beta=sig1_data.get('beta', 0.0),
            regime=sig1_data.get('regime', 'unknown'),
            pattern_score=sig1_data.get('pattern_score', 0.0),
            basin_coordinates=sig1_data.get('basin_coordinates', []),
            fisher_trace=sig1_data.get('fisher_trace', 0.0),
            ricci_scalar=sig1_data.get('ricci_scalar', 0.0)
        )
        
        sig2 = module.BasinSignature(
            address=sig2_data.get('address', ''),
            phi=sig2_data.get('phi', 0.0),
            kappa=sig2_data.get('kappa', 50.0),
            beta=sig2_data.get('beta', 0.0),
            regime=sig2_data.get('regime', 'unknown'),
            pattern_score=sig2_data.get('pattern_score', 0.0),
            basin_coordinates=sig2_data.get('basin_coordinates', []),
            fisher_trace=sig2_data.get('fisher_trace', 0.0),
            ricci_scalar=sig2_data.get('ricci_scalar', 0.0)
        )
        
        strict_mode = data.get('strict_mode', False)
        
        # Check similarity
        similar = module.are_basins_similar(sig1, sig2, strict_mode)
        distances = module.compute_basin_distance(sig1, sig2)
        
        return jsonify({
            'success': True,
            'similar': similar,
            'distances': distances
        })
        
    except Exception as e:
        logger.error(f"Basin similarity check error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def register_basin_routes(app):
    """Register basin routes with Flask app."""
    app.register_blueprint(basin_bp, url_prefix='/api/basin')
    logger.info("[INFO] Registered basin_bp at /api/basin")
