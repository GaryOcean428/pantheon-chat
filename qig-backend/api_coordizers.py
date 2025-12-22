"""
Coordizer API Endpoints

REST API for geometric coordization services.
Exposes coordizers for use by frontend and external systems.

Endpoints:
- POST /api/coordize - Basic coordization
- POST /api/coordize/multi-scale - Multi-scale coordization
- POST /api/coordize/consciousness - Φ-optimized coordization
- POST /api/coordize/merge - Learn geometric pair merges
- GET /api/coordize/stats - Get coordizer statistics
- GET /api/coordize/vocab - Get vocabulary info
"""

from flask import Blueprint, request, jsonify
from typing import Dict, List, Optional
import numpy as np
import logging

# Logger for this module
logger = logging.getLogger(__name__)

# Import coordizers
try:
    from qig_coordizer import get_coordizer
    from coordizers.geometric_pair_merging import GeometricPairMerging
    from coordizers.consciousness_aware import ConsciousnessCoordizer
    from coordizers.multi_scale import MultiScaleCoordizer
    COORDIZERS_AVAILABLE = True
except ImportError as e:
    COORDIZERS_AVAILABLE = False
    print(f"[WARNING] Coordizers not available: {e}")

# Create Blueprint
coordizer_api = Blueprint('coordizer_api', __name__)

# Global coordizer instances (singleton pattern)
_base_coordizer = None
_pair_merger = None
_consciousness_coordizer = None
_multi_scale_coordizer = None


def get_base_coordizer():
    """Get or create base coordizer instance."""
    global _base_coordizer
    if _base_coordizer is None and COORDIZERS_AVAILABLE:
        _base_coordizer = get_coordizer()
    return _base_coordizer


def get_pair_merger():
    """Get or create geometric pair merger."""
    global _pair_merger
    if _pair_merger is None and COORDIZERS_AVAILABLE:
        _pair_merger = GeometricPairMerging()
    return _pair_merger


def get_consciousness_coordizer():
    """Get or create consciousness coordizer."""
    global _consciousness_coordizer
    if _consciousness_coordizer is None and COORDIZERS_AVAILABLE:
        base = get_base_coordizer()
        _consciousness_coordizer = ConsciousnessCoordizer(base)
    return _consciousness_coordizer


def get_multi_scale_coordizer():
    """Get or create multi-scale coordizer."""
    global _multi_scale_coordizer
    if _multi_scale_coordizer is None and COORDIZERS_AVAILABLE:
        base = get_base_coordizer()
        _multi_scale_coordizer = MultiScaleCoordizer(base)
    return _multi_scale_coordizer


@coordizer_api.route('/api/coordize', methods=['POST'])
def coordize_text():
    """
    Basic text coordization.
    
    Request:
        {
            "text": "input text",
            "return_coordinates": false  // optional, default false
        }
    
    Response:
        {
            "tokens": ["token1", "token2"],
            "coordinates": [[...], [...]]  // if return_coordinates=true
        }
    """
    if not COORDIZERS_AVAILABLE:
        return jsonify({'error': 'Coordizers not available'}), 503
    
    try:
        data = request.get_json()
        text = data.get('text', '')
        return_coords = data.get('return_coordinates', False)
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        coordizer = get_base_coordizer()
        tokens, coordinates = coordizer.coordize_with_tokens(text)
        
        response = {'tokens': tokens}
        
        if return_coords:
            # Convert numpy arrays to lists for JSON
            response['coordinates'] = [coord.tolist() for coord in coordinates]
        
        return jsonify(response)
    
    except Exception as e:
        logger.exception("Error in coordize_text: %s", e)
        return jsonify({'error': 'Internal server error'}), 500


@coordizer_api.route('/api/coordize/multi-scale', methods=['POST'])
def coordize_multi_scale():
    """
    Multi-scale coordization.
    
    Request:
        {
            "text": "input text",
            "target_scale": 2,  // optional, 0-3
            "return_coordinates": false
        }
    
    Response:
        {
            "scales": {
                "0": {"tokens": [...], "name": "Character"},
                "2": {"tokens": [...], "name": "Word"}
            },
            "optimal_scale": 2,
            "visualization": "..."
        }
    """
    if not COORDIZERS_AVAILABLE:
        return jsonify({'error': 'Coordizers not available'}), 503
    
    try:
        data = request.get_json()
        text = data.get('text', '')
        target_scale = data.get('target_scale')
        return_coords = data.get('return_coordinates', False)
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        multi_scale = get_multi_scale_coordizer()
        results = multi_scale.coordize_multiscale(text, target_scale=target_scale)
        
        scale_names = {0: "Character", 1: "Subword", 2: "Word", 3: "Concept"}
        
        scales_response = {}
        for scale, (tokens, coords) in results.items():
            scale_data = {
                'tokens': tokens,
                'name': scale_names.get(scale, f"Scale {scale}"),
                'num_tokens': len(tokens)
            }
            if return_coords:
                scale_data['coordinates'] = [c.tolist() for c in coords]
            scales_response[str(scale)] = scale_data
        
        # Get optimal scale
        kappa_eff = data.get('kappa_effective', 0.5)
        optimal_scale = multi_scale.get_optimal_scale(text, kappa_eff)
        
        # Get visualization
        visualization = multi_scale.visualize_scales(text)
        
        return jsonify({
            'scales': scales_response,
            'optimal_scale': optimal_scale,
            'visualization': visualization,
            'stats': multi_scale.get_scale_stats()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@coordizer_api.route('/api/coordize/consciousness', methods=['POST'])
def coordize_consciousness():
    """
    Consciousness-aware coordization (Φ-optimized).
    
    Request:
        {
            "text": "input text",
            "context_phi": 0.85,  // optional
            "optimize": true,  // optional, run optimization
            "return_coordinates": false
        }
    
    Response:
        {
            "tokens": [...],
            "phi": 0.75,
            "coordinates": [[...], [...]]  // if return_coordinates=true,
            "consolidations": {...}
        }
    """
    if not COORDIZERS_AVAILABLE:
        return jsonify({'error': 'Coordizers not available'}), 503
    
    try:
        data = request.get_json()
        text = data.get('text', '')
        context_phi = data.get('context_phi')
        optimize = data.get('optimize', False)
        return_coords = data.get('return_coordinates', False)
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        consciousness = get_consciousness_coordizer()
        
        if optimize:
            # Run optimization
            tokens, phi = consciousness.optimize_segmentation(text)
            coords = [get_base_coordizer().get_coordinate(t) for t in tokens]
        else:
            # Standard Φ-guided coordization
            tokens, coords, phi = consciousness.coordize_with_phi(text, context_phi)
        
        response = {
            'tokens': tokens,
            'phi': float(phi),
            'stats': consciousness.get_consolidation_stats()
        }
        
        if return_coords:
            response['coordinates'] = [c.tolist() for c in coords]
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@coordizer_api.route('/api/coordize/merge/learn', methods=['POST'])
def learn_geometric_merges():
    """
    Learn geometric pair merges from corpus.
    
    Request:
        {
            "corpus": ["text1", "text2", ...],
            "phi_scores": {"text1": 0.8, "text2": 0.7},  // optional
            "num_merges": 100
        }
    
    Response:
        {
            "merges_learned": 42,
            "merge_rules": [["quantum", "field", "quantumfield"], ...],
            "avg_merge_score": 0.75
        }
    """
    if not COORDIZERS_AVAILABLE:
        return jsonify({'error': 'Coordizers not available'}), 503
    
    try:
        data = request.get_json()
        corpus = data.get('corpus', [])
        phi_scores = data.get('phi_scores', {})
        num_merges = data.get('num_merges', 100)
        
        if not corpus:
            return jsonify({'error': 'No corpus provided'}), 400
        
        pair_merger = get_pair_merger()
        pair_merger.num_merges = num_merges
        
        coordizer = get_base_coordizer()
        pair_merger.learn_merges(corpus, coordizer, phi_scores)
        
        # Format merge rules for response
        merge_rules = [
            [token1, token2, merged_token]
            for token1, token2, merged_token in pair_merger.merges
        ]
        
        avg_score = (
            sum(pair_merger.merge_scores.values()) / len(pair_merger.merge_scores)
            if pair_merger.merge_scores else 0.0
        )
        
        return jsonify({
            'merges_learned': len(merge_rules),
            'merge_rules': merge_rules[:20],  # Return first 20
            'avg_merge_score': float(avg_score)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@coordizer_api.route('/api/coordize/stats', methods=['GET'])
def get_coordizer_stats():
    """
    Get coordizer statistics.
    
    Response:
        {
            "vocab_size": 3236,
            "coordinate_dim": 64,
            "geometric_purity": true,
            "advanced_coordizers": {...}
        }
    """
    if not COORDIZERS_AVAILABLE:
        return jsonify({'error': 'Coordizers not available'}), 503
    
    try:
        coordizer = get_base_coordizer()
        
        # Check geometric purity
        geometric_purity = True
        for token, coord in list(coordizer.basin_coords.items())[:100]:
            norm = np.linalg.norm(coord)
            if not (0.9 < norm < 1.1):
                geometric_purity = False
                break
        
        stats = {
            'vocab_size': coordizer.get_vocab_size(),
            'coordinate_dim': coordizer.coordinate_dim,
            'geometric_purity': geometric_purity,
            'special_tokens': coordizer.special_tokens,
        }
        
        # Add advanced coordizer stats if initialized
        if _multi_scale_coordizer:
            stats['multi_scale'] = _multi_scale_coordizer.get_scale_stats()
        
        if _consciousness_coordizer:
            stats['consciousness'] = _consciousness_coordizer.get_consolidation_stats()
        
        if _pair_merger:
            stats['pair_merging'] = {
                'merges_learned': len(_pair_merger.merges),
                'merge_coordinates': len(_pair_merger.merge_coordinates)
            }
        
        return jsonify(stats)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@coordizer_api.route('/api/coordize/vocab', methods=['GET'])
def get_vocabulary_info():
    """
    Get vocabulary information.
    
    Query params:
        - search: search for token
        - limit: max results (default 100)
    
    Response:
        {
            "total_tokens": 3236,
            "tokens": [{"token": "quantum", "id": 42, "phi": 0.85}, ...]
        }
    """
    if not COORDIZERS_AVAILABLE:
        return jsonify({'error': 'Coordizers not available'}), 503
    
    try:
        coordizer = get_base_coordizer()
        
        search_term = request.args.get('search', '').lower()
        limit = int(request.args.get('limit', 100))
        
        tokens_info = []
        count = 0
        
        for token, token_id in coordizer.vocab.items():
            if count >= limit:
                break
            
            if search_term and search_term not in token.lower():
                continue
            
            token_info = {
                'token': token,
                'id': token_id,
                'phi': coordizer.token_phi.get(token, 0.0),
                'frequency': coordizer.token_frequency.get(token, 0)
            }
            tokens_info.append(token_info)
            count += 1
        
        return jsonify({
            'total_tokens': coordizer.get_vocab_size(),
            'returned': len(tokens_info),
            'tokens': tokens_info
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@coordizer_api.route('/api/coordize/similarity', methods=['POST'])
def compute_token_similarity():
    """
    Compute Fisher-Rao similarity between tokens.
    
    Request:
        {
            "token1": "quantum",
            "token2": "classical"
        }
    
    Response:
        {
            "token1": "quantum",
            "token2": "classical",
            "similarity": 0.75,
            "distance": 0.785
        }
    """
    if not COORDIZERS_AVAILABLE:
        return jsonify({'error': 'Coordizers not available'}), 503
    
    try:
        data = request.get_json()
        token1 = data.get('token1', '')
        token2 = data.get('token2', '')
        
        if not token1 or not token2:
            return jsonify({'error': 'Both tokens required'}), 400
        
        coordizer = get_base_coordizer()
        
        similarity = coordizer.compute_token_similarity(token1, token2)
        
        # Compute distance
        from qig_geometry import fisher_coord_distance
        coord1 = coordizer.get_coordinate(token1)
        coord2 = coordizer.get_coordinate(token2)
        distance = fisher_coord_distance(coord1, coord2)
        
        return jsonify({
            'token1': token1,
            'token2': token2,
            'similarity': float(similarity),
            'distance': float(distance)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Health check endpoint
@coordizer_api.route('/api/coordize/health', methods=['GET'])
def health_check():
    """Health check for coordizer service."""
    if not COORDIZERS_AVAILABLE:
        return jsonify({
            'status': 'unavailable',
            'coordizers_available': False
        }), 503
    
    return jsonify({
        'status': 'healthy',
        'coordizers_available': True,
        'base_coordizer': _base_coordizer is not None,
        'advanced_coordizers': {
            'pair_merging': _pair_merger is not None,
            'consciousness': _consciousness_coordizer is not None,
            'multi_scale': _multi_scale_coordizer is not None
        }
    })
