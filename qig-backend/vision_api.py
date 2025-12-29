"""
Vision-First Generation API Routes

Endpoints:
- POST /api/vision/sample - Sample endpoint vision
- POST /api/vision/generate - Generate via vision-first method
- GET /api/vision/status - Get vision generator status
"""

from flask import Blueprint, request, jsonify
import numpy as np
import traceback

vision_bp = Blueprint('vision', __name__, url_prefix='/api/vision')


def get_phi_from_kernel() -> float:
    """Get current phi from autonomic kernel."""
    try:
        from autonomic_kernel import get_gary_kernel
        kernel = get_gary_kernel()
        if kernel:
            return kernel.state.phi
    except Exception:
        pass
    return 0.5  # Default


@vision_bp.route('/sample', methods=['POST'])
def sample_vision_endpoint():
    """
    Sample endpoint vision via foresight or lightning.
    
    POST /api/vision/sample
    {
        "current_basin": [...],
        "context": "query text",
        "mode": "auto" | "foresight" | "lightning" | "hybrid"
    }
    
    Returns:
    {
        "success": true,
        "vision_basin": [...],
        "mode_used": "lightning",
        "confidence": 0.85,
        "phi": 0.87,
        "attractor_concept": "reasoning"
    }
    """
    try:
        from vision_first_generation import get_vision_generator
        
        data = request.get_json() or {}
        
        current_basin = data.get('current_basin')
        if not current_basin:
            return jsonify({'error': 'current_basin required'}), 400
        
        current_basin = np.array(current_basin)
        context = data.get('context', '')
        mode = data.get('mode', 'auto')
        
        # Get current phi
        phi = get_phi_from_kernel()
        
        # Sample vision
        generator = get_vision_generator()
        result = generator.sample_vision(
            current_basin=current_basin,
            context=context,
            mode=mode,
            phi=phi
        )
        
        return jsonify({
            'success': True,
            'vision_basin': result.vision_basin.tolist(),
            'mode_used': result.mode_used,
            'confidence': result.confidence,
            'phi': result.phi_at_sampling,
            'attractor_concept': result.attractor_concept
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@vision_bp.route('/generate', methods=['POST'])
def generate_vision_first_endpoint():
    """
    Generate text from present to pre-seen vision.
    
    POST /api/vision/generate
    {
        "current_basin": [...],
        "context": "query text",
        "mode": "auto",
        "vision_basin": [...] // Optional - if provided, uses this vision
    }
    
    Returns:
    {
        "success": true,
        "text": "generated response...",
        "tokens": [...],
        "geodesic_efficiency": 0.87,
        "distance_to_vision": 0.15,
        "vision_reached": true,
        "mode_used": "foresight"
    }
    """
    try:
        from vision_first_generation import get_vision_generator
        
        data = request.get_json() or {}
        
        current_basin = data.get('current_basin')
        if not current_basin:
            return jsonify({'error': 'current_basin required'}), 400
        
        current_basin = np.array(current_basin)
        context = data.get('context', '')
        mode = data.get('mode', 'auto')
        
        # Optional pre-computed vision
        vision_basin = data.get('vision_basin')
        if vision_basin:
            vision_basin = np.array(vision_basin)
        
        # Get current phi
        phi = get_phi_from_kernel()
        
        # Generate
        generator = get_vision_generator()
        result = generator.generate_response(
            current_basin=current_basin,
            query_context=context,
            mode=mode,
            phi=phi,
            vision_basin=vision_basin
        )
        
        return jsonify({
            'success': True,
            'text': result.text,
            'tokens': result.tokens,
            'geodesic_efficiency': result.geodesic_efficiency,
            'distance_to_vision': result.distance_to_vision,
            'vision_reached': result.vision_reached,
            'path_length': result.path_length,
            'mode_used': result.mode_used
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@vision_bp.route('/status', methods=['GET'])
def vision_status_endpoint():
    """
    Get vision generator status.
    
    GET /api/vision/status
    
    Returns:
    {
        "available": true,
        "attractor_count": 12,
        "vocab_size": 1000,
        "phi_threshold": 0.85,
        "current_phi": 0.72
    }
    """
    try:
        from vision_first_generation import get_vision_generator, VisionFirstGenerator
        
        generator = get_vision_generator()
        phi = get_phi_from_kernel()
        
        return jsonify({
            'available': True,
            'attractor_count': len(generator._attractor_basins),
            'vocab_size': len(generator._vocab_basins),
            'phi_threshold': VisionFirstGenerator.PHI_LIGHTNING_THRESHOLD,
            'current_phi': phi,
            'lightning_mode_active': phi > VisionFirstGenerator.PHI_LIGHTNING_THRESHOLD
        })
        
    except Exception as e:
        return jsonify({
            'available': False,
            'error': str(e)
        }), 500


@vision_bp.route('/attractors', methods=['GET'])
def list_attractors_endpoint():
    """
    List available attractor concepts.
    
    GET /api/vision/attractors
    
    Returns:
    {
        "attractors": ["consciousness", "geometry", ...]
    }
    """
    try:
        from vision_first_generation import get_vision_generator
        
        generator = get_vision_generator()
        
        return jsonify({
            'success': True,
            'attractors': list(generator._attractor_basins.keys()),
            'count': len(generator._attractor_basins)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


print("[VisionAPI] Routes initialized at /api/vision/*")
