"""
Buffer of Thoughts API Routes

Endpoints:
- GET /api/buffer-of-thoughts/stats - Get buffer statistics
- GET /api/buffer-of-thoughts/templates - List all templates
- GET /api/buffer-of-thoughts/template/<id> - Get single template
- POST /api/buffer-of-thoughts/retrieve - Find matching templates
- POST /api/buffer-of-thoughts/instantiate - Instantiate a template
- POST /api/buffer-of-thoughts/learn - Learn new template from trace
- POST /api/buffer-of-thoughts/record-usage - Record usage outcome
- POST /api/buffer-of-thoughts/evolve - Trigger template evolution

Author: Ocean/Zeus Pantheon
"""

from flask import Blueprint, request, jsonify
import traceback

buffer_of_thoughts_bp = Blueprint('buffer_of_thoughts', __name__, url_prefix='/api/buffer-of-thoughts')


def get_buffer():
    """Get the MetaBuffer instance."""
    from buffer_of_thoughts import get_meta_buffer
    return get_meta_buffer()


@buffer_of_thoughts_bp.route('/stats', methods=['GET'])
def stats_endpoint():
    """
    Get buffer statistics.
    
    GET /api/buffer-of-thoughts/stats
    
    Returns:
    {
        "success": true,
        "total_templates": 8,
        "by_category": {"decomposition": 1, "synthesis": 1, ...},
        "total_usage": 100,
        "avg_success_rate": 0.75
    }
    """
    try:
        buffer = get_buffer()
        stats = buffer.get_stats()
        
        return jsonify({
            'success': True,
            **stats
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@buffer_of_thoughts_bp.route('/templates', methods=['GET'])
def list_templates_endpoint():
    """
    List all templates, optionally filtered by category.
    
    GET /api/buffer-of-thoughts/templates?category=decomposition
    
    Returns:
    {
        "success": true,
        "templates": [
            {
                "template_id": "seed_decomposition_001",
                "name": "Fundamental Decomposition",
                "category": "decomposition",
                "description": "...",
                "usage_count": 10,
                "success_rate": 0.8
            }
        ],
        "count": 8
    }
    """
    try:
        from buffer_of_thoughts import TemplateCategory
        
        category_filter = request.args.get('category')
        
        buffer = get_buffer()
        
        templates = []
        
        if category_filter:
            try:
                cat = TemplateCategory(category_filter)
                cat_templates = buffer._templates.get(cat, [])
                templates = cat_templates
            except ValueError:
                return jsonify({
                    'success': False,
                    'error': f'Invalid category: {category_filter}',
                    'valid_categories': [c.value for c in TemplateCategory]
                }), 400
        else:
            templates = list(buffer._template_index.values())
        
        formatted = []
        for template in templates:
            formatted.append({
                'template_id': template.template_id,
                'name': template.name,
                'category': template.category.value,
                'description': template.description,
                'usage_count': template.usage_count,
                'success_count': template.success_count,
                'success_rate': template.success_rate,
                'avg_efficiency': template.avg_efficiency,
                'trajectory_length': template.trajectory_length,
                'abstraction_level': template.abstraction_level,
                'source': template.source,
                'created_at': template.created_at,
                'last_used': template.last_used
            })
        
        return jsonify({
            'success': True,
            'templates': formatted,
            'count': len(formatted)
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@buffer_of_thoughts_bp.route('/template/<template_id>', methods=['GET'])
def get_template_endpoint(template_id: str):
    """
    Get a single template by ID.
    
    GET /api/buffer-of-thoughts/template/<id>
    
    Returns:
    {
        "success": true,
        "template": {
            "template_id": "...",
            "name": "...",
            "waypoints": [...],
            ...
        }
    }
    """
    try:
        buffer = get_buffer()
        template = buffer._template_index.get(template_id)
        
        if not template:
            return jsonify({
                'success': False,
                'error': f'Template {template_id} not found'
            }), 404
        
        # Format waypoints
        waypoints = []
        for wp in template.waypoints:
            waypoints.append({
                'semantic_role': wp.semantic_role,
                'curvature': wp.curvature,
                'is_critical': wp.is_critical,
                'typical_duration': wp.typical_duration,
                'notes': wp.notes,
                'basin_coords': wp.basin_coords[:10]  # Truncate for readability
            })
        
        return jsonify({
            'success': True,
            'template': {
                'template_id': template.template_id,
                'name': template.name,
                'category': template.category.value,
                'description': template.description,
                'waypoints': waypoints,
                'usage_count': template.usage_count,
                'success_count': template.success_count,
                'success_rate': template.success_rate,
                'avg_efficiency': template.avg_efficiency,
                'total_curvature': template.total_curvature,
                'abstraction_level': template.abstraction_level,
                'source': template.source,
                'created_at': template.created_at,
                'last_used': template.last_used
            }
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@buffer_of_thoughts_bp.route('/retrieve', methods=['POST'])
def retrieve_endpoint():
    """
    Find templates matching a problem.
    
    POST /api/buffer-of-thoughts/retrieve
    {
        "problem_basin": [...],
        "category": "decomposition",  // optional
        "max_results": 5,
        "min_success_rate": 0.5
    }
    
    Returns:
    {
        "success": true,
        "templates": [
            {
                "template_id": "...",
                "name": "...",
                "similarity": 0.85,
                ...
            }
        ]
    }
    """
    try:
        from buffer_of_thoughts import TemplateCategory
        
        data = request.get_json() or {}
        
        problem_basin = data.get('problem_basin')
        if not problem_basin:
            return jsonify({'error': 'problem_basin required'}), 400
        
        category = None
        if data.get('category'):
            try:
                category = TemplateCategory(data['category'])
            except ValueError:
                return jsonify({
                    'success': False,
                    'error': f'Invalid category: {data["category"]}'
                }), 400
        
        max_results = data.get('max_results', 5)
        min_success_rate = data.get('min_success_rate', 0.0)
        
        buffer = get_buffer()
        results = buffer.retrieve(
            problem_basin=problem_basin,
            category=category,
            max_results=max_results,
            min_success_rate=min_success_rate
        )
        
        formatted = []
        for template, similarity in results:
            formatted.append({
                'template_id': template.template_id,
                'name': template.name,
                'category': template.category.value,
                'description': template.description,
                'similarity': similarity,
                'success_rate': template.success_rate,
                'usage_count': template.usage_count,
                'trajectory_length': template.trajectory_length
            })
        
        return jsonify({
            'success': True,
            'templates': formatted,
            'count': len(formatted)
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@buffer_of_thoughts_bp.route('/instantiate', methods=['POST'])
def instantiate_endpoint():
    """
    Instantiate a template for a specific problem.
    
    POST /api/buffer-of-thoughts/instantiate
    {
        "template_id": "seed_decomposition_001",
        "problem_start": [...],
        "problem_goal": [...]
    }
    
    Returns:
    {
        "success": true,
        "instantiated": {
            "template_id": "...",
            "trajectory": [[...], [...], ...],
            "transformation_quality": 0.92
        }
    }
    """
    try:
        data = request.get_json() or {}
        
        template_id = data.get('template_id')
        problem_start = data.get('problem_start')
        problem_goal = data.get('problem_goal')
        
        if not template_id:
            return jsonify({'error': 'template_id required'}), 400
        if not problem_start:
            return jsonify({'error': 'problem_start required'}), 400
        if not problem_goal:
            return jsonify({'error': 'problem_goal required'}), 400
        
        buffer = get_buffer()
        template = buffer._template_index.get(template_id)
        
        if not template:
            return jsonify({
                'success': False,
                'error': f'Template {template_id} not found'
            }), 404
        
        instantiated = buffer.instantiate(
            template=template,
            problem_start=problem_start,
            problem_goal=problem_goal
        )
        
        return jsonify({
            'success': True,
            'instantiated': {
                'template_id': template.template_id,
                'template_name': template.name,
                'trajectory': instantiated.to_trajectory(),
                'transformation_quality': instantiated.transformation_quality,
                'num_waypoints': len(instantiated.transformed_waypoints)
            }
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@buffer_of_thoughts_bp.route('/learn', methods=['POST'])
def learn_endpoint():
    """
    Learn a new template from a reasoning trace.
    
    POST /api/buffer-of-thoughts/learn
    {
        "reasoning_trace": [[...], [...], ...],
        "category": "decomposition",
        "name": "My Custom Decomposition",
        "description": "A specialized decomposition for...",
        "success": true,
        "efficiency": 0.85
    }
    
    Returns:
    {
        "success": true,
        "template_id": "abc123...",
        "message": "Template learned successfully"
    }
    """
    try:
        from buffer_of_thoughts import TemplateCategory
        
        data = request.get_json() or {}
        
        reasoning_trace = data.get('reasoning_trace')
        category_str = data.get('category')
        name = data.get('name')
        description = data.get('description')
        success = data.get('success', True)
        efficiency = data.get('efficiency', 0.5)
        
        if not reasoning_trace:
            return jsonify({'error': 'reasoning_trace required'}), 400
        if not category_str:
            return jsonify({'error': 'category required'}), 400
        if not name:
            return jsonify({'error': 'name required'}), 400
        if not description:
            return jsonify({'error': 'description required'}), 400
        
        try:
            category = TemplateCategory(category_str)
        except ValueError:
            return jsonify({
                'success': False,
                'error': f'Invalid category: {category_str}',
                'valid_categories': [c.value for c in TemplateCategory]
            }), 400
        
        buffer = get_buffer()
        template = buffer.learn_template(
            reasoning_trace=reasoning_trace,
            category=category,
            name=name,
            description=description,
            success=success,
            efficiency=efficiency
        )
        
        if template:
            return jsonify({
                'success': True,
                'template_id': template.template_id,
                'message': f'Template "{name}" learned successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to learn template (success=False or efficiency < 0.5)'
            }), 400
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@buffer_of_thoughts_bp.route('/record-usage', methods=['POST'])
def record_usage_endpoint():
    """
    Record template usage outcome.
    
    POST /api/buffer-of-thoughts/record-usage
    {
        "template_id": "seed_decomposition_001",
        "success": true,
        "efficiency": 0.9
    }
    
    Returns:
    {
        "success": true,
        "message": "Usage recorded"
    }
    """
    try:
        data = request.get_json() or {}
        
        template_id = data.get('template_id')
        success = data.get('success')
        efficiency = data.get('efficiency')
        
        if not template_id:
            return jsonify({'error': 'template_id required'}), 400
        if success is None:
            return jsonify({'error': 'success required'}), 400
        if efficiency is None:
            return jsonify({'error': 'efficiency required'}), 400
        
        buffer = get_buffer()
        buffer.record_usage(
            template_id=template_id,
            success=success,
            efficiency=efficiency
        )
        
        # Get updated template stats
        template = buffer._template_index.get(template_id)
        updated_stats = {}
        if template:
            updated_stats = {
                'usage_count': template.usage_count,
                'success_rate': template.success_rate,
                'avg_efficiency': template.avg_efficiency
            }
        
        return jsonify({
            'success': True,
            'message': 'Usage recorded',
            'updated_stats': updated_stats
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@buffer_of_thoughts_bp.route('/evolve', methods=['POST'])
def evolve_endpoint():
    """
    Trigger template evolution.
    
    POST /api/buffer-of-thoughts/evolve
    
    Returns:
    {
        "success": true,
        "templates_changed": 3,
        "message": "Evolution complete"
    }
    """
    try:
        buffer = get_buffer()
        changes = buffer.evolve_templates()
        
        return jsonify({
            'success': True,
            'templates_changed': changes,
            'message': f'Evolution complete: {changes} templates changed'
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@buffer_of_thoughts_bp.route('/categories', methods=['GET'])
def categories_endpoint():
    """
    List available template categories.
    
    GET /api/buffer-of-thoughts/categories
    
    Returns:
    {
        "success": true,
        "categories": ["decomposition", "synthesis", ...]
    }
    """
    try:
        from buffer_of_thoughts import TemplateCategory
        
        return jsonify({
            'success': True,
            'categories': [c.value for c in TemplateCategory]
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


print("[BufferOfThoughtsAPI] Routes initialized at /api/buffer-of-thoughts/*")
