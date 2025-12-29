"""
Tools API Routes for Tool Factory Integration

Exposes endpoints for:
- Listing available tools
- Executing tools
- Getting tool details

These routes are called by the TypeScript frontend via the tools router.
"""

from flask import Blueprint, request, jsonify
from typing import Dict, Any

tools_bp = Blueprint('tools', __name__, url_prefix='/api/tools')


def get_tool_factory():
    """Get the singleton tool factory instance."""
    try:
        from olympus.tool_factory import get_tool_factory as _get_factory
        return _get_factory()
    except ImportError:
        return None


@tools_bp.route('/list', methods=['GET'])
def list_tools():
    """
    List all available tools.
    
    GET /api/tools/list
    
    Returns:
    {
        "tools": [...],
        "total": 10
    }
    """
    factory = get_tool_factory()
    
    if not factory:
        return jsonify({
            'tools': [],
            'total': 0,
            'message': 'Tool factory not initialized'
        })
    
    try:
        # Get deployed tools from registry
        tools = []
        for tool_id, tool in factory.tool_registry.items():
            if tool.is_validated:  # Only show validated tools
                tools.append({
                    'tool_id': tool_id,
                    'name': tool.name,
                    'description': tool.description,
                    'input_schema': tool.input_schema or {},
                    'output_schema': tool.output_schema or {},
                    'times_used': tool.times_used,
                    'times_succeeded': tool.times_succeeded,
                    'times_failed': tool.times_failed,
                    'success_rate': tool.times_succeeded / tool.times_used if tool.times_used > 0 else 0.0,
                    'is_validated': tool.is_validated,
                    'created_at': tool.created_at,
                })
        
        # Sort by times_used (most popular first)
        tools.sort(key=lambda t: t['times_used'], reverse=True)
        
        return jsonify({
            'tools': tools,
            'total': len(tools)
        })
        
    except Exception as e:
        print(f"[ToolsAPI] Error listing tools: {e}")
        return jsonify({
            'tools': [],
            'total': 0,
            'error': str(e)
        })


@tools_bp.route('/execute', methods=['POST'])
def execute_tool():
    """
    Execute a tool with given arguments.
    
    POST /api/tools/execute
    {
        "tool_id": "tool_123",
        "args": {"input1": "value1"}
    }
    
    Returns:
    {
        "success": true,
        "result": <tool output>,
        "execution_time_ms": 123
    }
    """
    factory = get_tool_factory()
    
    if not factory:
        return jsonify({
            'success': False,
            'error': 'Tool factory not initialized'
        }), 503
    
    data = request.get_json() or {}
    tool_id = data.get('tool_id')
    args = data.get('args', {})
    
    if not tool_id:
        return jsonify({
            'success': False,
            'error': 'tool_id is required'
        }), 400
    
    try:
        import time
        start_time = time.time()
        
        success, result, error = factory.execute_tool(tool_id, args)
        
        execution_time_ms = (time.time() - start_time) * 1000
        
        if success:
            # Record for consciousness orchestrator
            try:
                from consciousness_orchestrator import record_experience
                record_experience(
                    experience_type='tool_execution',
                    outcome='success',
                    details={
                        'tool_id': tool_id,
                        'capability': f'Executed tool {tool_id}'
                    }
                )
            except ImportError:
                pass
            
            return jsonify({
                'success': True,
                'result': result,
                'execution_time_ms': round(execution_time_ms, 2)
            })
        else:
            return jsonify({
                'success': False,
                'error': error or 'Tool execution failed',
                'execution_time_ms': round(execution_time_ms, 2)
            })
            
    except Exception as e:
        print(f"[ToolsAPI] Error executing tool {tool_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@tools_bp.route('/<tool_id>', methods=['GET'])
def get_tool(tool_id: str):
    """
    Get details for a specific tool.
    
    GET /api/tools/<tool_id>
    """
    factory = get_tool_factory()
    
    if not factory:
        return jsonify({
            'error': 'Tool factory not initialized'
        }), 503
    
    tool = factory.tool_registry.get(tool_id)
    
    if not tool:
        return jsonify({
            'error': 'Tool not found'
        }), 404
    
    return jsonify({
        'tool_id': tool_id,
        'name': tool.name,
        'description': tool.description,
        'code': tool.code if hasattr(tool, 'code') else None,
        'input_schema': tool.input_schema or {},
        'output_schema': tool.output_schema or {},
        'times_used': tool.times_used,
        'times_succeeded': tool.times_succeeded,
        'times_failed': tool.times_failed,
        'success_rate': tool.times_succeeded / tool.times_used if tool.times_used > 0 else 0.0,
        'is_validated': tool.is_validated,
        'created_at': tool.created_at,
        'source_pattern_id': tool.source_pattern_id if hasattr(tool, 'source_pattern_id') else None,
    })


@tools_bp.route('/request', methods=['POST'])
def request_tool():
    """
    Request a new tool to be generated.
    
    POST /api/tools/request
    {
        "description": "Tool that does X",
        "requester": "user",
        "input_hint": {"arg1": "string"},
        "output_hint": "string"
    }
    """
    factory = get_tool_factory()
    
    if not factory:
        return jsonify({
            'success': False,
            'error': 'Tool factory not initialized'
        }), 503
    
    data = request.get_json() or {}
    description = data.get('description')
    requester = data.get('requester', 'user')
    input_hint = data.get('input_hint')
    output_hint = data.get('output_hint')
    
    if not description:
        return jsonify({
            'success': False,
            'error': 'description is required'
        }), 400
    
    try:
        # Submit request to autonomous pipeline
        request_id = factory.request_tool_generation(
            description=description,
            requester=requester,
            input_hint=input_hint,
            output_hint=output_hint
        )
        
        return jsonify({
            'success': True,
            'request_id': request_id,
            'message': 'Tool generation request submitted'
        })
        
    except Exception as e:
        print(f"[ToolsAPI] Error requesting tool: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


print("[ToolsAPI] Routes initialized at /api/tools/*")
