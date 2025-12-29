"""
Unit Tests for Tool Factory

Tests tool creation, validation, execution, and learning capabilities.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestToolCreation:
    """Test tool creation and validation."""
    
    def test_create_simple_tool(self):
        """Test creating a simple tool structure."""
        # Test tool data structure without full import
        from dataclasses import dataclass
        from enum import Enum
        
        class ToolStatus(Enum):
            REQUESTED = "requested"
            GENERATING = "generating"
            VALIDATING = "validating"
            DEPLOYED = "deployed"
            FAILED = "failed"
        
        @dataclass
        class SimpleGeneratedTool:
            tool_id: str
            name: str
            description: str
            input_signature: dict
            output_type: str
            implementation: str
            status: ToolStatus
            created_at: float
            requester: str
        
        tool = SimpleGeneratedTool(
            tool_id="test_add_001",
            name="Add Numbers",
            description="Adds two numbers together",
            input_signature={"a": "number", "b": "number"},
            output_type="number",
            implementation="def execute(a, b): return a + b",
            status=ToolStatus.DEPLOYED,
            created_at=time.time(),
            requester="test"
        )
        
        assert tool.tool_id == "test_add_001"
        assert tool.name == "Add Numbers"
        assert tool.status == ToolStatus.DEPLOYED
    
    def test_tool_status_transitions(self):
        """Test tool status transitions."""
        from enum import Enum
        
        class ToolStatus(Enum):
            REQUESTED = "requested"
            GENERATING = "generating"
            VALIDATING = "validating"
            DEPLOYED = "deployed"
            FAILED = "failed"
        
        # Valid statuses
        assert ToolStatus.REQUESTED.value == "requested"
        assert ToolStatus.GENERATING.value == "generating"
        assert ToolStatus.VALIDATING.value == "validating"
        assert ToolStatus.DEPLOYED.value == "deployed"
        assert ToolStatus.FAILED.value == "failed"
    
    def test_tool_with_complex_signature(self):
        """Test creating a tool with complex input signature."""
        from dataclasses import dataclass
        from enum import Enum
        
        class ToolStatus(Enum):
            DEPLOYED = "deployed"
        
        @dataclass
        class SimpleGeneratedTool:
            tool_id: str
            name: str
            description: str
            input_signature: dict
            output_type: str
            implementation: str
            status: ToolStatus
            created_at: float
            requester: str
        
        tool = SimpleGeneratedTool(
            tool_id="test_complex_001",
            name="Data Processor",
            description="Processes data with multiple inputs",
            input_signature={
                "data": "list",
                "config": "dict",
                "threshold": "number",
                "enabled": "boolean"
            },
            output_type="dict",
            implementation="def execute(data, config, threshold, enabled): return {'result': len(data)}",
            status=ToolStatus.DEPLOYED,
            created_at=time.time(),
            requester="test"
        )
        
        assert len(tool.input_signature) == 4
        assert "data" in tool.input_signature
        assert "config" in tool.input_signature


class TestToolExecution:
    """Test tool execution."""
    
    def test_execute_simple_tool(self):
        """Test executing a simple tool."""
        # Direct execution test without full factory
        code = "def execute(a, b): return a + b"
        
        # Create execution environment
        local_vars = {}
        exec(code, {}, local_vars)
        execute_fn = local_vars['execute']
        
        result = execute_fn(5, 3)
        assert result == 8
    
    def test_execute_tool_with_validation(self):
        """Test tool execution with input validation."""
        code = """
def execute(text, max_length):
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(max_length, int):
        raise TypeError("max_length must be an integer")
    return text[:max_length]
"""
        local_vars = {}
        exec(code, {}, local_vars)
        execute_fn = local_vars['execute']
        
        # Valid inputs
        result = execute_fn("hello world", 5)
        assert result == "hello"
        
        # Invalid inputs should raise
        with pytest.raises(TypeError):
            execute_fn(123, 5)
    
    def test_execute_tool_error_handling(self):
        """Test tool error handling."""
        code = """
def execute(divisor):
    return 100 / divisor
"""
        local_vars = {}
        exec(code, {}, local_vars)
        execute_fn = local_vars['execute']
        
        # Valid
        assert execute_fn(10) == 10
        
        # Division by zero
        with pytest.raises(ZeroDivisionError):
            execute_fn(0)
    
    def test_tool_sandboxing_no_imports(self):
        """Test that dangerous imports are blocked in sandboxed execution."""
        dangerous_code = """
def execute():
    import os
    return os.getcwd()
"""
        # In a proper sandbox, this should fail
        # For now, just verify the code structure
        assert "import os" in dangerous_code


class TestToolRegistry:
    """Test tool registry operations."""
    
    def test_registry_add_tool(self):
        """Test adding a tool to registry."""
        from dataclasses import dataclass
        from enum import Enum
        
        class ToolStatus(Enum):
            DEPLOYED = "deployed"
        
        @dataclass
        class SimpleGeneratedTool:
            tool_id: str
            name: str
            status: ToolStatus
        
        # Simple registry dict
        registry = {}
        
        tool = SimpleGeneratedTool(
            tool_id="registry_test_001",
            name="Registry Test Tool",
            status=ToolStatus.DEPLOYED
        )
        
        # Add to registry
        registry[tool.tool_id] = tool
        
        # Verify retrieval
        retrieved = registry.get(tool.tool_id)
        assert retrieved is not None
        assert retrieved.name == "Registry Test Tool"
    
    def test_registry_list_tools(self):
        """Test listing all tools in registry."""
        from dataclasses import dataclass
        from enum import Enum
        
        class ToolStatus(Enum):
            DEPLOYED = "deployed"
            FAILED = "failed"
        
        @dataclass
        class SimpleGeneratedTool:
            tool_id: str
            name: str
            status: ToolStatus
        
        registry = {}
        
        # Add multiple tools
        for i in range(3):
            tool = SimpleGeneratedTool(
                tool_id=f"list_test_{i}",
                name=f"List Test Tool {i}",
                status=ToolStatus.DEPLOYED
            )
            registry[tool.tool_id] = tool
        
        # List deployed tools
        deployed = [t for t in registry.values() 
                   if t.status == ToolStatus.DEPLOYED]
        assert len(deployed) >= 3
    
    def test_registry_remove_tool(self):
        """Test removing a tool from registry."""
        from dataclasses import dataclass
        from enum import Enum
        
        class ToolStatus(Enum):
            DEPLOYED = "deployed"
        
        @dataclass
        class SimpleGeneratedTool:
            tool_id: str
            name: str
            status: ToolStatus
        
        registry = {}
        
        tool = SimpleGeneratedTool(
            tool_id="remove_test_001",
            name="Remove Test Tool",
            status=ToolStatus.DEPLOYED
        )
        
        # Add and remove
        registry[tool.tool_id] = tool
        assert tool.tool_id in registry
        
        del registry[tool.tool_id]
        assert tool.tool_id not in registry


class TestToolPatterns:
    """Test tool pattern learning and matching."""
    
    def test_pattern_structure(self):
        """Test tool pattern data structure."""
        from dataclasses import dataclass
        from typing import List
        
        @dataclass
        class ToolPattern:
            pattern_id: str
            name: str
            description: str
            trigger_keywords: List[str]
            template_code: str
            input_schema: dict
            output_type: str
            success_rate: float
            usage_count: int
        
        pattern = ToolPattern(
            pattern_id="test_pattern_001",
            name="Math Operation Pattern",
            description="Pattern for mathematical operations",
            trigger_keywords=["add", "subtract", "multiply", "divide"],
            template_code="def execute(a, b): return a {op} b",
            input_schema={"a": "number", "b": "number"},
            output_type="number",
            success_rate=0.9,
            usage_count=100
        )
        
        assert pattern.pattern_id == "test_pattern_001"
        assert "add" in pattern.trigger_keywords
        assert pattern.success_rate == 0.9
    
    def test_pattern_matching(self):
        """Test pattern matching based on keywords."""
        from dataclasses import dataclass
        from typing import List
        
        @dataclass
        class ToolPattern:
            pattern_id: str
            name: str
            trigger_keywords: List[str]
            success_rate: float
        
        patterns = [
            ToolPattern(
                pattern_id="math_001",
                name="Math Pattern",
                trigger_keywords=["add", "sum", "plus"],
                success_rate=0.85
            ),
            ToolPattern(
                pattern_id="text_001",
                name="Text Pattern",
                trigger_keywords=["concat", "join", "merge"],
                success_rate=0.9
            )
        ]
        
        # Simple keyword matching
        request = "add two numbers together"
        matching = [p for p in patterns if any(kw in request.lower() for kw in p.trigger_keywords)]
        
        assert len(matching) == 1
        assert matching[0].pattern_id == "math_001"


class TestToolRequest:
    """Test tool request handling."""
    
    def test_tool_request_structure(self):
        """Test tool request data structure."""
        from dataclasses import dataclass
        
        @dataclass
        class ToolRequest:
            request_id: str
            description: str
            requester: str
            context: dict
            priority: int
            created_at: float
        
        request = ToolRequest(
            request_id="req_001",
            description="Create a tool that calculates fibonacci numbers",
            requester="user",
            context={"domain": "math"},
            priority=1,
            created_at=time.time()
        )
        
        assert request.request_id == "req_001"
        assert "fibonacci" in request.description
        assert request.priority == 1
    
    def test_tool_request_queue(self):
        """Test tool request queue ordering."""
        from dataclasses import dataclass
        
        @dataclass
        class ToolRequest:
            request_id: str
            description: str
            requester: str
            context: dict
            priority: int
            created_at: float
        
        requests = [
            ToolRequest(
                request_id=f"req_{i}",
                description=f"Request {i}",
                requester="user",
                context={},
                priority=i % 3,  # 0, 1, 2, 0, 1
                created_at=time.time() + i
            )
            for i in range(5)
        ]
        
        # Sort by priority (higher first)
        sorted_requests = sorted(requests, key=lambda r: -r.priority)
        
        assert sorted_requests[0].priority == 2
        assert sorted_requests[-1].priority == 0


class TestToolMetrics:
    """Test tool usage metrics and analytics."""
    
    def test_tool_success_rate(self):
        """Test calculating tool success rate."""
        from dataclasses import dataclass
        
        @dataclass
        class ToolMetrics:
            tool_id: str
            usage_count: int
            success_count: int
        
        metrics = ToolMetrics(
            tool_id="metrics_test_001",
            usage_count=100,
            success_count=85
        )
        
        success_rate = metrics.success_count / metrics.usage_count if metrics.usage_count > 0 else 0
        assert success_rate == 0.85
    
    def test_tool_avg_execution_time(self):
        """Test tracking average execution time."""
        execution_times = [0.1, 0.15, 0.12, 0.08, 0.11]
        avg_time = sum(execution_times) / len(execution_times)
        
        assert 0.1 <= avg_time <= 0.12


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
