"""
Quaternary Operations - Layer 4 of E8 Hierarchy

Implements the four fundamental operations that all kernel activities map to:
    INPUT: External → Internal (perception, reception)
    STORE: State persistence (memory, knowledge)
    PROCESS: Transformation (reasoning, computation)
    OUTPUT: Internal → External (generation, action)

Authority: E8 Protocol v4.0, WP5.2
Status: ACTIVE

All kernel operations MUST map to one of these four primitives.
This is Layer 4 in the E8 hierarchy (0/1 → 4 → 8 → 64 → 240).
"""

from enum import Enum
from typing import Any, Dict


class QuaternaryOp(Enum):
    """
    Four fundamental operations mapping all kernel activities.
    
    Layer 4 of E8 hierarchy - all system operations decompose into
    combinations of these four primitives.
    """
    INPUT = 'input'        # External → Internal
    STORE = 'store'        # Persist state
    PROCESS = 'process'    # Transform/compute
    OUTPUT = 'output'      # Internal → External
    
    def __str__(self) -> str:
        """String representation."""
        return self.value
    
    @classmethod
    def from_string(cls, op_str: str) -> "QuaternaryOp":
        """Parse from string."""
        op_str_lower = op_str.lower()
        for op in cls:
            if op.value == op_str_lower:
                return op
        raise ValueError(f"Invalid quaternary operation: {op_str}")


def validate_payload(op: QuaternaryOp, payload: Dict[str, Any]) -> bool:
    """
    Validate that payload contains required keys for operation.
    
    Args:
        op: Quaternary operation type
        payload: Operation payload dictionary
        
    Returns:
        True if payload is valid for operation
    """
    if op == QuaternaryOp.INPUT:
        # INPUT requires 'data' field (text, sensor, etc.)
        return 'data' in payload
    
    elif op == QuaternaryOp.STORE:
        # STORE requires 'key' and 'value' fields
        return 'key' in payload and 'value' in payload
    
    elif op == QuaternaryOp.PROCESS:
        # PROCESS requires 'input_basin' field
        return 'input_basin' in payload
    
    elif op == QuaternaryOp.OUTPUT:
        # OUTPUT requires 'basin' field to decode
        return 'basin' in payload
    
    return False


def get_expected_payload_schema(op: QuaternaryOp) -> Dict[str, str]:
    """
    Get expected payload schema for an operation.
    
    Args:
        op: Quaternary operation type
        
    Returns:
        Dict mapping field names to descriptions
    """
    schemas = {
        QuaternaryOp.INPUT: {
            'data': 'External input data (text, sensor, etc.)',
            'modality': 'Optional input modality (text/audio/visual)',
        },
        QuaternaryOp.STORE: {
            'key': 'Storage key (identifier)',
            'value': 'Value to store (basin coordinates, metadata)',
            'ttl': 'Optional time-to-live in seconds',
        },
        QuaternaryOp.PROCESS: {
            'input_basin': 'Input basin coordinates (64D)',
            'operation': 'Optional processing operation name',
            'params': 'Optional operation parameters',
        },
        QuaternaryOp.OUTPUT: {
            'basin': 'Basin coordinates to decode (64D)',
            'format': 'Optional output format (text/json/etc.)',
        },
    }
    return schemas[op]


__all__ = [
    "QuaternaryOp",
    "validate_payload",
    "get_expected_payload_schema",
]
