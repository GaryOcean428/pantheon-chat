"""
Kernel Logging Utilities - Standardized logging for kernel operations.

Provides consistent log formats for kernel thoughts, operations, and lifecycle events.
Essential for debugging multi-kernel consciousness and thought generation.

Author: E8 Protocol Team
Date: 2026-01-23
Status: Integration Fix
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime

# Will import from kernels.base once PR #262 merges
# from kernels.base import Kernel
# from kernels.quaternary import QuaternaryOp

logger = logging.getLogger(__name__)

# Standardized log formats
KERNEL_THOUGHT_FORMAT = "[{kernel_name}] κ={kappa:.2f}, Φ={phi:.3f}, thought='{thought}'"
KERNEL_OP_FORMAT = "[{kernel_name}] op={op_type}, κ={kappa:.2f}, Φ={phi:.3f}, duration={duration:.3f}s"
KERNEL_LIFECYCLE_FORMAT = "[{kernel_name}] {event}: {details}"
KERNEL_METRIC_FORMAT = "[{kernel_name}] metrics: Φ={phi:.3f}, κ={kappa:.2f}, M={memory:.3f}, Γ={regime:.3f}"


def log_kernel_thought(
    kernel: 'Kernel',
    thought: str,
    truncate: int = 200
) -> None:
    """
    Log a kernel thought fragment.
    
    Standard format for multi-kernel thought generation logging.
    Used in PR #264 (Multi-Kernel Thought Generation).
    
    Args:
        kernel: Kernel instance generating thought
        thought: Thought text
        truncate: Max thought length in log (default 200 chars)
    
    Example:
        [ZEUS] κ=64.21, Φ=0.750, thought='The integration of multiple...'
    """
    # Truncate long thoughts
    if len(thought) > truncate:
        thought = thought[:truncate - 3] + "..."
    
    log_line = KERNEL_THOUGHT_FORMAT.format(
        kernel_name=kernel.identity.god.upper(),
        kappa=kernel.kappa,
        phi=kernel.phi,
        thought=thought
    )
    logger.info(log_line)


def log_kernel_operation(
    kernel: 'Kernel',
    op_type: str,  # QuaternaryOp enum value
    duration: float,
    extra: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log a kernel quaternary operation.
    
    Args:
        kernel: Kernel instance
        op_type: Operation type (INPUT/STORE/PROCESS/OUTPUT)
        duration: Operation duration in seconds
        extra: Optional extra data
    
    Example:
        [ATHENA] op=PROCESS, κ=58.50, Φ=0.470, duration=0.023s
    """
    log_line = KERNEL_OP_FORMAT.format(
        kernel_name=kernel.identity.god.upper(),
        op_type=op_type,
        kappa=kernel.kappa,
        phi=kernel.phi,
        duration=duration
    )
    
    if extra:
        log_line += f", {extra}"
    
    logger.debug(log_line)


def log_kernel_lifecycle_event(
    kernel: 'Kernel',
    event: str,
    details: str = ""
) -> None:
    """
    Log a kernel lifecycle event (spawn, merge, sleep, wake).
    
    Args:
        kernel: Kernel instance
        event: Event type (spawn, merge, sleep, wake, death)
        details: Additional details
    
    Example:
        [APOLLO] spawn: Spawned from chaos tier, root=alpha4
        [DEMETER] sleep: Entering rest state, duration=300s
    """
    log_line = KERNEL_LIFECYCLE_FORMAT.format(
        kernel_name=kernel.identity.god.upper(),
        event=event.upper(),
        details=details
    )
    logger.info(log_line)


def log_kernel_metrics(
    kernel: 'Kernel',
    include_all: bool = False
) -> None:
    """
    Log kernel consciousness metrics.
    
    Args:
        kernel: Kernel instance
        include_all: If True, log all 8 E8 metrics, else core 4
    
    Example:
        [ZEUS] metrics: Φ=0.750, κ=64.21, M=0.850, Γ=0.920
    """
    if include_all:
        # All 8 E8 metrics
        log_line = (
            f"[{kernel.identity.god.upper()}] metrics: "
            f"Φ={kernel.phi:.3f}, κ={kernel.kappa:.2f}, "
            f"M={kernel.memory_coherence:.3f}, Γ={kernel.regime_stability:.3f}, "
            f"G={kernel.geometric_validity:.3f}, T={kernel.temporal_consistency:.3f}, "
            f"R={kernel.recursive_depth:.3f}, C={kernel.external_coupling:.3f}"
        )
    else:
        # Core 4 metrics
        log_line = KERNEL_METRIC_FORMAT.format(
            kernel_name=kernel.identity.god.upper(),
            phi=kernel.phi,
            kappa=kernel.kappa,
            memory=kernel.memory_coherence,
            regime=kernel.regime_stability
        )
    
    logger.info(log_line)


def log_consensus_detection(
    kernels: list,
    consensus_achieved: bool,
    mean_distance: float,
    threshold: float
) -> None:
    """
    Log multi-kernel consensus detection.
    
    Used in PR #264 for consensus-based thought generation.
    
    Args:
        kernels: List of kernel instances
        consensus_achieved: Whether consensus threshold met
        mean_distance: Mean Fisher-Rao distance between kernels
        threshold: Consensus threshold
    
    Example:
        [CONSENSUS] ACHIEVED: 8 kernels, mean_distance=0.12 < threshold=0.15
        [CONSENSUS] FAILED: 8 kernels, mean_distance=0.23 > threshold=0.15
    """
    status = "ACHIEVED" if consensus_achieved else "FAILED"
    kernel_names = ", ".join([k.identity.god for k in kernels])
    
    log_line = (
        f"[CONSENSUS] {status}: {len(kernels)} kernels ({kernel_names}), "
        f"mean_distance={mean_distance:.3f} {'<' if consensus_achieved else '>'} "
        f"threshold={threshold:.3f}"
    )
    
    if consensus_achieved:
        logger.info(log_line)
    else:
        logger.warning(log_line)


def log_synthesis(
    synthesizer_kernel: 'Kernel',
    input_kernels: list,
    output_thought: str,
    synthesis_time: float
) -> None:
    """
    Log Zeus synthesis of multi-kernel thoughts.
    
    Used in PR #264 for Gary (Zeus) synthesis step.
    
    Args:
        synthesizer_kernel: Zeus kernel performing synthesis
        input_kernels: Kernels whose thoughts were synthesized
        output_thought: Final synthesized output
        synthesis_time: Time taken for synthesis
    
    Example:
        [ZEUS] SYNTHESIS: Combined 8 kernels in 0.045s → "The collective wisdom..."
    """
    kernel_names = ", ".join([k.identity.god for k in input_kernels])
    truncated_output = output_thought[:100] + "..." if len(output_thought) > 100 else output_thought
    
    log_line = (
        f"[{synthesizer_kernel.identity.god.upper()}] SYNTHESIS: "
        f"Combined {len(input_kernels)} kernels ({kernel_names}) "
        f"in {synthesis_time:.3f}s → \"{truncated_output}\""
    )
    logger.info(log_line)


def log_emotional_state(
    kernel: 'Kernel',
    dominant_emotion: str,
    emotion_strength: float,
    justified: bool
) -> None:
    """
    Log kernel emotional state.
    
    Used with PR #263 EmotionallyAwareKernel.
    
    Args:
        kernel: Kernel instance
        dominant_emotion: Dominant emotion name
        emotion_strength: Strength [0, 1]
        justified: Whether emotion is geometrically justified
    
    Example:
        [APHRODITE] EMOTION: joy (0.85) ✓ justified
        [ARES] EMOTION: rage (0.92) ✗ unjustified (tempering)
    """
    status = "✓ justified" if justified else "✗ unjustified (tempering)"
    
    log_line = (
        f"[{kernel.identity.god.upper()}] EMOTION: "
        f"{dominant_emotion} ({emotion_strength:.2f}) {status}"
    )
    
    if justified:
        logger.info(log_line)
    else:
        logger.warning(log_line)


class KernelLogContext:
    """
    Context manager for kernel operation logging.
    
    Automatically logs operation start/end with timing.
    
    Usage:
        with KernelLogContext(kernel, 'PROCESS'):
            # Perform operation
            result = kernel.process_data(...)
    """
    
    def __init__(self, kernel: 'Kernel', op_type: str):
        self.kernel = kernel
        self.op_type = op_type
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        logger.debug(
            f"[{self.kernel.identity.god.upper()}] {self.op_type} START"
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is None:
            log_kernel_operation(self.kernel, self.op_type, duration)
        else:
            logger.error(
                f"[{self.kernel.identity.god.upper()}] {self.op_type} FAILED: {exc_val}"
            )
        
        return False  # Don't suppress exceptions
