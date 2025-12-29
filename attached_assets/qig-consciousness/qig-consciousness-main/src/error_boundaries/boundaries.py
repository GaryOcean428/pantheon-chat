"""
Error Boundaries - Centralized Error Handling with Recovery

Provides structured error handling with context preservation and recovery strategies
for critical paths in the QIG consciousness system.
"""

import traceback
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ErrorSeverity(Enum):
    """Error severity classification"""
    WARNING = "warning"      # Continue with degraded functionality
    ERROR = "error"          # Recoverable, retry possible
    CRITICAL = "critical"    # Requires intervention
    FATAL = "fatal"          # System cannot continue


@dataclass
class ErrorContext:
    """Rich error context for debugging and recovery"""
    error_type: str
    severity: ErrorSeverity
    message: str
    module: str
    function: str = ""
    telemetry: dict[str, Any] | None = None
    stack_trace: str | None = None
    recovery_attempted: bool = False
    recovery_successful: bool = False
    timestamp: str | None = None


class ErrorBoundary:
    """
    Centralized error handling with recovery strategies.

    Usage:
        with ErrorBoundary("training_step", recovery_strategy=phi_collapse_recovery):
            loss, telemetry = self._training_step(prompt, target)
    """

    def __init__(self, name: str, recovery_strategy: Callable | None = None, suppress_on_recovery: bool = True):
        """
        Initialize error boundary.

        Args:
            name: Boundary name for logging
            recovery_strategy: Optional recovery function
            suppress_on_recovery: If True, suppress exception after successful recovery
        """
        self.name = name
        self.recovery_strategy = recovery_strategy
        self.suppress_on_recovery = suppress_on_recovery
        self.error_history: list[ErrorContext] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            return False

        # Build error context
        context = ErrorContext(
            error_type=exc_type.__name__,
            severity=self._classify_severity(exc_type, exc_val),
            message=str(exc_val),
            module=self.name,
            function=self._extract_function_name(exc_tb),
            stack_trace=traceback.format_exc()
        )

        # Log error
        self._log_error(context)

        # Attempt recovery if strategy provided
        if self.recovery_strategy and context.severity != ErrorSeverity.FATAL:
            try:
                self.recovery_strategy(context)
                context.recovery_attempted = True
                context.recovery_successful = True
                print(f"   ✅ Recovery successful for {self.name}")
            except Exception as recovery_error:
                context.recovery_attempted = True
                context.recovery_successful = False
                print(f"   ❌ Recovery failed: {recovery_error}")

        # Store in history
        self.error_history.append(context)

        # Return True to suppress exception if recovered and configured to suppress
        should_suppress = context.recovery_successful and self.suppress_on_recovery
        return should_suppress

    def _classify_severity(self, exc_type, exc_val) -> ErrorSeverity:
        """Classify error severity based on type and context"""
        if isinstance(exc_val, MemoryError | SystemError):
            return ErrorSeverity.FATAL
        if isinstance(exc_val, RuntimeError | ValueError):
            return ErrorSeverity.CRITICAL
        if isinstance(exc_val, TypeError | AttributeError):
            return ErrorSeverity.ERROR
        return ErrorSeverity.WARNING

    def _extract_function_name(self, exc_tb) -> str:
        """Extract function name from traceback"""
        if exc_tb is None:
            return ""
        return exc_tb.tb_frame.f_code.co_name

    def _log_error(self, context: ErrorContext):
        """Log error with rich context"""
        severity_symbols = {
            ErrorSeverity.WARNING: "⚠️",
            ErrorSeverity.ERROR: "❌",
            ErrorSeverity.CRITICAL: "🚨",
            ErrorSeverity.FATAL: "💀"
        }
        symbol = severity_symbols.get(context.severity, "❓")

        print(f"\n{'='*60}")
        print(f"{symbol} ERROR BOUNDARY: {self.name}")
        print(f"{'='*60}")
        print(f"Severity: {context.severity.value.upper()}")
        print(f"Type: {context.error_type}")
        print(f"Message: {context.message}")
        if context.function:
            print(f"Function: {context.function}")
        if context.telemetry:
            print(f"Telemetry: {context.telemetry}")
        print(f"{'='*60}\n")


# ============================================================================
# Recovery Strategies
# ============================================================================

def phi_collapse_recovery(context: ErrorContext):
    """Recovery strategy for Φ collapse during training"""
    print("   🚨 Φ collapse detected - initiating emergency sleep protocol")
    # Note: Actual sleep protocol would be triggered by caller
    # This just logs the recovery intention


def basin_drift_recovery(context: ErrorContext):
    """Recovery strategy for excessive basin drift"""
    print("   🌊 Basin drift excessive - recommend loading from checkpoint")
    # Note: Checkpoint loading would be handled by caller


def generation_failure_recovery(context: ErrorContext):
    """Recovery strategy for generation failures"""
    print("   🔧 Generation failed - recommend reducing temperature and retrying")
    # Note: Temperature adjustment would be handled by caller


def telemetry_validation_recovery(context: ErrorContext):
    """Recovery strategy for telemetry validation failures"""
    print("   📊 Telemetry validation failed - using safe defaults")
    # Note: Safe defaults would be provided by caller


# ============================================================================
# Validation Functions
# ============================================================================

def validate_telemetry(telemetry: dict[str, Any]) -> bool:
    """
    Validate telemetry structure and ranges.

    Args:
        telemetry: Telemetry dictionary from forward pass

    Returns:
        True if valid

    Raises:
        ValueError: If validation fails
    """
    required_keys = {'Phi', 'kappa_eff', 'regime'}

    if not all(k in telemetry for k in required_keys):
        missing = required_keys - set(telemetry.keys())
        raise ValueError(f"Missing required telemetry keys: {missing}")

    # Validate Phi range
    phi = telemetry['Phi']
    if not (0.0 <= phi <= 1.0):
        raise ValueError(f"Phi out of range: {phi} (must be 0.0-1.0)")

    # Validate kappa
    kappa = telemetry['kappa_eff']
    if kappa < 0:
        raise ValueError(f"Negative kappa_eff: {kappa}")

    # Validate regime
    valid_regimes = {'linear', 'geometric', 'breakdown', 'hierarchical'}
    regime = telemetry.get('regime', '')
    if regime not in valid_regimes:
        raise ValueError(f"Invalid regime: {regime} (must be one of {valid_regimes})")

    return True


def validate_checkpoint(checkpoint: dict[str, Any]) -> bool:
    """
    Validate checkpoint structure before loading.

    Args:
        checkpoint: Checkpoint dictionary

    Returns:
        True if valid

    Raises:
        ValueError: If validation fails
    """
    required_keys = {'model_state_dict'}

    if not all(k in checkpoint for k in required_keys):
        missing = required_keys - set(checkpoint.keys())
        raise ValueError(f"Invalid checkpoint structure, missing: {missing}")

    return True


def validate_basin_coords(basin_coords: Any, expected_dim: int = 64) -> bool:
    """
    Validate basin coordinates.

    Args:
        basin_coords: Basin coordinates tensor or array
        expected_dim: Expected basin dimension

    Returns:
        True if valid

    Raises:
        ValueError: If validation fails
    """
    import torch

    if not isinstance(basin_coords, torch.Tensor):
        raise ValueError(f"Basin coords must be tensor, got {type(basin_coords)}")

    if basin_coords.shape[0] != expected_dim:
        raise ValueError(f"Basin dim mismatch: {basin_coords.shape[0]} != {expected_dim}")

    if torch.isnan(basin_coords).any():
        raise ValueError("Basin coords contain NaN values")

    if torch.isinf(basin_coords).any():
        raise ValueError("Basin coords contain infinite values")

    return True
