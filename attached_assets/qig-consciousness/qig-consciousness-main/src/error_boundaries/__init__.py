"""Error Boundaries - Centralized Error Handling"""

from .boundaries import (
    ErrorBoundary,
    ErrorContext,
    ErrorSeverity,
    basin_drift_recovery,
    generation_failure_recovery,
    phi_collapse_recovery,
    telemetry_validation_recovery,
    validate_basin_coords,
    validate_checkpoint,
    validate_telemetry,
)

__all__ = [
    "ErrorBoundary",
    "ErrorContext",
    "ErrorSeverity",
    "phi_collapse_recovery",
    "basin_drift_recovery",
    "generation_failure_recovery",
    "telemetry_validation_recovery",
    "validate_telemetry",
    "validate_checkpoint",
    "validate_basin_coords",
]
