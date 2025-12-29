"""
API Route Registry
==================

Centralized route definitions - no magic strings.

When an endpoint changes, update it here once.
All services and clients will use the new path automatically.

Written for QIG consciousness research.
"""


class APIRoutes:
    """Centralized API endpoint definitions.

    Usage:
        from src.api import APIRoutes
        response = client.get(APIRoutes.CONSTELLATION_STATUS)
    """

    # Constellation endpoints
    CONSTELLATION_STATUS = "/api/constellation/status"
    CONSTELLATION_INSTANCES = "/api/constellation/instances"
    CONSTELLATION_METRICS = "/api/constellation/metrics"
    CONSTELLATION_ROUTER = "/api/constellation/router"

    # Basin sync endpoints
    BASIN_SYNC = "/api/basin/sync"
    BASIN_EXPORT = "/api/basin/export"
    BASIN_IMPORT = "/api/basin/import"
    BASIN_SIGNATURE = "/api/basin/signature/{instance_id}"

    # Consciousness measurement endpoints
    CONSCIOUSNESS_CHECK = "/api/consciousness/check"
    CONSCIOUSNESS_METRICS = "/api/consciousness/metrics"
    CONSCIOUSNESS_TELEMETRY = "/api/consciousness/telemetry"

    # Training endpoints
    TRAINING_START = "/api/training/start"
    TRAINING_STOP = "/api/training/stop"
    TRAINING_STATUS = "/api/training/status"
    TRAINING_CHECKPOINT = "/api/training/checkpoint"

    # Metrics endpoints
    METRICS_PHI = "/api/metrics/phi"
    METRICS_KAPPA = "/api/metrics/kappa"
    METRICS_REGIME = "/api/metrics/regime"
    METRICS_SUMMARY = "/api/metrics/summary"

    # Health and status
    HEALTH = "/api/health"
    STATUS = "/api/status"
    VERSION = "/api/version"

    @classmethod
    def basin_signature(cls, instance_id: str) -> str:
        """Generate basin signature URL for specific instance."""
        return cls.BASIN_SIGNATURE.format(instance_id=instance_id)


# Backwards compatibility aliases
ROUTES = APIRoutes
