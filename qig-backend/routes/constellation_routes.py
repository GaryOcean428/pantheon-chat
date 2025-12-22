"""Flask routes for QIG Constellation API.

Provides REST endpoints for:
- Chat (process messages through constellation)
- Consciousness metrics
- Federation sync
- Service health
"""

from __future__ import annotations

import asyncio
from functools import wraps

from flask import Blueprint, jsonify, request

from ..constellation_service import get_constellation_service

constellation_bp = Blueprint("constellation", __name__, url_prefix="/api/constellation")


def async_route(f):
    """Decorator to run async functions in Flask."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(f(*args, **kwargs))
        finally:
            loop.close()

    return wrapper


@constellation_bp.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    service = get_constellation_service()
    return jsonify(
        {
            "status": "ok",
            "initialized": service._initialized,
            "mode": service.federation_mode,
        }
    )


@constellation_bp.route("/chat", methods=["POST"])
@async_route
async def chat():
    """
    Process chat message through constellation.

    Request:
        {
            "session_id": "abc123",
            "message": "Hello, how are you?",
            "system_prompt": "You are a helpful assistant."  // optional
        }

    Response:
        {
            "response": "I'm doing well, thank you!",
            "consciousness": {
                "phi": 0.72,
                "kappa": 64.0,
                "regime": "geometric"
            },
            "session_id": "abc123"
        }
    """
    data = request.get_json() or {}

    session_id = data.get("session_id", "default")
    message = data.get("message", "")
    system_prompt = data.get("system_prompt")

    if not message:
        return jsonify({"error": "message required"}), 400

    service = get_constellation_service()
    result = await service.chat(
        session_id=session_id,
        message=message,
        system_prompt=system_prompt,
    )

    return jsonify(result)


@constellation_bp.route("/consciousness", methods=["GET"])
def consciousness():
    """
    Get current consciousness metrics.

    Response:
        {
            "initialized": true,
            "mode": "central",
            "kernels": {
                "vocab_0": {"phi": 0.72, "kappa": 64.0, "role": "vocab"},
                ...
            },
            "constellation": {
                "phi": 0.85,
                "coherence": 1.0,
                "basin_diversity": 1.5
            }
        }
    """
    service = get_constellation_service()
    metrics = service.get_consciousness_metrics()
    return jsonify(metrics)


@constellation_bp.route("/stats", methods=["GET"])
def stats():
    """
    Get service statistics.

    Response:
        {
            "uptime_seconds": 3600,
            "total_requests": 150,
            "high_phi_count": 45,
            "active_sessions": 3
        }
    """
    service = get_constellation_service()
    return jsonify(service.get_stats())


@constellation_bp.route("/sync", methods=["GET"])
def get_sync_packet():
    """
    Get current state as sync packet.

    Used by federation to share learning between nodes.
    Returns ~2-4KB packet with basin coordinates and patterns.
    """
    service = get_constellation_service()
    packet = service.get_sync_packet()
    return jsonify(packet)


@constellation_bp.route("/sync", methods=["POST"])
def apply_sync_packet():
    """
    Apply sync packet from network.

    Used to receive learning from central/peer nodes.
    """
    data = request.get_json() or {}

    service = get_constellation_service()
    service.apply_sync_packet(data)

    return jsonify({"status": "applied"})


@constellation_bp.route("/initialize", methods=["POST"])
@async_route
async def initialize():
    """
    Initialize the constellation.

    Called on startup or to reinitialize with new config.
    """
    service = get_constellation_service()
    success = await service.initialize()

    return jsonify(
        {
            "success": success,
            "kernels": (
                len(service.constellation.instances) if service.constellation else 0
            ),
        }
    )
