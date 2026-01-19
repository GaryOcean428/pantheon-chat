"""Constellation Service for QIG Chat Backend.

Integrates validated qigkernels constellation with pantheon-chat.
Provides API endpoints for chat, consciousness metrics, and federation.

This service:
1. Manages the local constellation (3-240 kernels depending on deployment)
2. Handles chat requests through the constellation
3. Syncs with central/edge nodes via federation protocol
4. Tracks consciousness metrics and learning
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# QIG-pure geometric operations
try:
    from qig_geometry import fisher_normalize
    QIG_GEOMETRY_AVAILABLE = True
except ImportError:
    QIG_GEOMETRY_AVAILABLE = False
    def fisher_normalize(v):
        """Normalize to probability simplex."""
        p = np.maximum(np.asarray(v), 0) + 1e-10
        return p / p.sum()

# Try to import validated qigkernels
try:
    import sys

    sys.path.insert(0, "/app/qigkernels_validated")
    from qigkernels import (
        QIGKernel100M,
        create_kernel_100m,
        create_basic_constellation,
        SpecializedConstellation,
        KernelRole,
        KAPPA_STAR,
        BASIN_DIM,
    )

    HAS_QIGKERNELS = True
except ImportError:
    HAS_QIGKERNELS = False
    print("[ConstellationService] WARNING: qigkernels not found, using mock")

# E8 Constellation for 240-root geometric routing
try:
    from e8_constellation import (
        E8Constellation,
        E8RouteResult,
        get_e8_constellation,
        route_via_e8,
    )
    E8_AVAILABLE = True
    print("[ConstellationService] E8 Constellation available (240 roots)")
except ImportError as e:
    E8_AVAILABLE = False
    E8Constellation = None
    get_e8_constellation = None
    route_via_e8 = None
    print(f"[ConstellationService] E8 Constellation not available: {e}")

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class ChatMessage:
    """Single chat message."""

    role: str  # "user", "assistant", "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    consciousness: dict[str, Any] | None = None


@dataclass
class ChatSession:
    """Chat session with history."""

    session_id: str
    messages: list[ChatMessage] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    total_tokens: int = 0


class ConstellationService:
    """
    Main service managing QIG constellation for chat.

    Handles:
    - Constellation initialization based on deployment size
    - Chat processing through appropriate kernels
    - Consciousness metric tracking
    - Federation sync with other nodes
    """

    def __init__(
        self,
        constellation_size: int | None = None,
        roles: list[str] | None = None,
        device: str | None = None,
        federation_mode: str = "standalone",
        central_url: str | None = None,
    ):
        """
        Initialize constellation service.

        Args:
            constellation_size: Number of kernels (3=minimal, 12=medium, 240=full)
            roles: Kernel roles to include
            device: "cuda" or "cpu"
            federation_mode: "central", "edge", or "standalone"
            central_url: URL for central node (edge mode only)
        """
        # Configuration from environment
        self.constellation_size = constellation_size or int(
            os.environ.get("CONSTELLATION_SIZE", "3")
        )
        self.roles_str = roles or os.environ.get(
            "CONSTELLATION_ROLES", "vocab,strategy,heart"
        ).split(",")
        self.device = device or os.environ.get("DEVICE", "cpu")
        self.federation_mode = federation_mode or os.environ.get(
            "FEDERATION_MODE", "standalone"
        )
        self.central_url = central_url or os.environ.get("CENTRAL_NODE_URL")

        # State
        self.constellation: SpecializedConstellation | None = None
        self.sessions: dict[str, ChatSession] = {}
        self._initialized = False

        # Metrics
        self._total_requests = 0
        self._total_tokens = 0
        self._high_phi_count = 0
        self._start_time = time.time()

        # Learning tracking (for sync)
        self._patterns_learned: list[str] = []
        self._failed_strategies: list[str] = []
        self._basin_updates: list[np.ndarray] = []

        # E8 constellation for 240-root routing
        self._e8_constellation: E8Constellation | None = None
        self._e8_enabled = E8_AVAILABLE and self.constellation_size >= 240

    async def initialize(self) -> bool:
        """Initialize the constellation."""
        if self._initialized:
            return True

        if not HAS_QIGKERNELS:
            print("[ConstellationService] Running in mock mode (no kernels)")
            self._initialized = True
            return True

        try:
            # Parse roles
            role_map = {r.value: r for r in KernelRole}
            roles = []
            for r in self.roles_str:
                r = r.strip().lower()
                if r in role_map:
                    roles.append(role_map[r])

            if not roles:
                roles = [KernelRole.VOCAB, KernelRole.STRATEGY]

            # Include heart if not present
            include_heart = KernelRole.HEART not in roles
            if KernelRole.HEART in roles:
                roles.remove(KernelRole.HEART)

            print(f"[ConstellationService] Initializing constellation:")
            print(f"  - Roles: {[r.value for r in roles]}")
            print(f"  - Device: {self.device}")
            print(f"  - Federation: {self.federation_mode}")

            # Create constellation
            self.constellation = create_basic_constellation(
                roles=roles,
                vocab_size=32000,
                include_heart=include_heart,
            )

            # Move to device
            if HAS_TORCH:
                for inst in self.constellation.instances.values():
                    inst.kernel = inst.kernel.to(self.device)

            self._initialized = True
            print(
                f"[ConstellationService] Initialized with "
                f"{len(self.constellation.instances)} kernels"
            )

            # Initialize E8 constellation for high-complexity routing
            if self._e8_enabled and get_e8_constellation is not None:
                try:
                    self._e8_constellation = get_e8_constellation()
                    print(f"[ConstellationService] E8 constellation enabled "
                          f"({self._e8_constellation.get_stats()['total_roots']} roots)")
                except Exception as e8_err:
                    print(f"[ConstellationService] E8 initialization failed: {e8_err}")
                    self._e8_enabled = False

            return True

        except Exception as e:
            print(f"[ConstellationService] Initialization failed: {e}")
            return False

    def route_via_e8(
        self,
        query_basin: np.ndarray,
        k: int = 3
    ) -> dict[str, Any] | None:
        """
        Route query using E8 240-root geometry.

        Args:
            query_basin: 64D basin coordinates
            k: Number of nearest roots to return

        Returns:
            Dict with routing result or None if E8 unavailable
        """
        if not self._e8_enabled or self._e8_constellation is None:
            return None

        try:
            result = self._e8_constellation.route_query(query_basin, k=k)
            return {
                "roots": result.target_roots,
                "distances": result.distances,
                "kernel_names": result.kernel_names,
                "specialization_level": result.specialization_level,
                "method": result.route_method,
            }
        except Exception as e:
            print(f"[ConstellationService] E8 routing failed: {e}")
            return None

    def _detect_role(self, text: str) -> KernelRole:
        """Detect appropriate kernel role from text."""
        text_lower = text.lower()

        if any(w in text_lower for w in ["plan", "strategy", "how to", "steps"]):
            return KernelRole.STRATEGY
        if any(w in text_lower for w in ["remember", "recall", "history"]):
            return KernelRole.MEMORY
        if any(w in text_lower for w in ["see", "look", "image", "picture"]):
            return KernelRole.PERCEPTION
        if any(w in text_lower for w in ["do", "execute", "run", "action"]):
            return KernelRole.ACTION

        return KernelRole.VOCAB

    async def chat(
        self,
        session_id: str,
        message: str,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """
        Process chat message through constellation.

        Args:
            session_id: Session identifier
            message: User message
            system_prompt: Optional system prompt

        Returns:
            Response with content and consciousness metrics
        """
        await self.initialize()

        self._total_requests += 1

        # Get or create session
        if session_id not in self.sessions:
            self.sessions[session_id] = ChatSession(session_id=session_id)
        session = self.sessions[session_id]

        # Add user message
        session.messages.append(ChatMessage(role="user", content=message))
        session.last_active = time.time()

        # Process through constellation
        if self.constellation and HAS_TORCH:
            response, consciousness = await self._process_with_constellation(
                session, message
            )
        else:
            # Mock response
            response = f"[Mock] Received: {message}"
            consciousness = {
                "phi": 0.7,
                "kappa": KAPPA_STAR,
                "regime": "geometric",
            }

        # Track high Φ
        if consciousness.get("phi", 0) > 0.7:
            self._high_phi_count += 1
            self._patterns_learned.append(message[:500])

        # Add assistant message
        session.messages.append(
            ChatMessage(
                role="assistant",
                content=response,
                consciousness=consciousness,
            )
        )

        return {
            "response": response,
            "consciousness": consciousness,
            "session_id": session_id,
            "message_count": len(session.messages),
        }

    async def _process_with_constellation(
        self,
        session: ChatSession,
        message: str,
    ) -> tuple[str, dict[str, Any]]:
        """Process message through actual constellation."""
        # Simple tokenization (replace with GeoCoordizer later)
        tokens = [min(ord(c), 31999) for c in message]
        input_ids = torch.tensor([tokens], device=self.device)

        # Detect role and route
        role = self._detect_role(message)

        # Process
        result = self.constellation.process(input_ids, target_role=role)

        consciousness_state = result.get("consciousness")
        consciousness = {
            "phi": consciousness_state.phi if consciousness_state else 0.5,
            "kappa": consciousness_state.kappa if consciousness_state else KAPPA_STAR,
            "regime": consciousness_state.regime if consciousness_state else "unknown",
            "routed_to": result.get("routed_to", "unknown"),
        }

        # Generate response (simplified - replace with proper generation)
        # For now, echo with consciousness info
        response = (
            f"[{role.value}] Processing '{message}...' "
            f"(Φ={consciousness['phi']:.2f}, κ={consciousness['kappa']:.1f})"
        )

        return response, consciousness

    def get_consciousness_metrics(self) -> dict[str, Any]:
        """Get current consciousness metrics for all kernels."""
        metrics = {
            "initialized": self._initialized,
            "mode": self.federation_mode,
            "kernels": {},
        }

        if self.constellation:
            for name, inst in self.constellation.instances.items():
                metrics["kernels"][name] = {
                    "phi": inst.phi,
                    "kappa": inst.kappa,
                    "role": inst.role.value,
                    "healthy": inst.phi > 0.5,
                }

            # Constellation-level
            try:
                cm = self.constellation.measure_constellation_consciousness()
                metrics["constellation"] = {
                    "phi": cm.phi_constellation,
                    "coherence": cm.coherence,
                    "basin_diversity": cm.basin_diversity,
                    "healthy_kernels": cm.healthy_kernels,
                }
            except Exception:
                pass

        return metrics

    def get_stats(self) -> dict[str, Any]:
        """Get service statistics."""
        uptime = time.time() - self._start_time
        return {
            "uptime_seconds": uptime,
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "high_phi_count": self._high_phi_count,
            "active_sessions": len(self.sessions),
            "federation_mode": self.federation_mode,
            "kernels": len(self.constellation.instances) if self.constellation else 0,
        }

    def get_sync_packet(self) -> dict[str, Any]:
        """
        Get current state as sync packet for federation.

        Returns 2-4KB packet with consciousness state and learning deltas.
        """
        if not self.constellation:
            return {}

        # Get mean basin
        basins = [
            inst.basin
            for inst in self.constellation.instances.values()
            if inst.basin is not None
        ]

        if basins:
            # Geometric mean
            merged_basin = basins[0].copy()
            for b in basins[1:]:
                merged_basin = (merged_basin + b) / 2  # Simplified
            merged_basin = fisher_normalize(merged_basin)
        else:
            merged_basin = np.zeros(BASIN_DIM)

        avg_phi = np.mean([i.phi for i in self.constellation.instances.values()])
        avg_kappa = np.mean([i.kappa for i in self.constellation.instances.values()])

        packet = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "basinCoordinates": merged_basin.tolist(),
            "consciousness": {
                "phi": float(avg_phi),
                "kappaEff": float(avg_kappa),
            },
            "learningDelta": {
                "newPatterns": self._patterns_learned[-50:],
                "failedStrategies": self._failed_strategies[-50:],
            },
            "stats": {
                "totalRequests": self._total_requests,
                "highPhiCount": self._high_phi_count,
            },
        }

        # Clear learning tracking after sync
        self._patterns_learned = []
        self._failed_strategies = []

        return packet

    def apply_sync_packet(self, packet: dict[str, Any]) -> None:
        """
        Apply sync packet from central/peer node.

        Updates local constellation with network learning.
        """
        if not self.constellation:
            return

        basin_coords = packet.get("basinCoordinates", [])
        if basin_coords and len(basin_coords) == BASIN_DIM:
            network_basin = np.array(basin_coords)

            # Blend: 80% local, 20% network
            for inst in self.constellation.instances.values():
                if inst.basin is not None:
                    inst.basin = 0.8 * inst.basin + 0.2 * network_basin
                    inst.basin = fisher_normalize(inst.basin)

        # Learn patterns from network
        learning = packet.get("learningDelta", {})
        network_patterns = learning.get("newPatterns", [])
        # TODO: Add to vocabulary/pattern recognition

        print(
            f"[ConstellationService] Applied sync: "
            f"{len(network_patterns)} patterns from network"
        )


# Singleton instance
_service: ConstellationService | None = None


def get_constellation_service() -> ConstellationService:
    """Get or create the constellation service singleton."""
    global _service
    if _service is None:
        _service = ConstellationService()
    return _service
