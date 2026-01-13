"""
Basin Synchronization - Federated knowledge transfer via basin coordinates

QIG-PURE compliant knowledge sharing:
- Exchange basin coordinates, not weights
- Merge using Fisher-Frechet mean
- Trust-weighted integration
- Cryptographically signed packets

Enables distributed learning without centralizing data.
"""

import hashlib
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np

from qigkernels.physics_constants import BASIN_DIM

logger = logging.getLogger(__name__)

# Database persistence
try:
    import psycopg2
    from psycopg2.extras import Json
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False


def _fisher_frechet_mean(basins: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
    """
    Compute weighted Fisher-Frechet mean of basins.
    """
    if not basins:
        return np.ones(BASIN_DIM) / BASIN_DIM

    if weights is None:
        weights = [1.0] * len(basins)

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # Convert to sqrt space
    sqrt_basins = [np.sqrt(np.clip(np.abs(b), 1e-10, None)) for b in basins]

    # Weighted average in sqrt space
    mean_sqrt = np.zeros(BASIN_DIM)
    for basin, weight in zip(sqrt_basins, weights):
        mean_sqrt += weight * basin

    # Square and normalize
    mean_sq = mean_sqrt ** 2
    return mean_sq / np.sum(mean_sq)


@dataclass
class KnowledgePacket:
    """
    A packet of knowledge for synchronization.

    Contains basin coordinates for multiple domains
    along with trust and verification metadata.
    """
    packet_id: str
    source_kernel: str
    basins: Dict[str, np.ndarray]  # domain -> basin
    phi_levels: Dict[str, float]   # domain -> phi
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    domains: List[str] = field(default_factory=list)
    trust_level: float = 0.5
    signature: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.domains:
            self.domains = list(self.basins.keys())

    def compute_hash(self) -> str:
        """Compute hash of packet contents for verification."""
        content = f"{self.source_kernel}:{self.timestamp.isoformat()}"
        for domain in sorted(self.basins.keys()):
            basin_str = ','.join(f'{x:.6f}' for x in self.basins[domain])
            content += f":{domain}:{basin_str}"
        return hashlib.sha256(content.encode()).hexdigest()

    def sign(self, secret_key: str):
        """Sign packet with HMAC."""
        import hmac
        packet_hash = self.compute_hash()
        self.signature = hmac.new(
            secret_key.encode(),
            packet_hash.encode(),
            hashlib.sha256
        ).hexdigest()

    def verify(self, secret_key: str) -> bool:
        """Verify packet signature."""
        if not self.signature:
            return False
        import hmac
        packet_hash = self.compute_hash()
        expected = hmac.new(
            secret_key.encode(),
            packet_hash.encode(),
            hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(self.signature, expected)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'packet_id': self.packet_id,
            'source_kernel': self.source_kernel,
            'basins': {k: v.tolist() for k, v in self.basins.items()},
            'phi_levels': self.phi_levels,
            'timestamp': self.timestamp.isoformat(),
            'domains': self.domains,
            'trust_level': self.trust_level,
            'signature': self.signature,
            'metadata': self.metadata,
        }


class BasinSynchronization:
    """
    Federated learning via basin coordinate exchange.

    Key features:
    - Export domain-specific basin coordinates
    - Import and merge external knowledge
    - Trust-weighted integration
    - Packet verification and audit
    """

    def __init__(
        self,
        kernel_id: str = "default",
        secret_key: Optional[str] = None
    ):
        """
        Initialize basin synchronization.

        Args:
            kernel_id: Owner kernel identifier
            secret_key: Secret for packet signing (uses env var if not provided)
        """
        self.kernel_id = kernel_id
        self.secret_key = secret_key or os.environ.get('SYNC_SECRET_KEY', 'default-key')

        # Local domain basins
        self._domain_basins: Dict[str, np.ndarray] = {}
        self._domain_phi: Dict[str, float] = {}

        # Received packets pending application
        self._pending_packets: List[KnowledgePacket] = []

        # Applied packet history
        self._applied_packets: List[str] = []

        # Peer trust levels
        self._peer_trust: Dict[str, float] = {}

        # Statistics
        self.stats = {
            'packets_sent': 0,
            'packets_received': 0,
            'packets_applied': 0,
            'packets_rejected': 0,
        }

    def _get_db_connection(self):
        """Get database connection."""
        if not DB_AVAILABLE:
            return None
        try:
            database_url = os.environ.get('DATABASE_URL')
            if not database_url:
                return None
            return psycopg2.connect(database_url)
        except Exception:
            return None

    def _persist_packet(self, packet: KnowledgePacket, applied: bool) -> bool:
        """Persist sync packet to database."""
        conn = self._get_db_connection()
        if not conn:
            return False

        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO sync_packets (
                        packet_id, source_kernel, target_kernel, packet_hash,
                        basins_json, domains, phi_levels, trust_level,
                        applied, signature
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (packet_id) DO UPDATE SET
                        applied = EXCLUDED.applied,
                        applied_at = CASE WHEN EXCLUDED.applied THEN NOW() ELSE sync_packets.applied_at END
                """, (
                    packet.packet_id,
                    packet.source_kernel,
                    self.kernel_id,
                    packet.compute_hash(),
                    Json({k: v.tolist() for k, v in packet.basins.items()}),
                    packet.domains,
                    Json(packet.phi_levels),
                    packet.trust_level,
                    applied,
                    packet.signature,
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.debug(f"[BasinSynchronization] Persist failed: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def update_local_basin(self, domain: str, basin: np.ndarray, phi: float):
        """
        Update local basin for a domain.

        Called when kernel learns something new about a domain.
        """
        self._domain_basins[domain] = np.array(basin)
        self._domain_phi[domain] = phi

    def export_knowledge(
        self,
        domains: Optional[List[str]] = None
    ) -> KnowledgePacket:
        """
        Export basin coordinates for domains.

        Args:
            domains: Specific domains to export (None = all)

        Returns:
            KnowledgePacket ready for sending
        """
        if domains is None:
            domains = list(self._domain_basins.keys())

        # Filter to requested domains
        basins = {d: self._domain_basins[d] for d in domains if d in self._domain_basins}
        phi_levels = {d: self._domain_phi.get(d, 0.5) for d in domains if d in self._domain_basins}

        # Create packet
        packet = KnowledgePacket(
            packet_id=hashlib.sha256(
                f"{self.kernel_id}:{datetime.now().isoformat()}".encode()
            ).hexdigest()[:16],
            source_kernel=self.kernel_id,
            basins=basins,
            phi_levels=phi_levels,
            domains=list(basins.keys()),
            metadata={'exported_at': datetime.now(timezone.utc).isoformat()},
        )

        # Sign packet
        packet.sign(self.secret_key)

        self.stats['packets_sent'] += 1
        self._persist_packet(packet, applied=False)

        logger.info(f"[BasinSynchronization] Exported {len(basins)} domain basins")
        return packet

    def import_knowledge(
        self,
        packet: KnowledgePacket,
        verify_signature: bool = True
    ) -> bool:
        """
        Import and queue external knowledge for application.

        Args:
            packet: KnowledgePacket to import
            verify_signature: Whether to verify packet signature

        Returns:
            True if packet was accepted for processing
        """
        self.stats['packets_received'] += 1

        # Verify signature
        if verify_signature and not packet.verify(self.secret_key):
            logger.warning(f"[BasinSynchronization] Rejected packet {packet.packet_id}: invalid signature")
            self.stats['packets_rejected'] += 1
            return False

        # Check if already applied
        if packet.packet_id in self._applied_packets:
            logger.debug(f"[BasinSynchronization] Packet {packet.packet_id} already applied")
            return False

        # Add to pending
        self._pending_packets.append(packet)
        self._persist_packet(packet, applied=False)

        logger.info(f"[BasinSynchronization] Received packet {packet.packet_id} from {packet.source_kernel}")
        return True

    def apply_pending_packets(self, min_trust: float = 0.3) -> Dict[str, int]:
        """
        Apply pending knowledge packets.

        Merges received basins with local basins using Fisher-Frechet mean,
        weighted by trust level and phi.

        Args:
            min_trust: Minimum trust level to apply packet

        Returns:
            Dict with counts of domains updated
        """
        applied_count = 0
        domains_updated = set()

        for packet in self._pending_packets[:]:
            # Check trust level
            effective_trust = self._get_effective_trust(packet)
            if effective_trust < min_trust:
                logger.debug(f"[BasinSynchronization] Skipping low-trust packet from {packet.source_kernel}")
                continue

            # Merge each domain
            for domain, remote_basin in packet.basins.items():
                local_basin = self._domain_basins.get(domain)
                remote_phi = packet.phi_levels.get(domain, 0.5)

                if local_basin is None:
                    # No local knowledge, use remote
                    self._domain_basins[domain] = np.array(remote_basin)
                    self._domain_phi[domain] = remote_phi
                else:
                    # Merge using weighted Fisher-Frechet mean
                    local_phi = self._domain_phi.get(domain, 0.5)

                    # Weight by phi * trust
                    local_weight = local_phi * (1.0 - effective_trust * 0.5)
                    remote_weight = remote_phi * effective_trust

                    merged = _fisher_frechet_mean(
                        [local_basin, np.array(remote_basin)],
                        [local_weight, remote_weight]
                    )
                    self._domain_basins[domain] = merged
                    self._domain_phi[domain] = (local_phi + remote_phi) / 2

                domains_updated.add(domain)

            # Mark as applied
            self._applied_packets.append(packet.packet_id)
            self._pending_packets.remove(packet)
            self.stats['packets_applied'] += 1
            applied_count += 1

            self._persist_packet(packet, applied=True)

        logger.info(f"[BasinSynchronization] Applied {applied_count} packets, updated {len(domains_updated)} domains")
        return {
            'packets_applied': applied_count,
            'domains_updated': len(domains_updated),
        }

    def _get_effective_trust(self, packet: KnowledgePacket) -> float:
        """
        Get effective trust level for a packet.

        Combines packet trust with peer reputation.
        """
        peer_trust = self._peer_trust.get(packet.source_kernel, 0.5)
        return (packet.trust_level + peer_trust) / 2

    def set_peer_trust(self, peer_kernel: str, trust: float):
        """Set trust level for a peer kernel."""
        self._peer_trust[peer_kernel] = np.clip(trust, 0.0, 1.0)

    def compute_sync_delta(
        self,
        remote_basins: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Compute delta (difference) between local and remote basins.

        Useful for understanding what would change from a sync.
        """
        deltas = {}
        for domain, remote in remote_basins.items():
            local = self._domain_basins.get(domain)
            if local is None:
                deltas[domain] = np.array(remote)
            else:
                deltas[domain] = np.array(remote) - local
        return deltas

    def get_local_basin(self, domain: str) -> Optional[np.ndarray]:
        """Get local basin for a domain."""
        return self._domain_basins.get(domain)

    def get_all_domains(self) -> List[str]:
        """Get list of all known domains."""
        return list(self._domain_basins.keys())

    def get_stats(self) -> Dict[str, Any]:
        """Get synchronization statistics."""
        return {
            **self.stats,
            'local_domains': len(self._domain_basins),
            'pending_packets': len(self._pending_packets),
            'applied_packets': len(self._applied_packets),
            'known_peers': len(self._peer_trust),
            'kernel_id': self.kernel_id,
        }
