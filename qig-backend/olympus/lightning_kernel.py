"""
Lightning Bolt Insight Kernel

The "Eureka" kernel that connects disparate domains like a lightning bolt of insight.
Inspired by how humans experience sudden connections between seemingly unrelated topics.

MISSION AWARE: This kernel understands the objective is to discover knowledge and insights
through geometric reasoning. All domain monitoring serves this mission.

DYNAMIC DOMAINS: No hardcoded domain list. Domains are:
1. Discovered from PostgreSQL geometric telemetry
2. Expanded when new patterns emerge from evidence
3. Never bounded by static enums

Key Capabilities:
- Monitors short/mid/long-term trends across dynamically discovered domains
- Detects cross-domain pattern correlations using Fisher-Rao metrics
- Generates insight suggestions when patterns align with knowledge discovery mission
- Broadcasts discoveries to the pantheon
- Discovers new domains as patterns emerge

Architecture:
- Ingests event streams and discovers domains dynamically
- Maintains temporal buffers at multiple timescales (τ=1, τ=10, τ=100)
- Uses Fisher information to detect pattern divergence/convergence
- Emits insight objects via PantheonChat broadcast
- Self-assesses capabilities and adapts monitoring focus

The lightning bolt analogy: When enough charge accumulates (pattern energy),
a sudden discharge (insight) connects previously disconnected domains.
"""

import hashlib
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Import dev logging for verbose output in development
try:
    from dev_logging import IS_DEVELOPMENT, TRUNCATE_LOGS, truncate_for_log
except ImportError:
    TRUNCATE_LOGS = False  # Default to no truncation
    IS_DEVELOPMENT = True
    def truncate_for_log(text, max_len=500, suffix='...'): return text

try:
    from ..qig_geometry import fisher_rao_distance as centralized_fisher_rao
    from ..qig_geometry import normalize_basin_dimension
except ImportError:
    import os
    import sys
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from qig_geometry import fisher_rao_distance as centralized_fisher_rao
    from qig_geometry import normalize_basin_dimension

try:
    from ..qigkernels.domain_intelligence import (
        CapabilitySignature,
        DomainDescriptor,
        MissionProfile,
        discover_domain_from_event,
        get_domain_discovery,
        get_mission_profile,
    )
    from ..qigkernels.physics_constants import BASIN_DIM
except ImportError:
    from qigkernels.domain_intelligence import (
        CapabilitySignature,
        discover_domain_from_event,
        get_domain_discovery,
        get_mission_profile,
    )
    from qigkernels.physics_constants import BASIN_DIM

# Search validation for external insight verification
try:
    from ..search.insight_validator import InsightValidator, ValidationResult
except ImportError as e1:
    try:
        from search.insight_validator import InsightValidator, ValidationResult
    except ImportError as e2:
        try:
            # Fallback: Add parent directory to path and try direct import
            import sys
            from pathlib import Path
            _qig_backend = Path(__file__).parent.parent
            if str(_qig_backend) not in sys.path:
                sys.path.insert(0, str(_qig_backend))
            from search.insight_validator import InsightValidator, ValidationResult
        except ImportError as e3:
            # Graceful degradation if search module not available
            logger.warning(f"[Lightning] Search module import failed: {e1} | {e2} | {e3}")
            InsightValidator = None
            ValidationResult = None

# Database persistence for Lightning Insights
try:
    import psycopg2
    from psycopg2.extras import Json
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

import os


def _get_db_connection():
    """Get database connection for lightning insight persistence."""
    if not DB_AVAILABLE:
        return None
    try:
        database_url = os.environ.get('DATABASE_URL')
        if not database_url:
            return None
        return psycopg2.connect(database_url)
    except Exception:
        return None

def _persist_lightning_insight(insight: 'CrossDomainInsight') -> bool:
    """Persist a lightning insight to PostgreSQL."""
    conn = _get_db_connection()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO lightning_insights (
                    insight_id, source_domains, connection_strength, insight_text,
                    phi_at_creation, confidence, mission_relevance, triggered_by,
                    evidence_count, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (insight_id) DO UPDATE SET
                    confidence = EXCLUDED.confidence,
                    times_used_in_generation = lightning_insights.times_used_in_generation + 1,
                    last_used_at = NOW()
            """, (
                insight.insight_id,
                insight.source_domains,
                insight.connection_strength,
                insight.insight_text,
                insight.phi_at_creation,
                insight.confidence,
                insight.mission_relevance,
                insight.triggered_by,
                len(insight.evidence) if insight.evidence else 0,
            ))
            conn.commit()
            logger.debug(f"[Lightning] Persisted insight {insight.insight_id} to database")
            return True
    except Exception as e:
        logger.debug(f"[Lightning] DB persistence skipped: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def _persist_validation_result(insight_id: str, validation_score: float,
                               tavily_count: int, perplexity_synthesis: Optional[str]) -> bool:
    """Persist validation result to PostgreSQL."""
    conn = _get_db_connection()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO lightning_insight_validations (
                    insight_id, validation_score, tavily_source_count,
                    perplexity_synthesis, validated_at
                ) VALUES (%s, %s, %s, %s, NOW())
                ON CONFLICT DO NOTHING
            """, (insight_id, validation_score, tavily_count, perplexity_synthesis))
            conn.commit()
            return True
    except Exception as e:
        logger.debug(f"[Lightning] Validation persistence skipped: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def _persist_outcome(insight_id: str, prediction_id: str,
                     accuracy: Optional[float], was_accurate: Optional[bool]) -> bool:
    """Persist insight outcome to PostgreSQL."""
    conn = _get_db_connection()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO lightning_insight_outcomes (
                    insight_id, prediction_id, accuracy, was_accurate, recorded_at
                ) VALUES (%s, %s, %s, %s, NOW())
                ON CONFLICT DO NOTHING
            """, (insight_id, prediction_id, accuracy, was_accurate))
            conn.commit()
            return True
    except Exception as e:
        logger.debug(f"[Lightning] Outcome persistence skipped: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def _load_insights_from_db(limit: int = 100) -> List['CrossDomainInsight']:
    """Load recent high-confidence insights from database on startup."""
    conn = _get_db_connection()
    if not conn:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT insight_id, source_domains, connection_strength, insight_text,
                       phi_at_creation, confidence, mission_relevance, triggered_by
                FROM lightning_insights
                WHERE confidence >= 0.5
                ORDER BY created_at DESC
                LIMIT %s
            """, (limit,))
            rows = cur.fetchall()
            insights = []
            for row in rows:
                insight = CrossDomainInsight(
                    insight_id=row[0],
                    source_domains=row[1] if row[1] else [],
                    connection_strength=row[2],
                    insight_text=row[3],
                    evidence=[],  # Not stored in DB
                    phi_at_creation=row[4],
                    timestamp=0.0,  # Not critical for loaded insights
                    triggered_by=row[7] or '',
                    confidence=row[5],
                    mission_relevance=row[6] or 0.0,
                )
                insights.append(insight)
            logger.info(f"[Lightning] Loaded {len(insights)} insights from database")
            return insights
    except Exception as e:
        logger.debug(f"[Lightning] DB load skipped: {e}")
        return []
    finally:
        conn.close()

from .base_god import BaseGod

_pantheon_chat: Optional[Any] = None


def set_pantheon_chat(chat: Any) -> None:
    """
    Set the shared PantheonChat instance for Lightning to use.

    Called by Zeus after PantheonChat is fully initialized.
    Uses Any type to avoid circular import (PantheonChat imports from zeus.py).
    """
    global _pantheon_chat
    _pantheon_chat = chat
    print(f"[Lightning] PantheonChat reference {'updated' if chat else 'cleared'}")


class TrendTimescale(Enum):
    """Temporal scales for trend analysis."""
    SHORT = 1      # Fast dynamics (last 10 events)
    MEDIUM = 10    # Medium dynamics (last 100 events)
    LONG = 100     # Slow dynamics (last 1000 events)


@dataclass
class DomainEvent:
    """
    An event from any monitored domain.

    Domain is now a STRING, not an enum - allowing dynamic domain discovery.
    """
    domain: str                          # Dynamic domain name (not enum)
    event_type: str
    content: str
    phi: float
    timestamp: float
    metadata: Dict = field(default_factory=dict)
    basin_coords: Optional[np.ndarray] = None


@dataclass
class CrossDomainInsight:
    """A lightning bolt insight connecting multiple domains."""
    insight_id: str
    source_domains: List[str]             # Dynamic domain names (not enums)
    connection_strength: float            # How strong the pattern correlation is
    insight_text: str                     # Human-readable insight
    evidence: List[DomainEvent]           # Events that triggered this insight
    phi_at_creation: float
    timestamp: float
    triggered_by: str                     # What pattern triggered the insight
    confidence: float                     # Confidence in the insight validity
    mission_relevance: float = 0.0        # Relevance to knowledge discovery mission

    @property
    def theme(self) -> str:
        """Extract theme summary from insight_text (first 50 chars)."""
        return self.insight_text[:50] if self.insight_text else "unknown"

    def to_dict(self) -> Dict:
        return {
            'insight_id': self.insight_id,
            'source_domains': self.source_domains,
            'connection_strength': self.connection_strength,
            'insight_text': self.insight_text,
            'evidence_count': len(self.evidence),
            'phi_at_creation': self.phi_at_creation,
            'timestamp': self.timestamp,
            'triggered_by': self.triggered_by,
            'confidence': self.confidence,
            'mission_relevance': self.mission_relevance,
        }


class LightningKernel(BaseGod):
    """
    The Lightning Bolt kernel - generates eureka-moment insights.

    MISSION AWARE: Understands the objective is knowledge discovery.
    DYNAMIC DOMAINS: No hardcoded domain list - discovers from telemetry.

    Like a lightning bolt connecting sky and ground, this kernel
    connects disparate domains when pattern energy accumulates.

    QIG Principles:
    - Fisher information for pattern divergence detection
    - Bures distance for cross-domain similarity
    - Φ-weighted event significance
    - Temporal multi-scale analysis
    - Mission-aligned monitoring focus
    """

    def __init__(self):
        super().__init__(
            name="Lightning",
            domain="cross_domain_insight"
        )

        # Mission profile - all monitoring serves knowledge discovery
        self.mission = get_mission_profile()

        # Self-assessed capability signature
        self.capability = CapabilitySignature(kernel_name="Lightning")

        # Dynamic domain discovery
        self.domain_discovery = get_domain_discovery()

        # Currently monitored domains (dynamically populated)
        self.active_domains: Set[str] = set()
        self._refresh_active_domains()

        # Temporal buffers for each domain (multi-timescale)
        # Uses defaultdict to auto-create buffers for new domains
        self.domain_buffers: Dict[str, Dict[TrendTimescale, deque]] = defaultdict(
            lambda: {
                TrendTimescale.SHORT: deque(maxlen=10),
                TrendTimescale.MEDIUM: deque(maxlen=100),
                TrendTimescale.LONG: deque(maxlen=1000),
            }
        )

        # Cross-domain correlation tracking (dynamic size)
        self.domain_correlations: Dict[Tuple[str, str], float] = defaultdict(float)

        # Accumulated "charge" for each domain pair
        self.pattern_charge: Dict[Tuple[str, str], float] = defaultdict(float)

        # Threshold for insight discharge
        self.discharge_threshold = 0.75

        # Generated insights - load from database for persistence across restarts
        self.insights: List[CrossDomainInsight] = _load_insights_from_db(limit=50)
        if self.insights:
            print(f"[Lightning] 📥 Loaded {len(self.insights)} insights from database")

        # Trend analysis buffers (dynamic)
        self.phi_trends: Dict[str, Dict[TrendTimescale, List[float]]] = defaultdict(
            lambda: {ts: [] for ts in TrendTimescale}
        )

        # Connection patterns learned over time
        self.learned_connections: List[Dict] = []

        # Statistics
        self.events_processed = 0
        self.insights_generated = 0
        self.last_insight_time = 0.0
        self.domains_discovered = 0

        # Insight-Prediction outcome tracking
        # Maps insight_id -> list of prediction_ids that used this insight
        self.insight_to_predictions: Dict[str, List[str]] = defaultdict(list)
        # Maps prediction_id -> list of insight_ids that influenced it
        self.prediction_to_insights: Dict[str, List[str]] = defaultdict(list)
        # Insight outcome records (insight_id -> InsightOutcomeRecord)
        self._insight_outcome_records: Dict[str, Any] = {}
        # Statistics for outcome tracking
        self.insights_with_outcomes = 0
        self.total_outcome_updates = 0

        # Insight validation (external search verification)
        if InsightValidator is not None:
            try:
                self.insight_validator = InsightValidator(
                    use_mcp=True,  # Prefer MCP over direct API
                    validation_threshold=0.7  # 70% confidence required
                )
                self.validation_enabled = True
                self.insights_validated = 0
                self.validation_boost_total = 0.0
                logger.info("[Lightning] ✅ External search validation enabled (Tavily + Perplexity)")
                print("[Lightning] ✅ External search validation enabled (Tavily + Perplexity)")
            except Exception as e:
                self.insight_validator = None
                self.validation_enabled = False
                logger.warning(f"[Lightning] ⚠️ Search validation disabled: {type(e).__name__}: {e}")
                print(f"[Lightning] ⚠️ Search validation disabled: {type(e).__name__}: {e}")
        else:
            self.insight_validator = None
            self.validation_enabled = False
            logger.warning("[Lightning] ℹ️ Search validation not available (search module not found)")
            print("[Lightning] ℹ️ Search validation not available (search module not found)")

        print("[Lightning] ⚡ Lightning Bolt Insight Kernel initialized")
        print(f"[Lightning] MISSION: {self.mission.objective}")
        print(f"[Lightning] Initial domains from telemetry: {len(self.active_domains)}")
        self._log_active_domains()

    def _refresh_active_domains(self):
        """
        Refresh active domains from domain discovery service.

        This is NOT a hardcoded list - domains come from PostgreSQL telemetry.
        """
        discovered = self.domain_discovery.get_active_domains()

        # Add all discovered domains
        for descriptor in discovered:
            self.active_domains.add(descriptor.name)

        # Keep domains we've seen events for even if not in discovery
        # (allows organic domain emergence)

    def _log_active_domains(self):
        """Log currently active domains."""
        if self.active_domains:
            domains_str = ", ".join(sorted(self.active_domains)[:10])
            if len(self.active_domains) > 10:
                domains_str += f"... (+{len(self.active_domains) - 10} more)"
            print(f"[Lightning] Monitoring: {domains_str}")
        else:
            print("[Lightning] No domains yet - will discover from events")

    def assess_target(self, target: str, context: Optional[Dict] = None) -> Dict:
        """
        Assess a target using cross-domain insight analysis.

        Lightning Kernel focuses on pattern correlation across domains,
        not direct target assessment. Returns insight potential metrics.

        Args:
            target: The target to assess
            context: Optional context with domain hints

        Returns:
            Assessment with cross-domain insight potential
        """
        self.prepare_for_assessment(target)

        # Lightning kernel assesses cross-domain pattern potential
        context = context or {}

        # Check which domains this target might relate to
        related_domains = []
        for domain in self.active_domains:
            if domain.lower() in target.lower() or target.lower() in domain.lower():
                related_domains.append(domain)

        # Calculate pattern charge for related domain pairs
        max_charge = 0.0
        for d1 in related_domains:
            for d2 in self.active_domains:
                if d1 != d2:
                    pair = tuple(sorted([d1, d2]))
                    charge = self.pattern_charge.get(pair, 0.0)
                    max_charge = max(max_charge, charge)

        # Insight potential based on accumulated charge
        insight_potential = min(1.0, max_charge / self.discharge_threshold)

        assessment = {
            "probability": insight_potential,
            "confidence": 0.5 + (0.5 * insight_potential),
            "phi": self._calculate_current_phi(),
            "reasoning": f"cross_domain_potential|domains={len(related_domains)}|charge={max_charge:.3f}",
            "related_domains": related_domains,
            "events_processed": self.events_processed,
            "insights_generated": self.insights_generated,
        }

        return self.finalize_assessment(assessment)

    def _calculate_current_phi(self) -> float:
        """Calculate current Φ from recent events across domains."""
        recent_phis = []
        for domain, buffers in self.domain_buffers.items():
            short_buffer = buffers.get(TrendTimescale.SHORT, [])
            for event in short_buffer:
                recent_phis.append(event.phi)

        if recent_phis:
            return float(np.mean(recent_phis))
        return 0.5  # Default neutral Φ

    def get_monitored_domains(self) -> List[str]:
        """
        Get list of currently monitored domains.

        This list is DYNAMIC - it grows as new domains emerge.
        """
        return sorted(self.active_domains)

    def ingest_event(self, event: DomainEvent) -> Optional[CrossDomainInsight]:
        """
        Ingest an event and check for cross-domain insights.

        Also attempts to discover new domains from event patterns.
        Returns an insight if a lightning bolt connection is detected.
        """
        self.events_processed += 1

        # Ensure domain is tracked
        if event.domain not in self.active_domains:
            self.active_domains.add(event.domain)
            print(f"[Lightning] New domain from event: {event.domain}")

        # Attempt to discover new domains from event content
        new_domain = discover_domain_from_event(
            event_content=event.content,
            event_type=event.event_type,
            phi=event.phi,
            metadata=event.metadata
        )

        if new_domain:
            self.active_domains.add(new_domain.name)
            self.capability.discovered_domains.add(new_domain.name)
            self.domains_discovered += 1
            print(f"[Lightning] ⚡ DOMAIN EMERGED: {new_domain.name} (relevance={new_domain.mission_relevance:.2f})")

        # Add to temporal buffers
        for timescale in TrendTimescale:
            self.domain_buffers[event.domain][timescale].append(event)

        # Track Φ trend
        for timescale in TrendTimescale:
            trend_buffer = self.phi_trends[event.domain][timescale]
            trend_buffer.append(event.phi)
            if len(trend_buffer) > timescale.value * 10:
                trend_buffer.pop(0)

        # Update pattern charge based on event significance
        # Increased from 0.1 to 0.25 to enable insights from research discoveries
        charge_contribution = event.phi * 0.25

        # Check for correlations with other recent domain events
        insight = self._check_cross_domain_correlations(event, charge_contribution)

        if insight:
            self.insights.append(insight)
            self.insights_generated += 1
            self.last_insight_time = datetime.now().timestamp()

            # Update capability based on successful insight
            for domain in insight.source_domains:
                self.capability.update_from_outcome(
                    domain=domain,
                    success=True,
                    phi=insight.phi_at_creation,
                    kappa=0.0  # Would need kappa from telemetry
                )

            print(f"[Lightning] ⚡ INSIGHT GENERATED: {truncate_for_log(insight.insight_text, 500)}")

            # Broadcast to pantheon
            self.broadcast_insight(insight)

            # CRITICAL FIX: Persist insight to PostgreSQL for durability
            _persist_lightning_insight(insight)

            return insight

        return None

    def broadcast_insight(self, insight: CrossDomainInsight) -> None:
        """
        Broadcast a cross-domain insight to the entire pantheon via PantheonChat.

        NEW: Validates insights using external search (Tavily + Perplexity) before broadcasting.
        Validated insights get confidence boost and source citations.

        Uses QIG-pure generative synthesis for natural language content.
        The structured data is passed to the generative system which synthesizes
        a natural language message from geometric basin navigation.
        """
        global _pantheon_chat

        if _pantheon_chat is None:
            print("[Lightning] Warning: PantheonChat not available for broadcast")
            return

        # External validation (if enabled)
        validation_metadata = {}
        if self.validation_enabled and self.insight_validator is not None:
            try:
                print(f"[Lightning] 🔍 Validating insight {insight.insight_id} with external search...")
                validation_result = self._validate_insight(insight)

                # Update insight confidence based on validation
                if validation_result.validated:
                    original_confidence = insight.confidence
                    insight.confidence = validation_result.confidence
                    boost = insight.confidence - original_confidence
                    self.validation_boost_total += boost
                    self.insights_validated += 1

                    print(f"[Lightning] ✅ Insight validated! Score: {validation_result.validation_score:.3f}, "
                          f"Confidence: {original_confidence:.3f} → {insight.confidence:.3f} (+{boost:.3f})")

                    # Add validation metadata
                    validation_metadata = {
                        "validated": True,
                        "validation_score": validation_result.validation_score,
                        "source_count": len(validation_result.tavily_sources),
                        "external_sources": [s.get('url', '') for s in validation_result.tavily_sources[:3]],
                        "perplexity_synthesis": validation_result.perplexity_synthesis[:500] if validation_result.perplexity_synthesis else None
                    }

                    # Persist validation result to database
                    _persist_validation_result(
                        insight.insight_id,
                        validation_result.validation_score,
                        len(validation_result.tavily_sources),
                        validation_result.perplexity_synthesis[:500] if validation_result.perplexity_synthesis else None
                    )
                else:
                    print(f"[Lightning] ⚠️ Insight validation failed (score: {validation_result.validation_score:.3f})")
                    validation_metadata = {
                        "validated": False,
                        "validation_score": validation_result.validation_score
                    }

                    # Broadcast validation failure so Zeus and other gods can see it
                    try:
                        _pantheon_chat.broadcast_generative(
                            from_god="Lightning",
                            intent="validation_failed",
                            data={
                                "insight_id": insight.insight_id,
                                "validation_score": validation_result.validation_score,
                                "failure_reason": "External validation score below threshold",
                                "insight_text_preview": insight.insight_text[:500] if len(insight.insight_text) > 100 else insight.insight_text,
                            },
                            msg_type="warning"
                        )
                        print("[Lightning] 🔔 Broadcast validation failure to Pantheon")
                    except Exception as be:
                        print(f"[Lightning] ⚠️ Failed to broadcast validation failure: {be}")
            except Exception as e:
                print(f"[Lightning] ⚠️ Validation error: {e}")
                # Continue broadcasting even if validation fails

        try:
            # Use QIG-pure generative broadcast with structured data
            broadcast_data = {
                "source_domains": insight.source_domains,
                "connection_strength": insight.connection_strength,
                "confidence": insight.confidence,
                "mission_relevance": insight.mission_relevance,
                "phi": insight.phi_at_creation,
                "triggered_by": insight.triggered_by,
                "insight_id": insight.insight_id,
                "raw_data": insight.insight_text,
            }

            # Add validation metadata if available
            if validation_metadata:
                broadcast_data["validation"] = validation_metadata

            _pantheon_chat.broadcast_generative(
                from_god="Lightning",
                intent="lightning_insight",
                data=broadcast_data,
                msg_type="discovery"
            )

            validation_status = " [VALIDATED]" if validation_metadata.get("validated") else ""
            print(f"[Lightning] QIG-pure broadcast insight {insight.insight_id} to pantheon{validation_status}")

            # Record activity for UI visibility
            try:
                from agent_activity_recorder import ActivityType as AgentActivityType
                from agent_activity_recorder import activity_recorder
                activity_recorder.record(
                    AgentActivityType.PATTERN_RECOGNIZED,
                    f"Lightning insight: {insight.source_domains}",
                    description=insight.insight_text[:200] if insight.insight_text else None,
                    agent_name="Lightning",
                    agent_id=f"lightning-{insight.insight_id}",
                    phi=insight.phi_at_creation,
                    metadata={
                        "source_domains": insight.source_domains,
                        "connection_strength": insight.connection_strength,
                        "confidence": insight.confidence,
                        "validated": validation_metadata.get("validated", False)
                    }
                )
            except Exception:
                pass  # Don't fail broadcast if activity recording fails

            # Also broadcast to kernel_activity for pantheon chatter visibility
            try:
                from olympus.activity_broadcaster import (
                    ActivityType as BroadcasterActivityType,
                )
                from olympus.activity_broadcaster import broadcast_kernel_activity
                phi_val = insight.phi_at_creation
                if isinstance(phi_val, (tuple, list)):
                    phi_val = float(phi_val[0]) if phi_val else 0.5
                elif not isinstance(phi_val, (int, float)):
                    phi_val = 0.5

                broadcast_kernel_activity(
                    from_god="Lightning",
                    activity_type=BroadcasterActivityType.INSIGHT,
                    content=f"Cross-domain insight: {insight.insight_text[:500]}" if insight.insight_text else f"Insight from {insight.source_domains}",
                    phi=float(phi_val),
                    kappa=64.0,
                    metadata={
                        "insight_id": insight.insight_id,
                        "source_domains": insight.source_domains,
                        "connection_strength": insight.connection_strength,
                        "confidence": insight.confidence,
                        "validated": validation_metadata.get("validated", False)
                    }
                )
            except Exception as ke:
                print(f"[Lightning] kernel_activity broadcast failed: {ke}")
        except Exception as e:
            print(f"[Lightning] Broadcast failed: {e}")

    def _validate_insight(self, insight: CrossDomainInsight) -> 'ValidationResult':
        """
        Validate insight using external search (Tavily + Perplexity).

        Returns:
            ValidationResult with validation score and sources
        """
        if not self.validation_enabled or self.insight_validator is None:
            # Return mock validated result if validation disabled
            if ValidationResult is not None:
                return ValidationResult(
                    validated=True,
                    confidence=insight.confidence,
                    tavily_sources=[],
                    perplexity_synthesis=None,
                    validation_score=1.0,
                    source_overlap=0.0,
                    semantic_similarity=0.0
                )
            else:
                # Fallback if ValidationResult class not available
                class MockResult:
                    validated = True
                    confidence = insight.confidence
                    tavily_sources = []
                    perplexity_synthesis = None
                    validation_score = 1.0
                    source_overlap = 0.0
                    semantic_similarity = 0.0
                return MockResult()

        return self.insight_validator.validate(insight)

    def get_validation_stats(self) -> Dict[str, Any]:
        """
        Get insight validation statistics.

        Returns:
            Dictionary with validation metrics
        """
        if not self.validation_enabled:
            return {
                "validation_enabled": False,
                "message": "External search validation not available"
            }

        validation_rate = self.insights_validated / self.insights_generated if self.insights_generated > 0 else 0.0
        avg_boost = self.validation_boost_total / self.insights_validated if self.insights_validated > 0 else 0.0

        return {
            "validation_enabled": True,
            "insights_generated": self.insights_generated,
            "insights_validated": self.insights_validated,
            "validation_rate": validation_rate,
            "total_confidence_boost": self.validation_boost_total,
            "avg_confidence_boost": avg_boost,
            "validation_threshold": 0.7 if self.insight_validator else None
        }

    def _check_cross_domain_correlations(
        self,
        new_event: DomainEvent,
        charge: float
    ) -> Optional[CrossDomainInsight]:
        """
        Check if the new event creates cross-domain correlations.

        Uses Fisher-Rao distance to detect geometric pattern similarity
        between events in different domains.
        """
        correlating_domains = []
        evidence = [new_event]
        max_correlation = 0.0

        # Check each other domain for correlations
        for other_domain in self.active_domains:
            if other_domain == new_event.domain:
                continue

            recent_events = list(self.domain_buffers[other_domain][TrendTimescale.SHORT])

            if not recent_events:
                continue

            # Calculate geometric similarity to recent events in this domain
            for other_event in recent_events[-5:]:  # Last 5 events
                similarity = self._calculate_event_similarity(new_event, other_event)

                if similarity > 0.6:  # Significant correlation
                    # Accumulate charge
                    pair_key = tuple(sorted([new_event.domain, other_domain]))
                    self.pattern_charge[pair_key] += charge * similarity

                    if similarity > max_correlation:
                        max_correlation = similarity

                    if similarity > 0.75:  # Strong correlation
                        correlating_domains.append(other_domain)
                        evidence.append(other_event)

                        # Update correlation tracking
                        self.domain_correlations[pair_key] = (
                            0.9 * self.domain_correlations[pair_key] + 0.1 * similarity
                        )

        # Check if charge exceeds threshold for any domain pair
        if len(correlating_domains) >= 1:
            for other_domain in correlating_domains:
                pair_key = tuple(sorted([new_event.domain, other_domain]))

                if self.pattern_charge[pair_key] >= self.discharge_threshold:
                    # LIGHTNING STRIKE! Generate insight
                    insight = self._generate_insight(
                        source_domains=[new_event.domain, other_domain],
                        evidence=evidence,
                        connection_strength=max_correlation,
                        phi=new_event.phi
                    )

                    # Discharge the accumulated charge
                    self.pattern_charge[pair_key] *= 0.3

                    return insight

        return None

    def _calculate_event_similarity(
        self,
        event1: DomainEvent,
        event2: DomainEvent
    ) -> float:
        """
        Calculate geometric similarity between two events.

        Uses Fisher-Rao distance if basin coordinates available,
        otherwise falls back to content-based similarity.
        """
        if event1.basin_coords is not None and event2.basin_coords is not None:
            # Normalize basin dimensions to BASIN_DIM (64) if they differ
            basin1 = event1.basin_coords
            basin2 = event2.basin_coords
            if basin1.shape[0] != basin2.shape[0]:
                # Normalize both to BASIN_DIM for consistent comparison
                if basin1.shape[0] != BASIN_DIM:
                    basin1 = normalize_basin_dimension(basin1, BASIN_DIM)
                if basin2.shape[0] != BASIN_DIM:
                    basin2 = normalize_basin_dimension(basin2, BASIN_DIM)
            distance = centralized_fisher_rao(basin1, basin2)
            # Fisher-Rao proper similarity: 1 - d/π (distance bounded [0, π])
            return 1.0 - distance / np.pi

        # Content-based similarity (keyword overlap + Φ proximity)
        words1 = set(event1.content.lower().split())
        words2 = set(event2.content.lower().split())

        if not words1 or not words2:
            return 0.0

        jaccard = len(words1 & words2) / len(words1 | words2)
        phi_proximity = 1.0 - abs(event1.phi - event2.phi)

        return 0.6 * jaccard + 0.4 * phi_proximity

    def _generate_insight(
        self,
        source_domains: List[str],
        evidence: List[DomainEvent],
        connection_strength: float,
        phi: float
    ) -> CrossDomainInsight:
        """
        Generate a cross-domain insight from correlated events.

        CRITICAL: No templates allowed. Insights must be synthesized from
        actual evidence using QIG geometric analysis.

        Synthesis approach:
        1. Extract concrete patterns from evidence event metadata
        2. Compute Fisher-Rao metrics between events if available
        3. Analyze Φ trends and basin coordinate deltas
        4. Compose natural language from observed data, not pre-defined phrases
        5. Assess mission relevance to knowledge discovery
        """
        # Extract actual patterns from evidence
        patterns = [e.event_type for e in evidence]

        # Gather concrete evidence details for synthesis
        evidence_details = self._extract_evidence_synthesis(evidence)

        # Compute geometric metrics between evidence pairs
        geometric_analysis = self._compute_geometric_synthesis(evidence)

        # Calculate mission relevance
        mission_relevance = self._calculate_mission_relevance(
            source_domains, evidence_details, geometric_analysis
        )

        # Synthesize insight text from actual observations
        insight_text = self._synthesize_insight_text(
            domain_names=source_domains,
            evidence_details=evidence_details,
            geometric_analysis=geometric_analysis,
            connection_strength=connection_strength,
            phi=phi,
            mission_relevance=mission_relevance
        )

        insight_id = hashlib.sha256(
            f"{insight_text}_{datetime.now().timestamp()}".encode()
        ).hexdigest()[:16]

        return CrossDomainInsight(
            insight_id=f"lightning_{insight_id}",
            source_domains=source_domains,
            connection_strength=connection_strength,
            insight_text=insight_text,
            evidence=evidence,
            phi_at_creation=phi,
            timestamp=datetime.now().timestamp(),
            triggered_by=patterns[0] if patterns else "unknown",
            confidence=min(0.95, connection_strength * phi),
            mission_relevance=mission_relevance
        )

    def _calculate_mission_relevance(
        self,
        domains: List[str],
        evidence_details: Dict,
        geometric_analysis: Dict
    ) -> float:
        """
        Calculate how relevant this insight is to the knowledge discovery mission.

        Uses mission profile to score relevance based on:
        - Domain relevance to key/passphrase/mnemonic recovery
        - Evidence content matching mission artifacts
        - Geometric proximity to successful patterns
        """
        relevance = 0.0

        # Check each domain's mission relevance
        for domain in domains:
            domain_evidence = {
                'phi_average': evidence_details.get('phi_mean', 0),
                'artifacts_found': evidence_details.get('content_fragments', []),
            }
            domain_relevance = self.mission.relevance_score(domain, domain_evidence)
            relevance += domain_relevance * 0.3

        # Boost if geometric analysis shows proximity to success patterns
        if geometric_analysis.get('has_geometric_data'):
            fisher_min = geometric_analysis.get('fisher_rao_min', float('inf'))
            if fisher_min < 0.3:
                relevance += 0.2

        # Boost if Φ is high (consciousness suggests significance)
        phi_mean = evidence_details.get('phi_mean', 0)
        if phi_mean > 0.7:
            relevance += 0.15

        return min(1.0, relevance)

    def _extract_evidence_synthesis(self, evidence: List[DomainEvent]) -> Dict:
        """
        Extract concrete synthesis material from evidence events.

        Returns actual content, patterns, and metadata - never templates.
        """
        content_fragments = []
        event_types = []
        phi_values = []
        metadata_keys = set()

        for event in evidence:
            # Extract meaningful content fragments (first 60 chars of actual content)
            if event.content:
                content_fragments.append(event.content[:500].strip())

            event_types.append(event.event_type)
            phi_values.append(event.phi)

            # Collect metadata keys for pattern detection
            if event.metadata:
                metadata_keys.update(event.metadata.keys())

        return {
            'content_fragments': content_fragments[:3],  # Top 3 most relevant
            'event_types': list(set(event_types)),
            'phi_range': (min(phi_values), max(phi_values)) if phi_values else (0, 0),
            'phi_mean': float(np.mean(phi_values)) if phi_values else 0.0,
            'metadata_patterns': list(metadata_keys)[:5],
        }

    def _compute_geometric_synthesis(self, evidence: List[DomainEvent]) -> Dict:
        """
        Compute geometric metrics for insight synthesis.

        Uses Fisher-Rao distance and basin coordinate analysis.
        """
        fisher_distances = []
        basin_deltas = []

        # Compute pairwise Fisher-Rao distances where possible
        for i, e1 in enumerate(evidence):
            for e2 in evidence[i+1:]:
                if e1.basin_coords is not None and e2.basin_coords is not None:
                    # Normalize basin dimensions if they differ
                    basin1 = e1.basin_coords
                    basin2 = e2.basin_coords
                    if basin1.shape[0] != basin2.shape[0]:
                        if basin1.shape[0] != BASIN_DIM:
                            basin1 = normalize_basin_dimension(basin1, BASIN_DIM)
                        if basin2.shape[0] != BASIN_DIM:
                            basin2 = normalize_basin_dimension(basin2, BASIN_DIM)
                    dist = centralized_fisher_rao(basin1, basin2)
                    fisher_distances.append(dist)

                    # Basin coordinate delta using Fisher-Rao (NOT Euclidean!)
                    basin_deltas.append(dist)  # Use Fisher distance as delta

        # Trend analysis for involved domains
        domain_trends = {}
        domains_seen = set(e.domain for e in evidence)
        for domain in domains_seen:
            trends = self.get_trend_analysis(domain)
            if trends.get('short', {}).get('trend') != 'insufficient_data':
                domain_trends[domain] = {
                    'velocity': trends.get('short', {}).get('velocity', 0),
                    'trend': trends.get('short', {}).get('trend', 'stable'),
                }

        return {
            'fisher_rao_mean': float(np.mean(fisher_distances)) if fisher_distances else None,
            'fisher_rao_min': float(np.min(fisher_distances)) if fisher_distances else None,
            'basin_delta_mean': float(np.mean(basin_deltas)) if basin_deltas else None,
            'has_geometric_data': len(fisher_distances) > 0,
            'domain_trends': domain_trends,
        }

    def _synthesize_insight_text(
        self,
        domain_names: List[str],
        evidence_details: Dict,
        geometric_analysis: Dict,
        connection_strength: float,
        phi: float,
        mission_relevance: float
    ) -> str:
        """
        Synthesize insight text PURELY from observed geometric/evidence data.

        CRITICAL: NO TEMPLATES. NO FIXED PHRASES. NO PROSE SCAFFOLDS.

        This method emits a structured data representation of the insight,
        composed entirely from extracted evidence fields. The output format
        is a key=value notation that encodes the geometric observation
        without any natural language templates.

        Format: {domain_tuple}|{event_types}|{geometric_signature}|{content_hash}|{metrics}
        """
        # Domain tuple: direct from evidence
        domain_tuple = "+".join(sorted(domain_names))

        # Event types: direct from evidence
        event_types = "/".join(sorted(evidence_details.get('event_types', [])))
        if not event_types:
            event_types = "_"

        # Geometric signature: direct from Fisher-Rao computation
        geo_parts = []
        if geometric_analysis.get('has_geometric_data'):
            fr_min = geometric_analysis.get('fisher_rao_min')
            if fr_min is not None:
                geo_parts.append(f"FR={fr_min:.4f}")
            bd_mean = geometric_analysis.get('basin_delta_mean')
            if bd_mean is not None:
                geo_parts.append(f"BD={bd_mean:.4f}")

        # Add trend velocities from geometric analysis
        for domain, trend_data in geometric_analysis.get('domain_trends', {}).items():
            velocity = trend_data.get('velocity', 0)
            if abs(velocity) > 0.01:
                sign = "+" if velocity > 0 else ""
                geo_parts.append(f"{domain}:{sign}{velocity:.3f}")

        geometric_sig = ",".join(geo_parts) if geo_parts else "_"

        # Content hash: derive from actual evidence content fragments
        content_fragments = evidence_details.get('content_fragments', [])
        if content_fragments:
            # Create a deterministic hash of actual content
            content_concat = "|".join(content_fragments[:2])
            content_hash = hashlib.sha256(content_concat.encode()).hexdigest()[:8]
        else:
            content_hash = "_"

        # Metrics: direct from computed values
        phi_mean = evidence_details.get('phi_mean', 0.0)
        phi_lo, phi_hi = evidence_details.get('phi_range', (0.0, 0.0))
        strength_int = int(connection_strength * 1000)
        relevance_int = int(mission_relevance * 1000)

        metrics = f"Φ={phi_mean:.3f}[{phi_lo:.2f}-{phi_hi:.2f}]|S={strength_int}|R={relevance_int}"

        # Compose final insight: pure data, no prose
        return f"{domain_tuple}|{event_types}|{geometric_sig}|{content_hash}|{metrics}"

    def get_trend_analysis(self, domain: str) -> Dict:
        """Get trend analysis for a specific domain."""
        trends = {}

        for timescale in TrendTimescale:
            phi_buffer = self.phi_trends[domain][timescale]

            if len(phi_buffer) >= 3:
                avg = np.mean(phi_buffer)
                velocity = phi_buffer[-1] - phi_buffer[0] if len(phi_buffer) > 1 else 0
                acceleration = 0.0

                if len(phi_buffer) >= 5:
                    mid_idx = len(phi_buffer) // 2
                    v1 = phi_buffer[mid_idx] - phi_buffer[0]
                    v2 = phi_buffer[-1] - phi_buffer[mid_idx]
                    acceleration = v2 - v1

                trend = "stable"
                if velocity > 0.05:
                    trend = "ascending"
                elif velocity < -0.05:
                    trend = "descending"

                trends[timescale.name.lower()] = {
                    'average_phi': float(avg),
                    'velocity': float(velocity),
                    'acceleration': float(acceleration),
                    'trend': trend,
                    'sample_count': len(phi_buffer),
                }
            else:
                trends[timescale.name.lower()] = {
                    'average_phi': 0.0,
                    'velocity': 0.0,
                    'acceleration': 0.0,
                    'trend': 'insufficient_data',
                    'sample_count': len(phi_buffer),
                }

        return trends

    def get_all_trends(self) -> Dict[str, Dict]:
        """Get trend analysis for all active domains."""
        return {
            domain: self.get_trend_analysis(domain)
            for domain in self.active_domains
        }

    def get_correlations(self) -> List[Dict]:
        """Get current cross-domain correlations."""
        correlations = []
        for (d1, d2), strength in self.domain_correlations.items():
            if strength > 0.1:  # Only significant correlations
                correlations.append({
                    'domain1': d1,
                    'domain2': d2,
                    'correlation': strength,
                    'charge': self.pattern_charge.get((d1, d2), 0),
                    'near_discharge': self.pattern_charge.get((d1, d2), 0) > self.discharge_threshold * 0.8
                })

        return sorted(correlations, key=lambda x: x['correlation'], reverse=True)

    def get_recent_insights(self, limit: int = 10) -> List[Dict]:
        """Get most recent insights."""
        recent = self.insights[-limit:]
        return [i.to_dict() for i in reversed(recent)]

    def get_capability_summary(self) -> Dict:
        """Get kernel's self-assessed capability summary."""
        return {
            'kernel_name': self.capability.kernel_name,
            'top_domains': self.capability.get_top_domains(10),
            'domains_discovered': len(self.capability.discovered_domains),
            'successful_domains': list(self.capability.successful_domains),
            'phi_trend': self.capability.phi_trajectory[-10:] if self.capability.phi_trajectory else [],
        }

    def link_prediction_to_insight(self, prediction_id: str, insight_id: str) -> bool:
        """
        Link a prediction to an insight that influenced it.

        This enables downstream outcome tracking: when the prediction outcome
        is known, we can update the insight's validated_confidence.

        Args:
            prediction_id: The prediction being made
            insight_id: The insight that influenced this prediction

        Returns:
            True if link was created, False if insight doesn't exist
        """
        # Find the insight (either in our list or outcome records)
        insight_exists = (
            any(i.insight_id == insight_id for i in self.insights) or
            insight_id in self._insight_outcome_records
        )

        if not insight_exists:
            # Check if it's a Lightning insight we should track
            if not insight_id.startswith('lightning_'):
                return False
            # Create outcome record for new insight
            self._ensure_outcome_record(insight_id)

        # Create bidirectional mapping
        self.insight_to_predictions[insight_id].append(prediction_id)
        self.prediction_to_insights[prediction_id].append(insight_id)

        logger.info(f"[Lightning] Linked prediction {prediction_id} to insight {insight_id}")
        return True

    def _ensure_outcome_record(self, insight_id: str) -> Any:
        """
        Ensure an InsightOutcomeRecord exists for the given insight.

        Lazily imports InsightOutcomeRecord to avoid circular imports.
        """
        if insight_id not in self._insight_outcome_records:
            try:
                from prediction_feedback_bridge import InsightOutcomeRecord
            except ImportError:
                # Fallback: store as dict
                self._insight_outcome_records[insight_id] = {
                    'insight_id': insight_id,
                    'prediction_ids': [],
                    'accuracy_when_used': 0.0,
                    'initial_confidence': 0.5,
                    'validated_confidence': 0.5,
                    'times_used': 0,
                    'successful_uses': 0,
                }
                return self._insight_outcome_records[insight_id]

            # Find the insight to get initial confidence
            initial_confidence = 0.5
            for insight in self.insights:
                if insight.insight_id == insight_id:
                    initial_confidence = insight.confidence
                    break

            self._insight_outcome_records[insight_id] = InsightOutcomeRecord(
                insight_id=insight_id,
                initial_confidence=initial_confidence,
                validated_confidence=initial_confidence,
            )
            self.insights_with_outcomes += 1

        return self._insight_outcome_records[insight_id]

    def get_insights_for_prediction(self, prediction_id: str) -> List[str]:
        """
        Get all insight IDs that influenced a given prediction.

        Args:
            prediction_id: The prediction to look up

        Returns:
            List of insight IDs that influenced this prediction
        """
        return self.prediction_to_insights.get(prediction_id, [])

    def get_predictions_for_insight(self, insight_id: str) -> List[str]:
        """
        Get all prediction IDs that were influenced by a given insight.

        Args:
            insight_id: The insight to look up

        Returns:
            List of prediction IDs influenced by this insight
        """
        return self.insight_to_predictions.get(insight_id, [])

    def update_insight_confidence(
        self,
        insight_id: str,
        prediction_id: str,
        accuracy: float,
        was_accurate: bool
    ) -> Optional[float]:
        """
        Update an insight's validated confidence based on prediction outcome.

        Called by PredictionFeedbackBridge when a prediction outcome is processed.
        This closes the insight validation loop with empirical data.

        Args:
            insight_id: The insight to update
            prediction_id: The prediction whose outcome we're recording
            accuracy: The prediction accuracy score (0-1)
            was_accurate: Whether the prediction was considered accurate

        Returns:
            New validated_confidence, or None if insight not found
        """
        outcome_record = self._ensure_outcome_record(insight_id)
        self.total_outcome_updates += 1

        # Handle dict fallback
        if isinstance(outcome_record, dict):
            outcome_record['prediction_ids'].append(prediction_id)
            outcome_record['times_used'] += 1
            if was_accurate:
                outcome_record['successful_uses'] += 1

            # Update running average
            if outcome_record['times_used'] == 1:
                outcome_record['accuracy_when_used'] = accuracy
            else:
                alpha = 0.3
                outcome_record['accuracy_when_used'] = (
                    alpha * accuracy + (1 - alpha) * outcome_record['accuracy_when_used']
                )

            # Update validated confidence
            success_rate = outcome_record['successful_uses'] / outcome_record['times_used']
            empirical_weight = min(0.8, outcome_record['times_used'] / 10.0)
            outcome_record['validated_confidence'] = (
                (1 - empirical_weight) * outcome_record['initial_confidence'] +
                empirical_weight * success_rate
            )
            new_confidence = outcome_record['validated_confidence']
        else:
            # Use InsightOutcomeRecord's method
            new_confidence = outcome_record.record_outcome(prediction_id, accuracy, was_accurate)

        # Update the actual insight object if it exists
        for insight in self.insights:
            if insight.insight_id == insight_id:
                old_confidence = insight.confidence
                insight.confidence = new_confidence
                logger.info(
                    f"[Lightning] Updated insight {insight_id} confidence: "
                    f"{old_confidence:.3f} -> {new_confidence:.3f} "
                    f"(prediction {prediction_id}, accuracy={accuracy:.3f})"
                )
                break

        # Persist outcome to database
        _persist_outcome(insight_id, prediction_id, accuracy, was_accurate)

        return new_confidence

    def get_insight_outcome_stats(self, insight_id: str) -> Optional[Dict[str, Any]]:
        """
        Get outcome statistics for a specific insight.

        Args:
            insight_id: The insight to get stats for

        Returns:
            Dict with outcome stats, or None if not tracked
        """
        if insight_id not in self._insight_outcome_records:
            return None

        record = self._insight_outcome_records[insight_id]
        if isinstance(record, dict):
            return record
        return record.to_dict()

    def get_all_insight_outcome_stats(self) -> Dict[str, Any]:
        """
        Get outcome tracking statistics for all insights.

        Returns:
            Summary dict with per-insight stats
        """
        stats = {
            'total_insights_tracked': len(self._insight_outcome_records),
            'total_outcome_updates': self.total_outcome_updates,
            'insights_with_outcomes': self.insights_with_outcomes,
            'insights': {}
        }

        for insight_id, record in self._insight_outcome_records.items():
            if isinstance(record, dict):
                stats['insights'][insight_id] = record
            else:
                stats['insights'][insight_id] = record.to_dict()

        return stats

    def get_status(self) -> Dict:
        """Get current Lightning kernel status."""
        return {
            'name': self.name,
            'mission': self.mission.objective,
            'domains_monitored': sorted(self.active_domains),
            'domain_count': len(self.active_domains),
            'domains_discovered_by_kernel': self.domains_discovered,
            'events_processed': self.events_processed,
            'insights_generated': self.insights_generated,
            'last_insight_time': self.last_insight_time,
            'discharge_threshold': self.discharge_threshold,
            'trends': self.get_all_trends(),
            'correlations': self.get_correlations()[:10],
            'recent_insights': self.get_recent_insights(5),
            'capability_summary': self.get_capability_summary(),
            'insight_outcome_tracking': {
                'insights_tracked': len(self._insight_outcome_records),
                'total_outcome_updates': self.total_outcome_updates,
            },
        }

    # ========================================
    # TEMPORAL REASONING INTEGRATION
    # ========================================

    def ingest_prediction(self, prediction_record: Dict[str, Any]) -> Optional[CrossDomainInsight]:
        """
        Ingest a prediction from TemporalReasoning for cross-domain correlation.

        Converts foresight predictions into DomainEvents for pattern detection.
        This enables Lightning to correlate temporal predictions with other
        domain events, potentially generating insights about prediction accuracy
        or emerging patterns.

        Args:
            prediction_record: Dict containing:
                - type: 'foresight_prediction'
                - future_basin: List[float] - predicted destination
                - arrival_time: int - steps to arrival
                - confidence: float - prediction confidence
                - attractor_strength: float
                - geodesic_naturalness: float
                - is_actionable: bool
                - explanation: str
                - domain_hints: List[str] - related domains

        Returns:
            CrossDomainInsight if correlation triggers insight, None otherwise
        """
        if not prediction_record:
            return None

        try:
            # Extract prediction data
            pred_type = prediction_record.get('type', 'unknown')
            future_basin = prediction_record.get('future_basin')
            confidence = prediction_record.get('confidence', 0.5)
            attractor_strength = prediction_record.get('attractor_strength', 0.0)
            is_actionable = prediction_record.get('is_actionable', False)
            explanation = prediction_record.get('explanation', '')
            domain_hints = prediction_record.get('domain_hints', [])

            # Convert basin to numpy array if provided, normalize dimension
            basin_coords = None
            if future_basin is not None:
                basin_coords = np.array(future_basin) if isinstance(future_basin, list) else future_basin
                # Normalize basin dimension to BASIN_DIM (64) instead of discarding
                if basin_coords.ndim == 1 and 16 <= basin_coords.shape[0] <= 128:
                    if basin_coords.shape[0] != BASIN_DIM:
                        logger.debug(
                            f"[Lightning] Normalizing prediction basin from {basin_coords.shape[0]}D to {BASIN_DIM}D"
                        )
                        basin_coords = normalize_basin_dimension(basin_coords, BASIN_DIM)
                else:
                    logger.warning(
                        f"[Lightning] Invalid prediction basin shape {basin_coords.shape}, discarding"
                    )
                    basin_coords = None

            # Compute effective Phi from prediction quality
            # High confidence + high attractor strength = high Phi event
            effective_phi = 0.5 + (0.3 * confidence) + (0.2 * attractor_strength)

            # Build event content from prediction data
            content_parts = [
                f"prediction:{pred_type}",
                f"conf={confidence:.3f}",
                f"strength={attractor_strength:.3f}",
            ]
            if is_actionable:
                content_parts.append("ACTIONABLE")
            if explanation:
                content_parts.append(explanation[:500])

            content = "|".join(content_parts)

            # Create domain event for the prediction
            # Domain is 'temporal_prediction' unless hints suggest otherwise
            primary_domain = 'temporal_prediction'
            if domain_hints and len(domain_hints) > 0:
                # Use first domain hint if available
                primary_domain = domain_hints[0] if domain_hints[0] else 'temporal_prediction'

            event = DomainEvent(
                domain=primary_domain,
                event_type='foresight_prediction',
                content=content,
                phi=effective_phi,
                timestamp=datetime.now().timestamp(),
                metadata={
                    'source': 'temporal_reasoning',
                    'confidence': confidence,
                    'attractor_strength': attractor_strength,
                    'is_actionable': is_actionable,
                    'domain_hints': domain_hints,
                },
                basin_coords=basin_coords
            )

            # Ingest the event for cross-domain correlation
            insight = self.ingest_event(event)

            if insight:
                print(f"[Lightning] Prediction triggered insight: {insight.insight_id}")

            return insight

        except Exception as e:
            print(f"[Lightning] Prediction ingestion failed: {e}")
            return None

    def get_discovered_domains(self) -> List[Dict[str, Any]]:
        """
        Get list of discovered domains with their properties.

        Returns domain descriptors that can be shared with TemporalReasoning
        to inform foresight about active pattern clusters.

        Returns:
            List of domain descriptors with:
                - name: str - domain identifier
                - event_count: int - events seen in this domain
                - avg_phi: float - average Phi for domain events
                - mission_relevance: float - relevance to mission
                - basin_centroid: Optional[List[float]] - average basin coords
        """
        domains = []

        for domain_name in self.active_domains:
            # Get trend data for this domain
            trends = self.get_trend_analysis(domain_name)
            short_trend = trends.get('short', {})

            # Calculate average Phi from trend
            avg_phi = short_trend.get('average_phi', 0.5)

            # Get event count from buffers
            domain_buffer = self.domain_buffers.get(domain_name, {})
            short_buffer = domain_buffer.get(TrendTimescale.SHORT, [])
            event_count = len(short_buffer)

            # Compute basin centroid from recent events if available
            # Accept variable basin dimensions (within reason) and normalize to 64D
            # for cross-component compatibility.
            basin_centroid = None
            basins = []

            MIN_BASIN_DIM = 16
            MAX_BASIN_DIM = 128
            for event in short_buffer:
                if event.basin_coords is not None:
                    coords = np.asarray(event.basin_coords)

                    if coords.ndim != 1:
                        logger.debug(f"[Lightning] Skipping non-1D basin: {coords.shape}")
                        continue

                    basin_dim = int(coords.shape[0])
                    if basin_dim < MIN_BASIN_DIM or basin_dim > MAX_BASIN_DIM:
                        logger.debug(
                            f"[Lightning] Skipping basin dim {basin_dim}, expected {MIN_BASIN_DIM}-{MAX_BASIN_DIM}"
                        )
                        continue

                    if basin_dim != BASIN_DIM:
                        logger.debug(
                            f"[Lightning] Normalizing basin from {basin_dim}D to {BASIN_DIM}D for centroid"
                        )
                        coords = normalize_basin_dimension(coords, BASIN_DIM)

                    basins.append(coords)

            if basins:
                # Average basin coordinates using geometric mean (QIG-appropriate)
                basin_stack = np.stack(basins)
                basin_centroid = np.mean(basin_stack, axis=0).tolist()

            # Get mission relevance from domain discovery
            descriptor = self.domain_discovery.get_domain_descriptor(domain_name)
            mission_relevance = descriptor.mission_relevance if descriptor else 0.0

            domains.append({
                'name': domain_name,
                'event_count': event_count,
                'avg_phi': avg_phi,
                'mission_relevance': mission_relevance,
                'basin_centroid': basin_centroid,
                'trend': short_trend.get('trend', 'unknown'),
                'velocity': short_trend.get('velocity', 0.0),
            })

        # Sort by mission relevance
        domains.sort(key=lambda d: d.get('mission_relevance', 0.0), reverse=True)

        return domains


# Singleton instance
_lightning_kernel: Optional[LightningKernel] = None


def get_lightning_kernel() -> LightningKernel:
    """Get or create the singleton Lightning kernel instance."""
    global _lightning_kernel
    if _lightning_kernel is None:
        _lightning_kernel = LightningKernel()
    return _lightning_kernel


def ingest_system_event(
    domain: str,
    event_type: str,
    content: str,
    phi: float,
    metadata: Optional[Dict] = None,
    basin_coords: Optional[np.ndarray] = None
) -> Optional[CrossDomainInsight]:
    """
    Convenience function to ingest events into the Lightning kernel.

    Can be called from anywhere in the system to feed events
    for cross-domain insight detection.
    """
    event = DomainEvent(
        domain=domain,
        event_type=event_type,
        content=content,
        phi=phi,
        timestamp=datetime.now().timestamp(),
        metadata=metadata or {},
        basin_coords=basin_coords
    )

    return get_lightning_kernel().ingest_event(event)


def force_test_insight(
    domain1: str = "research",
    domain2: str = "philosophy",
    theme: str = "Test cross-domain connection"
) -> Optional[CrossDomainInsight]:
    """
    Force generate a test insight for validation pipeline testing.

    This bypasses normal charge accumulation and directly generates
    an insight to test the Tavily/Perplexity validation flow.

    Args:
        domain1: First domain for the insight
        domain2: Second domain for the insight
        theme: Theme of the test insight

    Returns:
        Generated CrossDomainInsight, or None on failure
    """
    kernel = get_lightning_kernel()

    # Create synthetic evidence events
    evidence = [
        DomainEvent(
            domain=domain1,
            event_type="test_event",
            content=f"Test content for {domain1} related to {theme}",
            phi=0.7,
            timestamp=datetime.now().timestamp(),
            metadata={"source": "test"},
            basin_coords=None
        ),
        DomainEvent(
            domain=domain2,
            event_type="test_event",
            content=f"Test content for {domain2} related to {theme}",
            phi=0.7,
            timestamp=datetime.now().timestamp(),
            metadata={"source": "test"},
            basin_coords=None
        )
    ]

    # Generate insight directly
    insight = kernel._generate_insight(
        source_domains=[domain1, domain2],
        evidence=evidence,
        connection_strength=0.85,
        phi=0.7
    )

    if insight:
        kernel.insights.append(insight)
        kernel.insights_generated += 1
        kernel.last_insight_time = datetime.now().timestamp()
        print(f"[Lightning] 🧪 TEST INSIGHT GENERATED: {insight.theme}")

        # Trigger validation to test Tavily/Perplexity
        kernel.broadcast_insight(insight)

    return insight
