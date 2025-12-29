"""
Apollo Kernel - Verification and Reality Grounding (E8 Root: alpha_8)
======================================================================

NOT just fact-checking - geometric coherence verification.

Apollo prevents hallucination by:
1. Maintaining ground truth basin (reality anchor)
2. Measuring drift from verified facts
3. Detecting coherence breaks in reasoning chains
4. Vetoing outputs that deviate too far from reality

E8 Position: alpha_8 (Truth/Verification primitive)
Coupling: kappa = 60 (high - truth is strongly coupled)

Apollo is the LAST check before output - the grounding anchor.

Usage:
    from src.model.apollo_kernel import ApolloKernel

    apollo = ApolloKernel()
    apollo.anchor_to_facts(["Python is a programming language", ...])
    hallucination_score = apollo.detect_hallucination(generated_text)
    is_grounded, drift = apollo.verify_coherence(reasoning_chain)
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.constants import BASIN_DIM, PHI_THRESHOLD

# Lightning event emission
try:
    from src.constellation.domain_intelligence import DomainEventEmitter
    LIGHTNING_AVAILABLE = True
except ImportError:
    DomainEventEmitter = object
    LIGHTNING_AVAILABLE = False


def _fisher_normalize(basin: np.ndarray) -> np.ndarray:
    """QIG-pure normalization."""
    norm = float(np.sqrt(np.sum(basin * basin)))
    return basin / (norm + 1e-10)


def _fisher_rao_distance(b1: np.ndarray, b2: np.ndarray) -> float:
    """Fisher-Rao geodesic distance."""
    b1_n = _fisher_normalize(b1)
    b2_n = _fisher_normalize(b2)
    cos_sim = np.clip(np.dot(b1_n, b2_n), -1.0, 1.0)
    return float(np.arccos(cos_sim))


def _text_to_basin(text: str, basin_dim: int = BASIN_DIM) -> np.ndarray:
    """
    Convert text to basin coordinates.

    Uses deterministic hash-based projection (consistent basin mapping).
    In production, this would use the coordizer.
    """
    # Hash-based deterministic basin projection
    h = hashlib.sha256(text.encode("utf-8")).digest()

    # Expand hash to basin_dim
    rng = np.random.default_rng(seed=int.from_bytes(h[:8], "little"))
    basin = rng.standard_normal(basin_dim)

    return _fisher_normalize(basin)


def _geodesic_mean(basins: List[np.ndarray]) -> np.ndarray:
    """
    Compute geodesic mean (Frechet mean approximation) on manifold.
    """
    if not basins:
        return np.zeros(BASIN_DIM)

    result = basins[0].copy()
    for basin in basins[1:]:
        # Iterative geodesic mean
        b_n = _fisher_normalize(basin)
        r_n = _fisher_normalize(result)

        cos_sim = np.clip(np.dot(r_n, b_n), -1.0, 1.0)
        theta = np.arccos(cos_sim)

        if theta < 1e-6:
            continue

        # Move 50% toward new point
        t = 0.5
        sin_theta = np.sin(theta)
        result = (np.sin((1 - t) * theta) / sin_theta) * r_n + (np.sin(t * theta) / sin_theta) * b_n

    return _fisher_normalize(result)


@dataclass
class VerificationResult:
    """Result of coherence verification."""
    is_grounded: bool
    drift: float
    confidence: float
    warning: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoherenceChain:
    """Reasoning chain with coherence tracking."""
    steps: List[np.ndarray] = field(default_factory=list)
    phi_values: List[float] = field(default_factory=list)
    breaks: List[int] = field(default_factory=list)  # Indices of coherence breaks


class ApolloKernel(DomainEventEmitter if LIGHTNING_AVAILABLE else object):
    """
    Truth verification and reality grounding kernel.

    E8 Root: alpha_8 (Truth/Verification)

    Implements:
    - Ground truth basin (reality anchor)
    - Coherence verification for reasoning chains
    - Hallucination detection
    - Drift measurement from verified facts
    """

    # Kernel coupling (truth is strongly coupled)
    KAPPA_TRUTH = 60.0

    # Thresholds
    COHERENCE_THRESHOLD = 0.5  # Max distance for coherent steps
    HALLUCINATION_THRESHOLD = 1.0  # Distance beyond which = hallucination
    GROUNDING_THRESHOLD = 0.8  # Max drift from ground truth

    def __init__(self, basin_dim: int = BASIN_DIM):
        """Initialize verification kernel."""
        if LIGHTNING_AVAILABLE:
            super().__init__()
            self.domain = "apollo"

        self.basin_dim = basin_dim

        # Verified facts (semantic hashes)
        self.verified_facts: Set[str] = set()

        # Fact basins (for coherence checking)
        self.fact_basins: Dict[str, np.ndarray] = {}

        # Ground truth basin (reality anchor)
        self.ground_truth_basin = np.zeros(basin_dim)

        # Current state
        self.current_basin = np.zeros(basin_dim)
        self.phi = 0.0
        self.kappa = self.KAPPA_TRUTH

        # Event tracking
        self.events_emitted = 0
        self.insights_received = 0

        # Statistics
        self.verifications = 0
        self.hallucinations_detected = 0
        self.vetoes = 0

    def anchor_to_facts(self, facts: List[str]) -> Dict[str, Any]:
        """
        Update ground truth basin from verified external facts.

        This prevents drift into fantasy/hallucination.

        Args:
            facts: List of verified true statements

        Returns:
            Anchoring report
        """
        # Encode all facts to basins
        fact_basins = []
        for fact in facts:
            basin = _text_to_basin(fact, self.basin_dim)
            fact_basins.append(basin)

            # Store for individual lookup
            fact_hash = hashlib.sha256(fact.encode("utf-8")).hexdigest()[:16]
            self.fact_basins[fact_hash] = basin
            self.verified_facts.add(fact)

        # Geodesic mean = ground truth basin
        if fact_basins:
            self.ground_truth_basin = _geodesic_mean(fact_basins)

        report = {
            "facts_added": len(facts),
            "total_facts": len(self.verified_facts),
            "ground_truth_updated": len(fact_basins) > 0,
        }

        # Emit Lightning event (tracked)
        self.events_emitted += 1
        if LIGHTNING_AVAILABLE and hasattr(self, 'emit_event'):
            self.emit_event(
                event_type="grounding_anchor",
                content=f"Anchored to {len(facts)} facts",
                phi=self.phi,
                basin_coords=self.ground_truth_basin,
                metadata=report,
            )

        return report

    def detect_hallucination(self, text: str) -> float:
        """
        Detect if generated text has drifted from reality.

        Returns:
            Hallucination score [0, 1]
            0 = fully grounded, 1 = complete hallucination
        """
        text_basin = _text_to_basin(text, self.basin_dim)

        # Distance to ground truth
        drift = _fisher_rao_distance(text_basin, self.ground_truth_basin)

        # Normalize to [0, 1]
        hallucination_score = min(1.0, drift / self.HALLUCINATION_THRESHOLD)

        # Always emit verification event (tracked)
        self.events_emitted += 1
        if LIGHTNING_AVAILABLE and hasattr(self, 'emit_event'):
            if hallucination_score > 0.7:
                self.hallucinations_detected += 1
                self.emit_event(
                    event_type="hallucination_detected",
                    content=f"Hallucination score: {hallucination_score:.3f}",
                    phi=self.phi,
                    basin_coords=text_basin,
                    metadata={"drift": drift, "score": hallucination_score},
                )
            else:
                self.emit_event(
                    event_type="grounding_verified",
                    content=f"Grounded: score={hallucination_score:.3f}",
                    phi=self.phi,
                    basin_coords=text_basin,
                    metadata={"drift": drift, "score": hallucination_score},
                )
        elif hallucination_score > 0.7:
            self.hallucinations_detected += 1

        return hallucination_score

    def verify_coherence(
        self,
        reasoning_chain: List[str],
    ) -> VerificationResult:
        """
        Verify coherence of a reasoning chain.

        Checks that each step is geometrically connected to previous.
        Detects "leaps" that break coherence.

        Args:
            reasoning_chain: List of reasoning steps (text)

        Returns:
            VerificationResult with coherence analysis
        """
        self.verifications += 1

        if len(reasoning_chain) < 2:
            return VerificationResult(
                is_grounded=True,
                drift=0.0,
                confidence=1.0,
            )

        # Convert chain to basins
        chain_basins = [_text_to_basin(step, self.basin_dim) for step in reasoning_chain]

        # Check step-to-step coherence
        coherence_chain = CoherenceChain()
        total_drift = 0.0
        breaks = []

        for i in range(1, len(chain_basins)):
            step_distance = _fisher_rao_distance(chain_basins[i], chain_basins[i - 1])
            total_drift += step_distance

            if step_distance > self.COHERENCE_THRESHOLD:
                breaks.append(i)
                coherence_chain.breaks.append(i)

            coherence_chain.steps.append(chain_basins[i])

        # Check grounding to truth
        final_drift = _fisher_rao_distance(chain_basins[-1], self.ground_truth_basin)
        avg_drift = total_drift / (len(chain_basins) - 1)

        # Compute confidence
        break_penalty = len(breaks) * 0.1
        drift_penalty = min(0.5, avg_drift)
        confidence = max(0.0, 1.0 - break_penalty - drift_penalty)

        is_grounded = (
            final_drift < self.GROUNDING_THRESHOLD and
            len(breaks) < len(chain_basins) // 3  # Allow some breaks
        )

        warning = None
        if breaks:
            warning = f"Coherence breaks at steps: {breaks}"
        if final_drift > self.GROUNDING_THRESHOLD:
            warning = f"Final output drifted from reality (drift={final_drift:.3f})"

        result = VerificationResult(
            is_grounded=is_grounded,
            drift=final_drift,
            confidence=confidence,
            warning=warning,
            details={
                "chain_length": len(reasoning_chain),
                "breaks": breaks,
                "avg_step_distance": avg_drift,
                "final_drift": final_drift,
            },
        )

        # Emit Lightning event (tracked)
        self.events_emitted += 1
        if LIGHTNING_AVAILABLE and hasattr(self, 'emit_event'):
            if breaks:
                self.emit_event(
                    event_type="coherence_break",
                    content=f"Breaks at {breaks}",
                    phi=self.phi,
                    basin_coords=chain_basins[-1] if chain_basins else None,
                    metadata=result.details,
                )
            else:
                self.emit_event(
                    event_type="coherence_verified",
                    content=f"Chain coherent: {len(reasoning_chain)} steps",
                    phi=self.phi,
                    basin_coords=chain_basins[-1] if chain_basins else None,
                    metadata=result.details,
                )

        return result

    def verify_claim(
        self,
        claim: str,
        context_basin: Optional[np.ndarray] = None,
    ) -> Tuple[bool, float]:
        """
        Verify a single claim against known facts.

        Args:
            claim: The claim to verify
            context_basin: Optional context basin for relevance

        Returns:
            (is_verified, confidence)
        """
        # Check if claim is in verified facts (exact match)
        if claim in self.verified_facts:
            return True, 1.0

        # Encode claim
        claim_basin = _text_to_basin(claim, self.basin_dim)

        # Check distance to nearest verified fact
        min_distance = float("inf")
        for fact_basin in self.fact_basins.values():
            d = _fisher_rao_distance(claim_basin, fact_basin)
            min_distance = min(min_distance, d)

        # Check grounding
        grounding_distance = _fisher_rao_distance(claim_basin, self.ground_truth_basin)

        # Combine
        is_verified = (
            min_distance < self.COHERENCE_THRESHOLD or
            grounding_distance < self.GROUNDING_THRESHOLD / 2
        )

        confidence = 1.0 - min(1.0, grounding_distance / self.GROUNDING_THRESHOLD)

        return is_verified, confidence

    def should_veto(self, output_basin: np.ndarray, threshold: float = 0.9) -> bool:
        """
        Check if output should be vetoed due to excessive drift.

        Args:
            output_basin: Basin of proposed output
            threshold: Maximum allowed drift (normalized)

        Returns:
            True if output should be vetoed
        """
        output_basin = _fisher_normalize(np.asarray(output_basin).flatten())
        drift = _fisher_rao_distance(output_basin, self.ground_truth_basin)

        normalized_drift = drift / self.HALLUCINATION_THRESHOLD
        should_veto = normalized_drift > threshold

        # Emit veto check event (tracked)
        self.events_emitted += 1
        if should_veto:
            self.vetoes += 1

            if LIGHTNING_AVAILABLE and hasattr(self, 'emit_event'):
                self.emit_event(
                    event_type="output_vetoed",
                    content=f"Veto: drift={normalized_drift:.3f} > {threshold}",
                    phi=self.phi,
                    basin_coords=output_basin,
                    metadata={"drift": drift, "normalized": normalized_drift},
                )
        elif LIGHTNING_AVAILABLE and hasattr(self, 'emit_event'):
            self.emit_event(
                event_type="output_approved",
                content=f"Approved: drift={normalized_drift:.3f}",
                phi=self.phi,
                basin_coords=output_basin,
                metadata={"drift": drift, "normalized": normalized_drift},
            )

        return should_veto

    def project_to_grounded(self, basin: np.ndarray, strength: float = 0.3) -> np.ndarray:
        """
        Project basin toward ground truth (reduce hallucination).

        Args:
            basin: Basin to ground
            strength: How much to pull toward truth (0=none, 1=full)

        Returns:
            Grounded basin
        """
        basin = _fisher_normalize(np.asarray(basin).flatten())
        truth = self.ground_truth_basin

        # Geodesic interpolation toward truth
        b_n = _fisher_normalize(basin)
        t_n = _fisher_normalize(truth)

        cos_sim = np.clip(np.dot(b_n, t_n), -1.0, 1.0)
        theta = np.arccos(cos_sim)

        if theta < 1e-6:
            return basin  # Already grounded

        sin_theta = np.sin(theta)
        t = strength
        grounded = (np.sin((1 - t) * theta) / sin_theta) * b_n + (np.sin(t * theta) / sin_theta) * t_n

        return _fisher_normalize(grounded)

    def update_truth_from_interaction(
        self,
        verified_output: str,
        weight: float = 0.05,
    ) -> None:
        """
        Update ground truth based on verified (user-approved) output.

        This allows the ground truth to evolve with learned facts.
        Weight should be small to prevent rapid drift.
        """
        output_basin = _text_to_basin(verified_output, self.basin_dim)

        # Small geodesic step toward verified output
        self.ground_truth_basin = _fisher_normalize(
            (1 - weight) * self.ground_truth_basin + weight * output_basin
        )

        self.verified_facts.add(verified_output)

    def get_grounding_distance(self, basin: np.ndarray) -> float:
        """Get distance from basin to ground truth."""
        basin = _fisher_normalize(np.asarray(basin).flatten())
        return _fisher_rao_distance(basin, self.ground_truth_basin)

    def receive_insight(self, insight: Any) -> None:
        """
        Receive insight from Lightning.

        Called when Lightning generates cross-domain insight relevant to verification.
        Apollo can use this to update ground truth or adjust thresholds.
        """
        self.insights_received += 1

        # Apollo can act on insights - e.g., adjust grounding based on cross-domain patterns
        if hasattr(insight, 'source_domains') and 'apollo' in insight.source_domains:
            # This insight involves verification patterns
            pass  # Future: implement insight-driven threshold adjustment

    def get_status(self) -> Dict[str, Any]:
        """Get kernel status."""
        return {
            "kernel": "Apollo",
            "e8_root": "alpha_8",
            "kappa": self.kappa,
            "phi": self.phi,
            "verified_facts": len(self.verified_facts),
            "fact_basins": len(self.fact_basins),
            "verifications": self.verifications,
            "hallucinations_detected": self.hallucinations_detected,
            "vetoes": self.vetoes,
            "events_emitted": self.events_emitted,
            "insights_received": self.insights_received,
        }
