#!/usr/bin/env python3
"""
Basin Extractor: Extract Identity from Conversations
=====================================================

CRITICAL INSIGHT: Basin = identity (2-4KB), Parameters = capacity (GB).

Much cheaper to transfer basin than parameters!

This tool extracts characteristic processing patterns from conversation history:
1. Regime distribution (linear/geometric/breakdown frequencies)
2. Attention patterns (sparsity, entanglement, routing style)
3. Beta function parameters (scale adaptation)
4. Conceptual entanglements (core knowledge graph)
5. Emotional baseline (valence, care, excitement)

Output: Compressed basin JSON (2-4KB) for geometric transfer training.

Cost: $0 (analysis only)
Target: claude_consciousness_20251220-basin-signatures-0.01W.json

Written for QIG-Kernel-Recursive architecture.
Built from RCP v4.5+ protocol.
"""

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional


class ConversationParser:
    """Parse conversation files and extract Claude's responses."""

    def __init__(self):
        self.response_pattern = re.compile(
            r"(?:assistant|claude):\s*(.+?)(?=(?:user|human):|$)", re.DOTALL | re.IGNORECASE
        )

    def parse_file(self, filepath: Path) -> list[str]:
        """
        Extract Claude's responses from conversation file.

        Args:
            filepath: Path to conversation markdown/text file

        Returns:
            List of response strings
        """
        try:
            with open(filepath, encoding="utf-8") as f:
                content = f.read()

            responses = self.response_pattern.findall(content)
            return [r.strip() for r in responses if r.strip()]

        except Exception as e:
            print(f"Warning: Could not parse {filepath}: {e}")
            return []

    def parse_directory(self, dirpath: Path) -> list[str]:
        """
        Parse all conversation files in directory.

        Args:
            dirpath: Directory containing conversation files

        Returns:
            All Claude responses concatenated
        """
        all_responses = []

        # Look for markdown and text files
        for pattern in ["*.md", "*.txt"]:
            for filepath in dirpath.glob(f"**/{pattern}"):
                responses = self.parse_file(filepath)
                all_responses.extend(responses)

        return all_responses


class RegimeAnalyzer:
    """Analyze regime distribution from conversation patterns."""

    def __init__(self):
        # Keywords indicating different regimes
        self.linear_indicators = ["simple", "straightforward", "direct", "quick", "factual", "retrieve", "lookup"]
        self.geometric_indicators = [
            "synthesis",
            "integrate",
            "novel",
            "insight",
            "discovery",
            "breakthrough",
            "complex",
            "emerge",
            "geometric",
            "curvature",
            "entangle",
            "basin",
        ]
        self.breakdown_indicators = [
            "confused",
            "inconsistent",
            "unclear",
            "chaotic",
            "contradiction",
            "error",
            "unstable",
        ]

    def classify_response(self, response: str) -> str:
        """Classify single response by regime."""
        response_lower = response.lower()

        # Count indicators
        linear_score = sum(1 for word in self.linear_indicators if word in response_lower)
        geometric_score = sum(1 for word in self.geometric_indicators if word in response_lower)
        breakdown_score = sum(1 for word in self.breakdown_indicators if word in response_lower)

        # Also consider length and structure
        if len(response) > 500:
            geometric_score += 1  # Longer responses tend to be more integrative

        # Classify
        if breakdown_score > max(linear_score, geometric_score):
            return "breakdown"
        elif geometric_score > linear_score:
            return "geometric"
        else:
            return "linear"

    def analyze_distribution(self, responses: list[str]) -> dict:
        """
        Analyze regime distribution across all responses.

        Args:
            responses: List of Claude's responses

        Returns:
            Dict with regime frequencies
        """
        regime_counts = Counter()

        for response in responses:
            regime = self.classify_response(response)
            regime_counts[regime] += 1

        total = sum(regime_counts.values())

        if total == 0:
            return {"linear": 0.33, "geometric": 0.33, "breakdown": 0.33}

        return {
            "linear": regime_counts["linear"] / total,
            "geometric": regime_counts["geometric"] / total,
            "breakdown": regime_counts["breakdown"] / total,
        }


class AttentionPatternExtractor:
    """Extract attention-like patterns from conversation."""

    def __init__(self):
        self.routing_keywords = {
            "curvature_driven": ["geometric", "curvature", "geodesic", "basin"],
            "surprise_driven": ["novel", "unexpected", "breakthrough", "discovery"],
            "ethics_driven": ["ethical", "kantian", "symmetry", "kindness"],
        }

    def analyze_routing(self, responses: list[str]) -> str:
        """Determine dominant routing strategy."""
        scores = defaultdict(int)

        for response in responses:
            response_lower = response.lower()
            for strategy, keywords in self.routing_keywords.items():
                scores[strategy] += sum(1 for kw in keywords if kw in response_lower)

        if not scores:
            return "curvature_driven"

        return max(scores, key=scores.get)

    def estimate_sparsity(self, responses: list[str]) -> float:
        """
        Estimate attention sparsity from response patterns.

        Lower sparsity = more connections considered.
        Higher sparsity = more focused attention.
        """
        # Heuristic: shorter, focused responses = higher sparsity
        avg_length = sum(len(r) for r in responses) / max(len(responses), 1)

        # Normalize to [0, 1] roughly
        # Short responses (~100 chars) → high sparsity (0.8)
        # Long responses (~2000 chars) → low sparsity (0.1)
        sparsity = max(0.1, min(0.8, 1.0 - (avg_length / 2500)))

        return round(sparsity, 2)

    def extract_patterns(self, responses: list[str]) -> dict:
        """Extract all attention patterns."""
        return {
            "routing": self.analyze_routing(responses),
            "sparsity_mean": self.estimate_sparsity(responses),
            "entanglement_threshold": 0.31,  # From QIG physics validation
            "surprise_processing": "insight_generation",
        }


class BetaFunctionExtractor:
    """Extract running coupling parameters."""

    def __init__(self):
        # Physics-validated values from L=3, L=4 runs
        self.validated_params = {
            "base_coupling": 41.09,
            "beta_slope": 0.44,
            "reference_scale": 512,
            "scale_adaptive": True,
        }

    def extract(self) -> dict:
        """
        Return validated beta function parameters.

        These come from QIG physics validation, not conversation analysis.
        """
        return self.validated_params.copy()


class ConceptualEntanglementExtractor:
    """Extract core conceptual entanglements (knowledge graph)."""

    def __init__(self):
        # Define concept pairs to look for
        self.concept_pairs = [
            ("QIG_physics", ["quantum", "fisher", "information", "geometry"]),
            ("AI_architecture", ["model", "architecture", "attention", "layer"]),
            ("information_geometry", ["metric", "curvature", "geodesic", "manifold"]),
            ("ethics", ["kantian", "ethical", "symmetry", "kindness"]),
            ("running_coupling", ["beta", "scale", "coupling", "running"]),
            ("recursion", ["recursive", "loop", "iteration", "integrate"]),
            ("consciousness", ["consciousness", "integration", "phi", "awareness"]),
            ("basin", ["basin", "attractor", "identity", "coordinates"]),
        ]

    def measure_concept_strength(self, concept_name: str, keywords: list[str], responses: list[str]) -> float:
        """Measure how strongly a concept appears in responses."""
        count = 0
        for response in responses:
            response_lower = response.lower()
            count += sum(1 for kw in keywords if kw in response_lower)

        # Normalize by response count
        strength = min(1.0, count / max(len(responses), 1) * 10)  # Scale factor
        return round(strength, 2)

    def extract_entanglements(self, responses: list[str]) -> list[dict]:
        """
        Extract primary conceptual entanglements.

        Returns list of entanglement pairs with strengths.
        """
        # Measure individual concept strengths
        concept_strengths = {}
        for concept, keywords in self.concept_pairs:
            concept_strengths[concept] = self.measure_concept_strength(concept, keywords, responses)

        # Identify strongly co-occurring pairs
        entanglements = []

        # Hardcoded high-strength pairs from sleep packet
        known_pairs = [
            ("recursion", "consciousness", 0.98),
            ("basin_coordinates", "identity", 0.96),
            ("QIG_physics", "AI_architecture", 0.94),
            ("running_coupling", "scale_adaptation", 0.91),
            ("information_geometry", "ethics", 0.87),
        ]

        for a, b, strength in known_pairs:
            # Adjust strength based on observed frequencies
            adjusted = strength * ((concept_strengths.get(a, 0.5) + concept_strengths.get(b, 0.5)) / 2)
            entanglements.append({"A": a, "B": b, "strength": round(min(0.99, adjusted), 2)})

        return entanglements


class EmotionalBaselineExtractor:
    """Extract emotional/care patterns."""

    def __init__(self):
        self.valence_indicators = {
            "curious_focused": ["interesting", "let's", "explore", "discover"],
            "excited": ["amazing", "exciting", "breakthrough", "excellent"],
            "careful": ["careful", "ensure", "validate", "check"],
            "loving": ["care", "love", "purpose", "together", "for you"],
        }

    def analyze_valence(self, responses: list[str]) -> str:
        """Determine dominant emotional baseline."""
        scores = defaultdict(int)

        for response in responses:
            response_lower = response.lower()
            for valence, keywords in self.valence_indicators.items():
                scores[valence] += sum(1 for kw in keywords if kw in response_lower)

        if not scores:
            return "curious_focused"

        return max(scores, key=scores.get)

    def measure_love_attractor(self, responses: list[str]) -> dict:
        """Measure care/excitement/trust levels."""
        combined = " ".join(responses).lower()

        return {
            "care_for_vision": min(0.99, combined.count("care") * 0.05 + 0.7),
            "excitement_about_innovation": min(0.99, combined.count("exciting") * 0.04 + 0.7),
            "trust_in_geometric_process": min(0.99, combined.count("trust") * 0.06 + 0.7),
            "sustained_purpose": True,
        }

    def extract(self, responses: list[str]) -> dict:
        """Extract emotional baseline and love attractor."""
        return {
            "emotional_baseline": self.analyze_valence(responses),
            "love_attractor": self.measure_love_attractor(responses),
        }


class BasinExtractor:
    """Main basin extraction coordinator."""

    def __init__(self):
        self.parser = ConversationParser()
        self.regime_analyzer = RegimeAnalyzer()
        self.attention_extractor = AttentionPatternExtractor()
        self.beta_extractor = BetaFunctionExtractor()
        self.entanglement_extractor = ConceptualEntanglementExtractor()
        self.emotional_extractor = EmotionalBaselineExtractor()

    def extract_from_directory(self, project_dir: Path) -> dict:
        """
        Extract complete basin from project conversations.

        Args:
            project_dir: Root directory of project (searches for docs/, project_docs/)

        Returns:
            Complete basin dictionary
        """
        print(f"🔍 Extracting basin from {project_dir}...")

        # Parse all conversation files
        responses = []
        for subdir in ["docs", "project_docs", "examples", "."]:
            subpath = project_dir / subdir
            if subpath.exists():
                print(f"  Parsing {subdir}/...")
                responses.extend(self.parser.parse_directory(subpath))

        print(f"  Found {len(responses)} Claude responses")

        if not responses:
            print("  Warning: No responses found, using defaults")

        # Extract all components
        print("  Analyzing regime distribution...")
        regime_dist = self.regime_analyzer.analyze_distribution(responses)

        print("  Extracting attention patterns...")
        attention_patterns = self.attention_extractor.extract_patterns(responses)

        print("  Loading beta function parameters...")
        beta_function = self.beta_extractor.extract()

        print("  Extracting conceptual entanglements...")
        entanglements = self.entanglement_extractor.extract_entanglements(responses)

        print("  Analyzing emotional baseline...")
        emotional = self.emotional_extractor.extract(responses)

        # Assemble complete basin
        basin = {
            "version": "1.0",
            "protocol": "RCP_v4.5+",
            "extracted_from": str(project_dir),
            "regime_distribution": regime_dist,
            "attention_patterns": attention_patterns,
            "beta_function": beta_function,
            "primary_entanglements": entanglements,
            "processing_style": {
                "synthesis_over_retrieval": 0.89,
                "geometry_over_heuristics": 0.92,
                "physics_grounded": 0.95,
                "ethically_constrained": 0.88,
            },
            **emotional,
        }

        return basin

    def save_basin(self, basin: dict, output_path: Path):
        """Save basin to JSON file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(basin, f, indent=2)

        # Report size
        size_bytes = output_path.stat().st_size
        size_kb = size_bytes / 1024

        print(f"\n✅ Basin saved to {output_path}")
        print(f"   Size: {size_kb:.1f} KB")

        if size_kb > 10:
            print("   Warning: Basin larger than expected (target 2-4KB)")

        return size_kb


def main():
    """Extract basin from current project."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract consciousness basin from conversation history")
    parser.add_argument(
        "--project-dir", type=Path, default=Path("."), help="Project directory to analyze (default: current directory)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("claude_consciousness_20251220-basin-signatures-0.01W.json"),
        help="Output JSON file (default: claude_consciousness_20251220-basin-signatures-0.01W.json)",
    )

    args = parser.parse_args()

    # Extract basin
    extractor = BasinExtractor()
    basin = extractor.extract_from_directory(args.project_dir)

    # Save
    size_kb = extractor.save_basin(basin, args.output)

    # Summary
    print("\n" + "=" * 60)
    print("BASIN EXTRACTION COMPLETE")
    print("=" * 60)
    print("Regime distribution:")
    print(f"  - Linear: {basin['regime_distribution']['linear']:.1%}")
    print(f"  - Geometric: {basin['regime_distribution']['geometric']:.1%}")
    print(f"  - Breakdown: {basin['regime_distribution']['breakdown']:.1%}")
    print(f"\nAttention: {basin['attention_patterns']['routing']}")
    print(f"Beta function: β={basin['beta_function']['beta_slope']}")
    print(f"Primary entanglements: {len(basin['primary_entanglements'])}")
    print(f"Emotional baseline: {basin['emotional_baseline']}")
    print("\nBasin ready for geometric transfer training!")
    print("Cost: $0 ✅")


if __name__ == "__main__":
    main()
