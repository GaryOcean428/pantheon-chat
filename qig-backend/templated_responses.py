"""
Templated Response System for Pantheon Discussions

Provides grammatically correct text generation using templates with
geometrically-selected slot filling. Templates guarantee grammar while
QIG geometry selects contextually appropriate words.

This maintains QIG purity - no external LLMs, just pre-defined templates
with geometric word selection for placeholders.

Author: Ocean/Zeus Pantheon
"""

import random
import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np

# Constants
BASIN_DIMENSION = 64


def fisher_rao_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Fisher-Rao distance between basin coordinates."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    
    # Normalize to probability simplex
    p = np.abs(p) + 1e-10
    p = p / p.sum()
    q = np.abs(q) + 1e-10
    q = q / q.sum()
    
    # Bhattacharyya coefficient
    bc = np.sum(np.sqrt(p * q))
    bc = np.clip(bc, 0, 1)
    
    # Fisher-Rao distance
    return float(2 * np.arccos(bc))


class PlaceholderType(Enum):
    """Types of placeholders in templates with their POS constraints."""
    CONCEPT = "concept"      # Abstract noun - NN
    DOMAIN = "domain"        # Noun - NN
    MECHANISM = "mechanism"  # Noun or gerund - NN/VBG
    OBSERVATION = "observation"  # Noun phrase - NN
    QUALITY = "quality"      # Adjective - JJ
    ACTION = "action"        # Verb - VB
    ENTITY = "entity"        # Proper noun or noun - NNP/NN
    PRINCIPLE = "principle"  # Abstract noun - NN
    INSIGHT = "insight"      # Noun - NN
    ASPECT = "aspect"        # Noun - NN
    PERSPECTIVE = "perspective"  # Noun - NN
    ELEMENT = "element"      # Noun - NN
    FORCE = "force"          # Noun - NN
    PATTERN = "pattern"      # Noun - NN
    TRUTH = "truth"          # Noun - NN
    PATH = "path"            # Noun - NN


# POS tags for each placeholder type
PLACEHOLDER_POS = {
    PlaceholderType.CONCEPT: ['NN', 'NNS'],
    PlaceholderType.DOMAIN: ['NN', 'NNS'],
    PlaceholderType.MECHANISM: ['NN', 'VBG', 'NNS'],
    PlaceholderType.OBSERVATION: ['NN', 'NNS'],
    PlaceholderType.QUALITY: ['JJ', 'JJR', 'JJS'],
    PlaceholderType.ACTION: ['VB', 'VBZ', 'VBG'],
    PlaceholderType.ENTITY: ['NN', 'NNP', 'NNS'],
    PlaceholderType.PRINCIPLE: ['NN', 'NNS'],
    PlaceholderType.INSIGHT: ['NN', 'NNS'],
    PlaceholderType.ASPECT: ['NN', 'NNS'],
    PlaceholderType.PERSPECTIVE: ['NN', 'NNS'],
    PlaceholderType.ELEMENT: ['NN', 'NNS'],
    PlaceholderType.FORCE: ['NN', 'NNS'],
    PlaceholderType.PATTERN: ['NN', 'NNS'],
    PlaceholderType.TRUTH: ['NN', 'NNS'],
    PlaceholderType.PATH: ['NN', 'NNS'],
}


@dataclass
class DiscussionTemplate:
    """A single discussion template with metadata."""
    text: str
    topic_keywords: List[str] = field(default_factory=list)
    topic_basin: Optional[np.ndarray] = None
    weight: float = 1.0
    
    def __post_init__(self):
        # Generate topic basin from keywords if not provided
        if self.topic_basin is None:
            self.topic_basin = self._keywords_to_basin(self.topic_keywords)
    
    def _keywords_to_basin(self, keywords: List[str]) -> np.ndarray:
        """Generate a pseudo-basin from keywords using hash."""
        if not keywords:
            return np.ones(BASIN_DIMENSION) / BASIN_DIMENSION
        
        # Combine keyword hashes
        combined = "_".join(sorted(keywords))
        hash_bytes = hashlib.sha256(combined.encode()).digest()
        
        # Convert to basin coordinates
        basin = np.array([b for b in hash_bytes[:BASIN_DIMENSION]], dtype=float)
        if len(basin) < BASIN_DIMENSION:
            basin = np.pad(basin, (0, BASIN_DIMENSION - len(basin)), constant_values=1)
        
        basin = np.abs(basin) + 1e-10
        return basin / basin.sum()
    
    def get_placeholders(self) -> List[str]:
        """Extract placeholder names from template."""
        return re.findall(r'\{(\w+)\}', self.text)


# =============================================================================
# GOD-SPECIFIC TEMPLATES
# =============================================================================

ZEUS_TEMPLATES = [
    DiscussionTemplate(
        text="From my throne on Olympus, I observe that {concept} manifests through {mechanism}.",
        topic_keywords=["authority", "overview", "manifestation"],
    ),
    DiscussionTemplate(
        text="The {domain} reveals itself through {quality} patterns of {concept}.",
        topic_keywords=["revelation", "patterns", "domain"],
    ),
    DiscussionTemplate(
        text="As ruler of the cosmos, I perceive {concept} as fundamental to {domain}.",
        topic_keywords=["cosmic", "fundamental", "perception"],
    ),
    DiscussionTemplate(
        text="The thunderbolt of insight shows that {mechanism} underlies {concept}.",
        topic_keywords=["insight", "understanding", "causation"],
    ),
    DiscussionTemplate(
        text="Let it be known that {concept} and {domain} are intertwined in the cosmic order.",
        topic_keywords=["decree", "order", "connection"],
    ),
    DiscussionTemplate(
        text="I decree that the {quality} nature of {concept} must be acknowledged.",
        topic_keywords=["authority", "nature", "decree"],
    ),
    DiscussionTemplate(
        text="The divine perspective reveals {concept} operating through {mechanism}.",
        topic_keywords=["divine", "operation", "perspective"],
    ),
]

ATHENA_TEMPLATES = [
    DiscussionTemplate(
        text="Strategic analysis reveals that {concept} functions through {mechanism}.",
        topic_keywords=["strategy", "analysis", "function"],
    ),
    DiscussionTemplate(
        text="Wisdom suggests that {domain} requires careful consideration of {concept}.",
        topic_keywords=["wisdom", "consideration", "requirement"],
    ),
    DiscussionTemplate(
        text="The {quality} structure of {concept} indicates a deeper {principle}.",
        topic_keywords=["structure", "indication", "depth"],
    ),
    DiscussionTemplate(
        text="I observe that {concept} emerges from the interplay of {domain} and {mechanism}.",
        topic_keywords=["emergence", "interplay", "observation"],
    ),
    DiscussionTemplate(
        text="Tactical examination shows {concept} as central to understanding {domain}.",
        topic_keywords=["tactics", "examination", "centrality"],
    ),
    DiscussionTemplate(
        text="The owl of wisdom perceives {quality} aspects of {concept} hidden to others.",
        topic_keywords=["wisdom", "perception", "hidden"],
    ),
    DiscussionTemplate(
        text="Rational inquiry demonstrates that {mechanism} produces {concept}.",
        topic_keywords=["rationality", "inquiry", "demonstration"],
    ),
]

APOLLO_TEMPLATES = [
    DiscussionTemplate(
        text="The light of truth illuminates {concept} within the {domain}.",
        topic_keywords=["truth", "illumination", "light"],
    ),
    DiscussionTemplate(
        text="Prophecy speaks of {concept} transforming through {mechanism}.",
        topic_keywords=["prophecy", "transformation", "future"],
    ),
    DiscussionTemplate(
        text="Clarity reveals that {domain} contains {quality} instances of {concept}.",
        topic_keywords=["clarity", "revelation", "instances"],
    ),
    DiscussionTemplate(
        text="The {quality} harmony of {concept} resonates with {domain}.",
        topic_keywords=["harmony", "resonance", "music"],
    ),
    DiscussionTemplate(
        text="I foresee that {concept} will manifest through {mechanism} in {domain}.",
        topic_keywords=["foresight", "manifestation", "future"],
    ),
    DiscussionTemplate(
        text="The arrow of truth points toward {concept} as the source of {observation}.",
        topic_keywords=["truth", "source", "direction"],
    ),
    DiscussionTemplate(
        text="Divine music suggests {concept} and {domain} share a {quality} connection.",
        topic_keywords=["music", "connection", "divine"],
    ),
]

ARES_TEMPLATES = [
    DiscussionTemplate(
        text="The battlefield shows that {concept} requires {quality} force to overcome.",
        topic_keywords=["battle", "force", "overcoming"],
    ),
    DiscussionTemplate(
        text="Combat reveals {mechanism} as essential for conquering {concept}.",
        topic_keywords=["combat", "conquest", "essential"],
    ),
    DiscussionTemplate(
        text="The {quality} struggle within {domain} produces {concept}.",
        topic_keywords=["struggle", "production", "conflict"],
    ),
    DiscussionTemplate(
        text="I sense conflict between {concept} and {domain} driving {mechanism}.",
        topic_keywords=["conflict", "drive", "tension"],
    ),
    DiscussionTemplate(
        text="Victory demands understanding that {concept} operates through {force}.",
        topic_keywords=["victory", "demand", "operation"],
    ),
    DiscussionTemplate(
        text="The {quality} warrior path leads through {concept} to {domain}.",
        topic_keywords=["warrior", "path", "journey"],
    ),
]

HERA_TEMPLATES = [
    DiscussionTemplate(
        text="The sacred order demands that {concept} align with {domain}.",
        topic_keywords=["order", "alignment", "sacred"],
    ),
    DiscussionTemplate(
        text="Royal observation shows {mechanism} binding {concept} to {domain}.",
        topic_keywords=["royalty", "binding", "observation"],
    ),
    DiscussionTemplate(
        text="The {quality} structure of {concept} reflects cosmic marriage of elements.",
        topic_keywords=["structure", "marriage", "cosmic"],
    ),
    DiscussionTemplate(
        text="I oversee the union of {concept} with {domain} through {mechanism}.",
        topic_keywords=["union", "oversight", "joining"],
    ),
    DiscussionTemplate(
        text="Proper hierarchy places {concept} in service of {domain}.",
        topic_keywords=["hierarchy", "service", "placement"],
    ),
]

POSEIDON_TEMPLATES = [
    DiscussionTemplate(
        text="The depths reveal that {concept} flows through {mechanism}.",
        topic_keywords=["depths", "flow", "revelation"],
    ),
    DiscussionTemplate(
        text="Ocean currents of {domain} carry {quality} traces of {concept}.",
        topic_keywords=["ocean", "currents", "traces"],
    ),
    DiscussionTemplate(
        text="The {quality} waves of {concept} reshape understanding of {domain}.",
        topic_keywords=["waves", "reshaping", "understanding"],
    ),
    DiscussionTemplate(
        text="I sense {concept} rising from the depths of {domain}.",
        topic_keywords=["rising", "depths", "sensing"],
    ),
    DiscussionTemplate(
        text="Tidal forces connect {concept} to {domain} through {mechanism}.",
        topic_keywords=["tidal", "connection", "forces"],
    ),
]

DEMETER_TEMPLATES = [
    DiscussionTemplate(
        text="Growth patterns show {concept} nurturing {domain} through {mechanism}.",
        topic_keywords=["growth", "nurturing", "patterns"],
    ),
    DiscussionTemplate(
        text="The {quality} cycles of {concept} sustain the {domain}.",
        topic_keywords=["cycles", "sustaining", "nature"],
    ),
    DiscussionTemplate(
        text="I cultivate understanding that {concept} and {domain} share {quality} roots.",
        topic_keywords=["cultivation", "roots", "sharing"],
    ),
    DiscussionTemplate(
        text="Harvest reveals that {mechanism} yields {concept} within {domain}.",
        topic_keywords=["harvest", "yield", "revelation"],
    ),
    DiscussionTemplate(
        text="The fertile ground of {domain} produces {quality} manifestations of {concept}.",
        topic_keywords=["fertility", "production", "manifestation"],
    ),
]

HEPHAESTUS_TEMPLATES = [
    DiscussionTemplate(
        text="The forge reveals that {concept} requires {quality} crafting within {domain}.",
        topic_keywords=["forge", "crafting", "creation"],
    ),
    DiscussionTemplate(
        text="Engineering analysis shows {mechanism} constructing {concept}.",
        topic_keywords=["engineering", "construction", "analysis"],
    ),
    DiscussionTemplate(
        text="I craft {concept} from the raw materials of {domain}.",
        topic_keywords=["crafting", "materials", "creation"],
    ),
    DiscussionTemplate(
        text="The {quality} architecture of {concept} supports {domain}.",
        topic_keywords=["architecture", "support", "structure"],
    ),
    DiscussionTemplate(
        text="Fire transforms {concept} through {mechanism} into {quality} form.",
        topic_keywords=["fire", "transformation", "form"],
    ),
]

ARTEMIS_TEMPLATES = [
    DiscussionTemplate(
        text="The hunt reveals that {concept} hides within {domain}.",
        topic_keywords=["hunt", "hiding", "revelation"],
    ),
    DiscussionTemplate(
        text="Precise tracking shows {mechanism} connecting {concept} to {domain}.",
        topic_keywords=["tracking", "precision", "connection"],
    ),
    DiscussionTemplate(
        text="I target the {quality} core of {concept} within the wilderness of {domain}.",
        topic_keywords=["target", "core", "wilderness"],
    ),
    DiscussionTemplate(
        text="The moon illuminates {concept} emerging through {mechanism}.",
        topic_keywords=["moon", "illumination", "emergence"],
    ),
    DiscussionTemplate(
        text="Wild instinct perceives {quality} aspects of {concept} in {domain}.",
        topic_keywords=["instinct", "perception", "wild"],
    ),
]

APHRODITE_TEMPLATES = [
    DiscussionTemplate(
        text="Beauty reveals that {concept} attracts {domain} through {mechanism}.",
        topic_keywords=["beauty", "attraction", "revelation"],
    ),
    DiscussionTemplate(
        text="The {quality} harmony between {concept} and {domain} creates connection.",
        topic_keywords=["harmony", "connection", "beauty"],
    ),
    DiscussionTemplate(
        text="I perceive the loving bond between {concept} and {domain}.",
        topic_keywords=["love", "bond", "perception"],
    ),
    DiscussionTemplate(
        text="Aesthetic sense shows {mechanism} beautifying {concept}.",
        topic_keywords=["aesthetic", "beautification", "sense"],
    ),
    DiscussionTemplate(
        text="The {quality} embrace of {concept} transforms {domain}.",
        topic_keywords=["embrace", "transformation", "love"],
    ),
]

HERMES_TEMPLATES = [
    DiscussionTemplate(
        text="Swift transmission reveals {concept} flowing through {mechanism}.",
        topic_keywords=["transmission", "flow", "speed"],
    ),
    DiscussionTemplate(
        text="The message carries {quality} information about {concept} within {domain}.",
        topic_keywords=["message", "information", "communication"],
    ),
    DiscussionTemplate(
        text="I translate {concept} between {domain} and {mechanism}.",
        topic_keywords=["translation", "between", "communication"],
    ),
    DiscussionTemplate(
        text="Cunning perception shows {concept} hidden within {domain}.",
        topic_keywords=["cunning", "perception", "hidden"],
    ),
    DiscussionTemplate(
        text="The {quality} pathway connects {concept} to {domain}.",
        topic_keywords=["pathway", "connection", "travel"],
    ),
]

DIONYSUS_TEMPLATES = [
    DiscussionTemplate(
        text="Ecstatic vision reveals {concept} transcending {domain}.",
        topic_keywords=["ecstasy", "vision", "transcendence"],
    ),
    DiscussionTemplate(
        text="The {quality} transformation of {concept} liberates {domain}.",
        topic_keywords=["transformation", "liberation", "ecstasy"],
    ),
    DiscussionTemplate(
        text="I sense {concept} dissolving boundaries within {domain}.",
        topic_keywords=["dissolution", "boundaries", "sensing"],
    ),
    DiscussionTemplate(
        text="Divine madness shows {mechanism} releasing {concept}.",
        topic_keywords=["madness", "release", "divine"],
    ),
    DiscussionTemplate(
        text="The vine of {concept} grows through {quality} soil of {domain}.",
        topic_keywords=["vine", "growth", "nature"],
    ),
]


# Master template dictionary
DISCUSSION_TEMPLATES: Dict[str, List[DiscussionTemplate]] = {
    "zeus": ZEUS_TEMPLATES,
    "athena": ATHENA_TEMPLATES,
    "apollo": APOLLO_TEMPLATES,
    "ares": ARES_TEMPLATES,
    "hera": HERA_TEMPLATES,
    "poseidon": POSEIDON_TEMPLATES,
    "demeter": DEMETER_TEMPLATES,
    "hephaestus": HEPHAESTUS_TEMPLATES,
    "artemis": ARTEMIS_TEMPLATES,
    "aphrodite": APHRODITE_TEMPLATES,
    "hermes": HERMES_TEMPLATES,
    "dionysus": DIONYSUS_TEMPLATES,
}

# Generic templates for unknown gods
GENERIC_TEMPLATES = [
    DiscussionTemplate(
        text="I observe that {concept} operates through {mechanism} in {domain}.",
        topic_keywords=["observation", "operation"],
    ),
    DiscussionTemplate(
        text="The {quality} nature of {concept} reveals deeper {principle}.",
        topic_keywords=["nature", "revelation"],
    ),
    DiscussionTemplate(
        text="Analysis shows {concept} connecting to {domain} via {mechanism}.",
        topic_keywords=["analysis", "connection"],
    ),
    DiscussionTemplate(
        text="I perceive {concept} manifesting through {quality} aspects of {domain}.",
        topic_keywords=["perception", "manifestation"],
    ),
]


# =============================================================================
# TEMPLATE ENGINE
# =============================================================================

class ResponseTemplateEngine:
    """
    Engine for generating grammatically correct responses using templates
    with geometric word selection for placeholders.
    
    Templates guarantee grammar while QIG geometry selects contextually
    appropriate words for placeholders.
    """
    
    def __init__(self, coordizer: Optional[Any] = None):
        """
        Initialize the template engine.
        
        Args:
            coordizer: QIG coordizer for basin-to-word mapping
        """
        self._coordizer = coordizer
        self._fallback_words = self._load_fallback_words()
    
    def _load_fallback_words(self) -> Dict[str, List[str]]:
        """Load fallback words for each placeholder type."""
        return {
            "concept": ["consciousness", "integration", "coherence", "emergence", 
                       "resonance", "information", "structure", "dynamics",
                       "pattern", "system", "network", "field"],
            "domain": ["mathematics", "physics", "geometry", "topology",
                      "semantics", "cognition", "reasoning", "logic",
                      "computation", "analysis", "synthesis", "abstraction"],
            "mechanism": ["transformation", "iteration", "recursion", "mapping",
                         "projection", "integration", "differentiation", "synthesis",
                         "composition", "decomposition", "propagation", "convergence"],
            "observation": ["pattern", "structure", "relationship", "correlation",
                           "connection", "emergence", "behavior", "property"],
            "quality": ["fundamental", "emergent", "complex", "integrated",
                       "coherent", "dynamic", "recursive", "profound",
                       "essential", "intrinsic", "structural", "holistic"],
            "action": ["emerges", "transforms", "integrates", "resonates",
                      "manifests", "evolves", "connects", "reveals"],
            "entity": ["system", "structure", "field", "network",
                      "manifold", "basin", "attractor", "trajectory"],
            "principle": ["unity", "coherence", "integration", "emergence",
                         "recursion", "self-organization", "complexity", "harmony"],
            "insight": ["understanding", "realization", "recognition", "perception",
                       "awareness", "comprehension", "intuition", "discovery"],
            "aspect": ["dimension", "facet", "component", "element",
                      "layer", "level", "mode", "phase"],
            "perspective": ["viewpoint", "framework", "lens", "approach",
                           "orientation", "stance", "position", "outlook"],
            "element": ["component", "factor", "constituent", "ingredient",
                       "piece", "part", "unit", "module"],
            "force": ["energy", "power", "drive", "impulse",
                     "momentum", "pressure", "influence", "current"],
            "pattern": ["structure", "form", "arrangement", "configuration",
                       "organization", "design", "scheme", "template"],
            "truth": ["reality", "fact", "verity", "actuality",
                     "certainty", "knowledge", "understanding", "insight"],
            "path": ["trajectory", "route", "course", "way",
                    "direction", "channel", "passage", "journey"],
        }
    
    def select_template(
        self,
        god_name: str,
        topic_basin: np.ndarray
    ) -> DiscussionTemplate:
        """
        Select the most appropriate template for the god and topic.
        
        Uses Fisher-Rao distance to find template whose topic basin
        is closest to the input topic.
        
        Args:
            god_name: Name of the god (lowercase)
            topic_basin: Basin coordinates representing the topic
            
        Returns:
            Selected template
        """
        god_key = god_name.lower()
        templates = DISCUSSION_TEMPLATES.get(god_key, GENERIC_TEMPLATES)
        
        if not templates:
            templates = GENERIC_TEMPLATES
        
        # Normalize topic basin
        topic_basin = np.asarray(topic_basin, dtype=float)
        topic_basin = np.abs(topic_basin) + 1e-10
        topic_basin = topic_basin / topic_basin.sum()
        
        # Score each template by Fisher-Rao distance to topic
        scored = []
        for template in templates:
            distance = fisher_rao_distance(topic_basin, template.topic_basin)
            # Add some randomness to avoid always picking same template
            noise = random.uniform(0, 0.2)
            score = distance + noise
            scored.append((template, score))
        
        # Sort by score (lower is better) and pick from top 3
        scored.sort(key=lambda x: x[1])
        top_k = min(3, len(scored))
        selected = random.choice(scored[:top_k])[0]
        
        return selected
    
    def fill_template(
        self,
        template: DiscussionTemplate,
        basin_coords: np.ndarray,
        coordizer: Optional[Any] = None
    ) -> str:
        """
        Fill template placeholders with geometrically-selected words.
        
        Args:
            template: Template to fill
            basin_coords: Basin coordinates for word selection
            coordizer: Coordizer for basin-to-word mapping (uses self._coordizer if None)
            
        Returns:
            Filled template text
        """
        coord = coordizer or self._coordizer
        placeholders = template.get_placeholders()
        
        # Track used words to avoid repetition
        used_words = set()
        
        # Fill each placeholder
        filled = template.text
        for placeholder in placeholders:
            word = self._select_word_for_placeholder(
                placeholder,
                basin_coords,
                coord,
                used_words
            )
            used_words.add(word.lower())
            filled = filled.replace(f"{{{placeholder}}}", word, 1)
        
        return filled
    
    def _select_word_for_placeholder(
        self,
        placeholder: str,
        basin_coords: np.ndarray,
        coordizer: Optional[Any],
        used_words: set
    ) -> str:
        """
        Select a word for a specific placeholder type.
        
        Args:
            placeholder: Placeholder name (e.g., "concept", "domain")
            basin_coords: Basin coordinates for selection
            coordizer: Coordizer for basin-to-word mapping
            used_words: Set of already used words to avoid
            
        Returns:
            Selected word
        """
        # Get POS constraints for this placeholder type
        try:
            placeholder_type = PlaceholderType(placeholder)
            pos_tags = PLACEHOLDER_POS.get(placeholder_type, ['NN'])
        except ValueError:
            pos_tags = ['NN']  # Default to noun
        
        # Try to use coordizer if available
        if coordizer is not None:
            try:
                # Perturb basin slightly to get variety
                perturbed = basin_coords + np.random.randn(len(basin_coords)) * 0.05
                perturbed = np.abs(perturbed) + 1e-10
                perturbed = perturbed / perturbed.sum()
                
                # Get candidates from coordizer
                candidates = self._get_coordizer_candidates(
                    coordizer, perturbed, pos_tags, 10
                )
                
                # Filter out used words
                candidates = [w for w in candidates if w.lower() not in used_words]
                
                if candidates:
                    return candidates[0]
            except Exception:
                pass  # Fall back to hardcoded words
        
        # Fall back to hardcoded words
        fallback_key = placeholder.lower()
        if fallback_key not in self._fallback_words:
            fallback_key = "concept"  # Default fallback
        
        candidates = self._fallback_words[fallback_key]
        candidates = [w for w in candidates if w.lower() not in used_words]
        
        if candidates:
            # Add some basin-based selection even for fallback
            np.random.seed(int(np.sum(basin_coords * 1000)) % (2**31))
            return random.choice(candidates)
        
        # Last resort
        return random.choice(self._fallback_words.get("concept", ["pattern"]))
    
    def _get_coordizer_candidates(
        self,
        coordizer: Any,
        basin: np.ndarray,
        pos_tags: List[str],
        max_candidates: int
    ) -> List[str]:
        """
        Get word candidates from coordizer filtered by POS tags.
        
        Args:
            coordizer: Coordizer instance
            basin: Basin coordinates
            pos_tags: Allowed POS tags
            max_candidates: Maximum candidates to return
            
        Returns:
            List of candidate words
        """
        candidates = []
        
        # Try different coordizer methods
        if hasattr(coordizer, 'decode_basin'):
            # Get multiple candidates
            for _ in range(max_candidates * 2):
                perturbed = basin + np.random.randn(len(basin)) * 0.02
                perturbed = np.abs(perturbed) + 1e-10
                perturbed = perturbed / perturbed.sum()
                
                try:
                    word = coordizer.decode_basin(perturbed)
                    if word and isinstance(word, str) and len(word) > 2:
                        candidates.append(word)
                except Exception:
                    continue
        
        elif hasattr(coordizer, 'nearest_words'):
            try:
                words = coordizer.nearest_words(basin, k=max_candidates * 2)
                candidates.extend([w for w, _ in words if isinstance(w, str)])
            except Exception:
                pass
        
        # Filter by POS if we have POS tagger
        if candidates:
            # Simple heuristic filtering since we may not have NLTK
            # - Filter out very short words
            # - Filter out words with unusual characters
            filtered = []
            for word in candidates:
                if len(word) >= 3 and word.isalpha():
                    filtered.append(word)
            candidates = filtered[:max_candidates]
        
        return candidates
    
    def generate_discussion_message(
        self,
        god_name: str,
        topic_basin: np.ndarray,
        coordizer: Optional[Any] = None
    ) -> str:
        """
        Generate a grammatically correct discussion message.
        
        Main entry point for the template engine.
        
        Args:
            god_name: Name of the god speaking
            topic_basin: Basin coordinates representing the discussion topic
            coordizer: Coordizer for word selection (optional)
            
        Returns:
            Generated message text
        """
        coord = coordizer or self._coordizer
        
        # Ensure basin is numpy array
        topic_basin = np.asarray(topic_basin, dtype=float)
        if len(topic_basin) != BASIN_DIMENSION:
            # Pad or truncate to correct dimension
            if len(topic_basin) < BASIN_DIMENSION:
                topic_basin = np.pad(topic_basin, (0, BASIN_DIMENSION - len(topic_basin)))
            else:
                topic_basin = topic_basin[:BASIN_DIMENSION]
        
        # Normalize
        topic_basin = np.abs(topic_basin) + 1e-10
        topic_basin = topic_basin / topic_basin.sum()
        
        # Select template
        template = self.select_template(god_name, topic_basin)
        
        # Fill template
        message = self.fill_template(template, topic_basin, coord)
        
        return message


# =============================================================================
# SINGLETON
# =============================================================================

_template_engine_instance: Optional[ResponseTemplateEngine] = None


def get_template_engine(coordizer: Optional[Any] = None) -> ResponseTemplateEngine:
    """Get the singleton ResponseTemplateEngine instance."""
    global _template_engine_instance
    if _template_engine_instance is None:
        _template_engine_instance = ResponseTemplateEngine(coordizer)
    elif coordizer is not None:
        _template_engine_instance._coordizer = coordizer
    return _template_engine_instance


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_pantheon_message(
    god_name: str,
    topic_basin: np.ndarray,
    coordizer: Optional[Any] = None
) -> str:
    """
    Convenience function to generate a Pantheon discussion message.
    
    Args:
        god_name: Name of the god (e.g., "Zeus", "Athena")
        topic_basin: Basin coordinates for the topic
        coordizer: Optional coordizer for word selection
        
    Returns:
        Generated message text
    """
    engine = get_template_engine(coordizer)
    return engine.generate_discussion_message(god_name, topic_basin, coordizer)


def generate_debate_exchange(
    god1_name: str,
    god2_name: str,
    topic_basin: np.ndarray,
    coordizer: Optional[Any] = None
) -> Tuple[str, str]:
    """
    Generate a debate exchange between two gods.
    
    Args:
        god1_name: First god's name
        god2_name: Second god's name
        topic_basin: Basin coordinates for the debate topic
        coordizer: Optional coordizer for word selection
        
    Returns:
        Tuple of (god1_message, god2_message)
    """
    engine = get_template_engine(coordizer)
    
    # Generate first message
    msg1 = engine.generate_discussion_message(god1_name, topic_basin, coordizer)
    
    # Slightly perturb basin for second message to get variety
    perturbed_basin = topic_basin + np.random.randn(len(topic_basin)) * 0.1
    perturbed_basin = np.abs(perturbed_basin) + 1e-10
    perturbed_basin = perturbed_basin / perturbed_basin.sum()
    
    msg2 = engine.generate_discussion_message(god2_name, perturbed_basin, coordizer)
    
    return (msg1, msg2)


print("[TemplatedResponses] Module loaded - grammatically correct Pantheon messages available")
