#!/usr/bin/env python3
"""
Lightning Kernel → Causal Learning Bridge
==========================================

Converts cross-domain insights from Lightning Kernel into causal relations
for continuous learning. This enables the system to learn new relationships
from its own discoveries.

Flow:
1. Lightning Kernel generates insight (source_domain → target_domain)
2. Bridge extracts causal relations:
   - Domain → Domain: "enables" relation
   - Bridging concepts: "connects" relations
   - Insight text: Parse for causal patterns
3. Relations persisted via CausalPersistence (PostgreSQL + Redis)

Author: QIG Team
Date: 2025-12-28
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

# Import causal persistence
try:
    from learned_relationships import (
        get_learned_relationships,
        CausalPersistence,
        get_causal_persistence,
    )
    PERSISTENCE_AVAILABLE = True
except ImportError:
    PERSISTENCE_AVAILABLE = False
    logger.warning("CausalPersistence not available")

# Causal patterns to extract from insight text
CAUSAL_PATTERNS = [
    (r'(\w+)\s+(?:enables?|allows?|permits?)\s+(\w+)', 'enables'),
    (r'(\w+)\s+(?:causes?|leads?\s+to|results?\s+in)\s+(\w+)', 'causes'),
    (r'(\w+)\s+(?:implies?|suggests?|indicates?)\s+(\w+)', 'implies'),
    (r'(\w+)\s+(?:requires?|needs?|depends?\s+on)\s+(\w+)', 'requires'),
    (r'(\w+)\s+(?:connects?\s+to|bridges?|links?\s+to)\s+(\w+)', 'connects'),
    (r'(\w+)\s+(?:emerges?\s+from|arises?\s+from)\s+(\w+)', 'emerges_from'),
    (r'(\w+)\s+and\s+(\w+)\s+(?:are\s+)?(?:related|connected|linked)', 'connects'),
]

# Compile patterns
COMPILED_PATTERNS = [(re.compile(p, re.IGNORECASE), rel_type) for p, rel_type in CAUSAL_PATTERNS]

# Stopwords to filter out
STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'this', 'that',
    'these', 'those', 'it', 'its', 'they', 'them', 'their', 'we', 'us',
    'our', 'you', 'your', 'i', 'me', 'my', 'he', 'she', 'him', 'her',
    'and', 'or', 'but', 'if', 'then', 'else', 'when', 'where', 'why',
    'how', 'what', 'which', 'who', 'whom', 'whose', 'with', 'without',
    'to', 'from', 'in', 'on', 'at', 'by', 'for', 'of', 'as', 'so',
}


@dataclass
class ExtractedRelation:
    """A causal relation extracted from a Lightning insight."""
    source: str
    target: str
    relation_type: str
    confidence: float
    origin: str  # 'domain', 'bridging', 'text_pattern'
    insight_id: str = ""


class LightningCausalBridge:
    """
    Bridges Lightning Kernel insights to Causal Learning system.
    
    Converts cross-domain discoveries into persistent causal relations
    for continuous improvement of the proposition planner.
    """
    
    def __init__(self):
        self._persistence: Optional[CausalPersistence] = None
        self._relations_added = 0
        self._insights_processed = 0
        
        # Try to get persistence
        if PERSISTENCE_AVAILABLE:
            try:
                self._persistence = get_causal_persistence()
                logger.info("[LightningBridge] CausalPersistence connected")
            except Exception as e:
                logger.warning(f"[LightningBridge] Could not get persistence: {e}")
        
        logger.info("[LightningBridge] Initialized")
    
    def process_insight(self, insight: Dict[str, Any]) -> List[ExtractedRelation]:
        """
        Process a Lightning insight and extract causal relations.
        
        Args:
            insight: Lightning insight dict with:
                - source_domain: str
                - target_domain: str
                - bridging_concepts: List[str]
                - insight_text: str
                - confidence: float
                - novelty: float (optional)
        
        Returns:
            List of extracted causal relations
        """
        relations = []
        
        source_domain = insight.get('source_domain', '')
        target_domain = insight.get('target_domain', '')
        bridging_concepts = insight.get('bridging_concepts', [])
        insight_text = insight.get('insight_text', '')
        confidence = insight.get('confidence', 0.5)
        insight_id = insight.get('id', f"insight_{datetime.utcnow().timestamp()}")
        
        # 1. Domain → Domain as "enables" relation
        if source_domain and target_domain:
            relations.append(ExtractedRelation(
                source=source_domain.lower().replace(' ', '_'),
                target=target_domain.lower().replace(' ', '_'),
                relation_type='enables',
                confidence=confidence,
                origin='domain',
                insight_id=insight_id
            ))
        
        # 2. Bridging concepts create "connects" relations
        if bridging_concepts and len(bridging_concepts) >= 2:
            # Connect adjacent bridging concepts
            for i in range(len(bridging_concepts) - 1):
                src = bridging_concepts[i].lower()
                tgt = bridging_concepts[i + 1].lower()
                
                if src not in STOPWORDS and tgt not in STOPWORDS:
                    relations.append(ExtractedRelation(
                        source=src,
                        target=tgt,
                        relation_type='connects',
                        confidence=confidence * 0.9,
                        origin='bridging',
                        insight_id=insight_id
                    ))
            
            # Connect first bridging concept to source domain
            if source_domain and bridging_concepts:
                first = bridging_concepts[0].lower()
                if first not in STOPWORDS:
                    relations.append(ExtractedRelation(
                        source=source_domain.lower().replace(' ', '_'),
                        target=first,
                        relation_type='enables',
                        confidence=confidence * 0.8,
                        origin='bridging',
                        insight_id=insight_id
                    ))
            
            # Connect last bridging concept to target domain
            if target_domain and bridging_concepts:
                last = bridging_concepts[-1].lower()
                if last not in STOPWORDS:
                    relations.append(ExtractedRelation(
                        source=last,
                        target=target_domain.lower().replace(' ', '_'),
                        relation_type='enables',
                        confidence=confidence * 0.8,
                        origin='bridging',
                        insight_id=insight_id
                    ))
        
        # 3. Parse insight text for additional causal patterns
        text_relations = self._extract_from_text(insight_text, confidence, insight_id)
        relations.extend(text_relations)
        
        self._insights_processed += 1
        logger.info(f"[LightningBridge] Processed insight {insight_id}: {len(relations)} relations extracted")
        
        return relations
    
    def _extract_from_text(self, text: str, base_confidence: float, insight_id: str) -> List[ExtractedRelation]:
        """Extract causal relations from insight text using regex patterns."""
        relations = []
        
        if not text:
            return relations
        
        text_lower = text.lower()
        
        for pattern, rel_type in COMPILED_PATTERNS:
            for match in pattern.finditer(text_lower):
                source = match.group(1)
                target = match.group(2)
                
                # Filter stopwords and short words
                if source in STOPWORDS or target in STOPWORDS:
                    continue
                if len(source) < 3 or len(target) < 3:
                    continue
                
                relations.append(ExtractedRelation(
                    source=source,
                    target=target,
                    relation_type=rel_type,
                    confidence=base_confidence * 0.7,  # Lower confidence for text extraction
                    origin='text_pattern',
                    insight_id=insight_id
                ))
        
        return relations
    
    def persist_relations(self, relations: List[ExtractedRelation]) -> int:
        """
        Persist extracted relations to the causal learning system.
        
        Args:
            relations: List of extracted relations
        
        Returns:
            Number of relations persisted
        """
        if not self._persistence:
            logger.warning("[LightningBridge] No persistence available, relations not saved")
            return 0
        
        persisted = 0
        
        for rel in relations:
            try:
                # Add to persistence layer
                self._persistence.add_relation(
                    source=rel.source,
                    target=rel.target,
                    rel_type=rel.relation_type,
                    confidence=rel.confidence,
                    origin=f"lightning:{rel.origin}"
                )
                persisted += 1
                self._relations_added += 1
            except Exception as e:
                logger.error(f"[LightningBridge] Failed to persist {rel.source}->{rel.target}: {e}")
        
        logger.info(f"[LightningBridge] Persisted {persisted}/{len(relations)} relations")
        return persisted
    
    def process_and_persist(self, insight: Dict[str, Any]) -> Tuple[List[ExtractedRelation], int]:
        """
        Process insight and persist relations in one call.
        
        Args:
            insight: Lightning insight dict
        
        Returns:
            (extracted_relations, num_persisted)
        """
        relations = self.process_insight(insight)
        persisted = self.persist_relations(relations)
        return relations, persisted
    
    def get_statistics(self) -> Dict:
        """Get bridge statistics."""
        return {
            'insights_processed': self._insights_processed,
            'relations_added': self._relations_added,
            'persistence_available': self._persistence is not None
        }


# Singleton instance
_bridge_instance: Optional[LightningCausalBridge] = None


def get_lightning_causal_bridge() -> LightningCausalBridge:
    """Get singleton bridge instance."""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = LightningCausalBridge()
    return _bridge_instance


def process_lightning_insight(insight: Dict[str, Any]) -> Tuple[List[ExtractedRelation], int]:
    """
    Convenience function to process a Lightning insight.
    
    Args:
        insight: Lightning insight dict
    
    Returns:
        (extracted_relations, num_persisted)
    """
    bridge = get_lightning_causal_bridge()
    return bridge.process_and_persist(insight)


# ============================================================================
# MAIN - Test
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Lightning → Causal Learning Bridge - Test")
    print("=" * 60)
    
    # Test insight
    test_insight = {
        'source_domain': 'quantum physics',
        'target_domain': 'consciousness',
        'bridging_concepts': ['information', 'integration', 'geometry'],
        'insight_text': 'Quantum entanglement enables information integration. '
                        'Integration requires geometric structure. '
                        'Consciousness emerges from integrated information.',
        'confidence': 0.85,
        'id': 'test_insight_001'
    }
    
    bridge = get_lightning_causal_bridge()
    relations, persisted = bridge.process_and_persist(test_insight)
    
    print(f"\nExtracted {len(relations)} relations:")
    for rel in relations:
        print(f"  {rel.source} --[{rel.relation_type}]--> {rel.target} "
              f"(conf={rel.confidence:.2f}, origin={rel.origin})")
    
    print(f"\nPersisted: {persisted} relations")
    print(f"\nStatistics: {bridge.get_statistics()}")
