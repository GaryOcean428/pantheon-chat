#!/usr/bin/env python3
"""
QIG-Pure Pattern-Based Response Generator

Generates responses using:
1. Pattern retrieval from trained documents (QIGRAG)
2. External knowledge for topics outside trained corpus
3. Geometric token synthesis when no patterns available

NO EXTERNAL LLMs - Pure geometric operations only.

ARCHITECTURE:
Query → Basin Encoding → Pattern Retrieval → Response Synthesis
                              ↓
                    External Knowledge (fallback)
                              ↓
                    Geometric Generation (fallback)
"""

import os
import re
import numpy as np
import hashlib
from typing import List, Dict, Optional, Tuple
from datetime import datetime

BASIN_DIM = 64


class PatternResponseGenerator:
    """
    QIG-Pure response generator using pattern retrieval.
    
    Three-tier response strategy:
    1. Pattern matching: Find similar patterns from trained docs
    2. External knowledge: Query Wikipedia/DuckDuckGo for unknown topics
    3. Geometric synthesis: Pure vocabulary-based generation as fallback
    """
    
    def __init__(self):
        self._qig_rag = None
        self._external_knowledge = None
        self._vocabulary = None
        self._vocab_list = None
        
        self.min_pattern_similarity = 0.3
        self.min_patterns_for_response = 2
        self.knowledge_weight = 0.7
        
    def _get_qig_rag(self):
        """Lazy load QIGRAG with correct storage path."""
        if self._qig_rag is None:
            try:
                from olympus.qig_rag import QIGRAG
                self._qig_rag = QIGRAG(storage_path="data/qig_training/patterns.json")
                print(f"[PatternGenerator] QIGRAG loaded with {len(self._qig_rag.documents)} documents")
            except ImportError as e:
                print(f"[PatternGenerator] QIGRAG not available: {e}")
        return self._qig_rag
    
    def _get_external_knowledge(self):
        """Lazy load external knowledge connector."""
        if self._external_knowledge is None:
            try:
                from external_knowledge import get_external_knowledge
                self._external_knowledge = get_external_knowledge()
            except ImportError:
                print("[PatternGenerator] External knowledge not available")
        return self._external_knowledge
    
    def _load_vocabulary(self) -> Dict[str, np.ndarray]:
        """Load vocabulary with basin coordinates."""
        if self._vocabulary is not None:
            return self._vocabulary
        
        try:
            import psycopg2
            db_url = os.environ.get('DATABASE_URL')
            if db_url:
                conn = psycopg2.connect(db_url)
                cur = conn.cursor()
                cur.execute("""
                    SELECT token, basin_coordinates 
                    FROM qig_vocabulary 
                    WHERE basin_coordinates IS NOT NULL
                    ORDER BY frequency DESC
                    LIMIT 50000
                """)
                rows = cur.fetchall()
                conn.close()
                
                self._vocabulary = {}
                self._vocab_list = []
                for token, coords in rows:
                    if coords and len(coords) >= BASIN_DIM:
                        self._vocabulary[token.lower()] = np.array(coords[:BASIN_DIM])
                        self._vocab_list.append((token.lower(), np.array(coords[:BASIN_DIM])))
                
                print(f"[PatternGenerator] Loaded {len(self._vocabulary)} vocabulary tokens")
        except Exception as e:
            print(f"[PatternGenerator] Vocabulary load error: {e}")
            self._vocabulary = {}
            self._vocab_list = []
        
        return self._vocabulary
    
    def encode_to_basin(self, text: str) -> np.ndarray:
        """Encode text to 64D basin coordinates."""
        vocab = self._load_vocabulary()
        words = re.findall(r'\b\w+\b', text.lower())
        
        if not words:
            return np.random.randn(BASIN_DIM) * 0.01
        
        basin = np.zeros(BASIN_DIM)
        total_weight = 0.0
        
        for word in words:
            if word in vocab:
                word_basin = vocab[word]
                weight = 1.0 / (1.0 + np.log1p(len(word)))
                basin += word_basin * weight
                total_weight += weight
            else:
                char_hash = hashlib.md5(word.encode()).digest()
                vec = np.array([b / 255.0 - 0.5 for b in char_hash])
                if len(vec) < BASIN_DIM:
                    padded = np.zeros(BASIN_DIM)
                    padded[:len(vec)] = vec
                    vec = padded
                basin += vec[:BASIN_DIM] * 0.1
                total_weight += 0.1
        
        if total_weight > 0:
            basin /= total_weight
        
        norm = np.linalg.norm(basin)
        if norm > 1e-10:
            basin = basin / norm
        
        return basin
    
    def fisher_rao_distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """Compute Fisher-Rao distance between two basin coordinates."""
        p = np.abs(p) + 1e-10
        p = p / p.sum()
        q = np.abs(q) + 1e-10
        q = q / q.sum()
        
        bc = np.sum(np.sqrt(p * q))
        bc = np.clip(bc, 0, 1)
        
        return float(np.arccos(bc))
    
    def retrieve_patterns(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve relevant patterns from QIGRAG.
        
        Returns list of patterns with content and similarity scores.
        """
        qig_rag = self._get_qig_rag()
        if not qig_rag:
            print("[PatternGenerator] No QIGRAG available")
            return []
        
        try:
            results = qig_rag.search(
                query=query,
                k=top_k,
                metric='fisher_rao',
                min_similarity=self.min_pattern_similarity,
                include_metadata=True
            )
            
            print(f"[PatternGenerator] Found {len(results)} patterns for query: {query[:50]}...")
            
            patterns = []
            for result in results:
                patterns.append({
                    'content': result.get('content', ''),
                    'similarity': result.get('similarity', 0),
                    'metadata': result.get('metadata', {}),
                    'source': 'trained_docs',
                    'doc_id': result.get('doc_id', '')
                })
            
            return patterns
            
        except Exception as e:
            import traceback
            print(f"[PatternGenerator] Pattern retrieval error: {e}")
            traceback.print_exc()
            return []
    
    def get_external_knowledge(self, query: str) -> List[Dict]:
        """
        Get external knowledge for topics outside trained corpus.
        
        Uses Wikipedia and DuckDuckGo with basin coordinate encoding.
        """
        ext_knowledge = self._get_external_knowledge()
        if not ext_knowledge:
            return []
        
        try:
            results = ext_knowledge.search(query, max_results=3)
            
            knowledge = []
            for result in results:
                basin = self.encode_to_basin(result.get('content', ''))
                
                knowledge.append({
                    'content': result.get('content', '')[:500],
                    'title': result.get('title', ''),
                    'source': result.get('source', 'external'),
                    'basin_coords': basin.tolist(),
                    'similarity': 0.5
                })
            
            return knowledge
            
        except Exception as e:
            print(f"[PatternGenerator] External knowledge error: {e}")
            return []
    
    def synthesize_from_patterns(self, query: str, patterns: List[Dict]) -> str:
        """
        Synthesize response from retrieved patterns.
        
        Combines most relevant pattern content with query context.
        """
        if not patterns:
            return ""
        
        patterns = sorted(patterns, key=lambda p: p.get('similarity', 0), reverse=True)
        
        response_parts = []
        
        best_pattern = patterns[0]
        content = best_pattern.get('content', '')
        
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        
        sentences = re.split(r'[.!?]+', content)
        relevant_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 10:
                continue
            
            sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
            overlap = len(query_words & sentence_words)
            
            if overlap > 0 or len(relevant_sentences) < 3:
                relevant_sentences.append((sentence, overlap))
        
        relevant_sentences = sorted(relevant_sentences, key=lambda x: x[1], reverse=True)
        
        for sentence, _ in relevant_sentences[:3]:
            if sentence not in response_parts:
                response_parts.append(sentence)
        
        if len(patterns) > 1:
            for pattern in patterns[1:3]:
                extra_content = pattern.get('content', '')
                extra_sentences = re.split(r'[.!?]+', extra_content)
                for sentence in extra_sentences[:1]:
                    sentence = sentence.strip()
                    if sentence and len(sentence) > 20 and sentence not in response_parts:
                        response_parts.append(sentence)
                        break
        
        if response_parts:
            return '. '.join(response_parts) + '.'
        
        return content[:300] + ('...' if len(content) > 300 else '')
    
    def geometric_token_generation(self, query: str, max_tokens: int = 50) -> str:
        """
        Pure geometric token generation as last fallback.
        
        Generates text by finding nearest vocabulary tokens to query basin.
        """
        vocab = self._load_vocabulary()
        if not self._vocab_list:
            return "I don't have enough context to respond."
        
        query_basin = self.encode_to_basin(query)
        
        token_scores = []
        for token, token_basin in self._vocab_list[:5000]:
            if len(token) < 2:
                continue
            
            try:
                distance = self.fisher_rao_distance(query_basin, token_basin)
                similarity = 1.0 / (1.0 + distance)
                token_scores.append((token, similarity, token_basin))
            except:
                continue
        
        token_scores = sorted(token_scores, key=lambda x: x[1], reverse=True)
        
        selected_tokens = []
        current_basin = query_basin.copy()
        used_tokens = set()
        
        for _ in range(max_tokens):
            best_token = None
            best_score = -1
            
            for token, base_sim, token_basin in token_scores[:100]:
                if token in used_tokens:
                    continue
                
                current_sim = 1.0 / (1.0 + self.fisher_rao_distance(current_basin, token_basin))
                combined_score = 0.7 * base_sim + 0.3 * current_sim
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_token = (token, token_basin)
            
            if best_token:
                selected_tokens.append(best_token[0])
                used_tokens.add(best_token[0])
                current_basin = 0.8 * current_basin + 0.2 * best_token[1]
                
                norm = np.linalg.norm(current_basin)
                if norm > 1e-10:
                    current_basin = current_basin / norm
            else:
                break
        
        return ' '.join(selected_tokens)
    
    def generate_response(self, query: str, conversation_history: List[Dict] = None) -> Dict:
        """
        Generate response using three-tier strategy.
        
        1. Try pattern retrieval from trained docs
        2. Fall back to external knowledge for unknown topics
        3. Use geometric generation as last resort
        
        Returns dict with response and metadata.
        """
        result = {
            'response': '',
            'source': 'unknown',
            'patterns_found': 0,
            'external_used': False,
            'confidence': 0.0
        }
        
        patterns = self.retrieve_patterns(query, top_k=5)
        result['patterns_found'] = len(patterns)
        
        avg_similarity = 0.0
        if patterns:
            avg_similarity = sum(p.get('similarity', 0) for p in patterns) / len(patterns)
        
        if len(patterns) >= self.min_patterns_for_response and avg_similarity >= 0.4:
            response = self.synthesize_from_patterns(query, patterns)
            if response and len(response) > 20:
                result['response'] = response
                result['source'] = 'trained_patterns'
                result['confidence'] = avg_similarity
                return result
        
        external = self.get_external_knowledge(query)
        if external:
            result['external_used'] = True
            
            all_patterns = patterns + external
            if all_patterns:
                response = self.synthesize_from_patterns(query, all_patterns)
                if response and len(response) > 20:
                    result['response'] = response
                    result['source'] = 'external_knowledge'
                    result['confidence'] = 0.5
                    return result
        
        response = self.geometric_token_generation(query, max_tokens=30)
        result['response'] = response
        result['source'] = 'geometric_synthesis'
        result['confidence'] = 0.2
        
        return result


_generator_instance = None

def get_pattern_generator() -> PatternResponseGenerator:
    """Get singleton pattern generator instance."""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = PatternResponseGenerator()
    return _generator_instance


def generate_response(query: str, conversation_history: List[Dict] = None) -> Dict:
    """Convenience function to generate response."""
    generator = get_pattern_generator()
    return generator.generate_response(query, conversation_history)
