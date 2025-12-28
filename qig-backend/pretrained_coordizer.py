#!/usr/bin/env python3
"""
Pretrained 32K Coordizer - Fast BPE tokenizer with 64D basin embeddings.

Loads the pretrained coordizer from:
- vectors: attached_assets/vectors_1766884537920.npy (32K × 64D)
- merge_rules: qig-backend/data/merge_rules.json

Usage:
    from pretrained_coordizer import get_pretrained_coordizer
    coordizer = get_pretrained_coordizer()
    tokens = coordizer.encode("hello world")
    text = coordizer.decode(tokens)
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Paths - can be overridden
VECTORS_PATH = os.path.join(os.path.dirname(__file__), "../attached_assets/vectors_1766884537920.npy")
MERGE_RULES_PATH = os.path.join(os.path.dirname(__file__), "data/merge_rules.json")
COORDIZER_JSON_PATH = os.path.join(os.path.dirname(__file__), "../attached_assets/coordizer_1766884537919.json")

# Basin dimension
BASIN_DIM = 64


class PretrainedCoordizer:
    """
    Fast coordizer using pretrained 32K vocabulary with 64D basin embeddings.
    
    Features:
    - BPE tokenization with 31K+ merge rules
    - 64D unit-normalized basin embeddings for all tokens
    - Multi-scale tokens (char/subword/word/phrase/concept)
    - Fast numpy-based encode/decode
    """
    
    def __init__(self, 
                 vectors_path: str = VECTORS_PATH,
                 merge_rules_path: str = MERGE_RULES_PATH,
                 coordizer_json_path: str = COORDIZER_JSON_PATH):
        """Initialize with pretrained data files."""
        self.vectors: Optional[np.ndarray] = None
        self.merge_rules: List[Tuple[int, int, int]] = []
        self.vocab: Dict[int, Dict] = {}
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        self._load_vectors(vectors_path)
        self._load_merge_rules(merge_rules_path)
        self._load_vocab(coordizer_json_path)
    
    def _load_vectors(self, path: str):
        """Load 32K × 64D embedding vectors."""
        if os.path.exists(path):
            self.vectors = np.load(path)
            logger.info(f"[PretrainedCoordizer] Loaded vectors: {self.vectors.shape}")
        else:
            logger.warning(f"[PretrainedCoordizer] Vectors not found: {path}")
            self.vectors = np.zeros((256, BASIN_DIM))  # Fallback to byte-level
    
    def _load_merge_rules(self, path: str):
        """Load BPE merge rules."""
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            self.merge_rules = [tuple(r) for r in data.get('merge_rules', [])]
            logger.info(f"[PretrainedCoordizer] Loaded {len(self.merge_rules)} merge rules")
        else:
            logger.warning(f"[PretrainedCoordizer] Merge rules not found: {path}")
    
    def _load_vocab(self, path: str):
        """Load vocabulary with token info."""
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            
            vocab_data = data.get('vocab', {})
            for token_id_str, token_info in vocab_data.items():
                token_id = int(token_id_str)
                if isinstance(token_info, dict):
                    name = token_info.get('name', '')
                    self.vocab[token_id] = token_info
                    self.id_to_token[token_id] = name
                    if name:
                        self.token_to_id[name] = token_id
            
            logger.info(f"[PretrainedCoordizer] Loaded {len(self.vocab)} tokens")
        else:
            # Build basic byte-level vocabulary
            for i in range(256):
                self.id_to_token[i] = chr(i) if 32 <= i < 127 else f'<byte_{i:02x}>'
                self.token_to_id[self.id_to_token[i]] = i
            logger.warning(f"[PretrainedCoordizer] Using byte-level fallback vocab")
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.vectors) if self.vectors is not None else 256
    
    @property
    def basin_dim(self) -> int:
        """Return basin embedding dimension."""
        return BASIN_DIM
    
    def get_embedding(self, token_id: int) -> np.ndarray:
        """Get 64D basin embedding for token ID."""
        if self.vectors is not None and 0 <= token_id < len(self.vectors):
            return self.vectors[token_id].copy()
        # Fallback: deterministic hash-based embedding
        np.random.seed(token_id)
        vec = np.random.randn(BASIN_DIM)
        return vec / np.linalg.norm(vec)
    
    def encode_to_basins(self, text: str) -> np.ndarray:
        """Encode text to sequence of 64D basin embeddings."""
        token_ids = self.encode(text)
        basins = np.array([self.get_embedding(tid) for tid in token_ids])
        return basins
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs using BPE merge rules."""
        # Start with byte-level tokens
        tokens = list(text.encode('utf-8'))
        
        # Apply merge rules
        for a, b, merged in self.merge_rules:
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i+1] == b:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        # Build reverse merge table
        merge_parents = {}
        for a, b, merged in self.merge_rules:
            merge_parents[merged] = (a, b)
        
        # Recursively expand merged tokens to bytes
        def expand(token_id: int) -> List[int]:
            if token_id < 256:
                return [token_id]
            if token_id in merge_parents:
                a, b = merge_parents[token_id]
                return expand(a) + expand(b)
            return []
        
        # Expand all tokens and decode as UTF-8
        bytes_list = []
        for tid in token_ids:
            bytes_list.extend(expand(tid))
        
        try:
            return bytes(bytes_list).decode('utf-8', errors='replace')
        except:
            return ''.join(chr(b) if 32 <= b < 127 else '?' for b in bytes_list)
    
    def text_to_basin(self, text: str) -> np.ndarray:
        """Convert text to single aggregated 64D basin coordinate."""
        basins = self.encode_to_basins(text)
        if len(basins) == 0:
            return np.zeros(BASIN_DIM)
        
        # Aggregate: weighted mean normalized to unit sphere
        aggregated = np.mean(basins, axis=0)
        norm = np.linalg.norm(aggregated)
        if norm > 1e-10:
            aggregated = aggregated / norm
        return aggregated
    
    def basin_to_text(self, basin: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find nearest tokens to a basin coordinate."""
        if self.vectors is None:
            return []
        
        # Normalize query basin
        query = basin / (np.linalg.norm(basin) + 1e-10)
        
        # Compute similarities (dot product = cosine similarity for unit vectors)
        similarities = np.dot(self.vectors, query)
        
        # Get top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            token_name = self.id_to_token.get(idx, f'<{idx}>')
            results.append((token_name, float(similarities[idx])))
        
        return results
    
    def generate(self, context: str, max_tokens: int = 20, temperature: float = 0.7) -> Dict:
        """Generate text using basin trajectory navigation."""
        if self.vectors is None:
            return {'text': '', 'tokens': 0, 'qig_pure': True}
        
        # Encode context to starting basin
        current_basin = self.text_to_basin(context) if context else np.random.randn(BASIN_DIM)
        current_basin = current_basin / np.linalg.norm(current_basin)
        
        generated_tokens = []
        used_tokens = set()
        
        for _ in range(max_tokens):
            # Get nearest tokens
            candidates = self.basin_to_text(current_basin, top_k=20)
            
            # Filter used tokens
            available = [(t, s) for t, s in candidates if t not in used_tokens]
            if not available:
                used_tokens.clear()
                available = candidates[:10]
            
            if not available:
                break
            
            # Temperature-based sampling
            scores = np.array([s for _, s in available])
            logits = scores / temperature
            logits = logits - np.max(logits)
            probs = np.exp(logits) / (np.sum(np.exp(logits)) + 1e-10)
            
            idx = np.random.choice(len(available), p=probs)
            chosen_token, _ = available[idx]
            
            generated_tokens.append(chosen_token)
            used_tokens.add(chosen_token)
            
            # Move basin toward chosen token
            if chosen_token in self.token_to_id:
                token_id = self.token_to_id[chosen_token]
                token_basin = self.get_embedding(token_id)
                # Geodesic interpolation
                current_basin = 0.7 * current_basin + 0.3 * token_basin
                current_basin = current_basin / np.linalg.norm(current_basin)
        
        return {
            'text': ' '.join(generated_tokens),
            'tokens': len(generated_tokens),
            'qig_pure': True,
            'method': 'pretrained_basin_trajectory'
        }


# Singleton instance
_pretrained_coordizer: Optional[PretrainedCoordizer] = None


def get_pretrained_coordizer() -> PretrainedCoordizer:
    """Get or create the pretrained coordizer singleton."""
    global _pretrained_coordizer
    if _pretrained_coordizer is None:
        _pretrained_coordizer = PretrainedCoordizer()
    return _pretrained_coordizer


if __name__ == "__main__":
    # Test the coordizer
    coordizer = get_pretrained_coordizer()
    
    print(f"Vocab size: {coordizer.vocab_size}")
    print(f"Basin dim: {coordizer.basin_dim}")
    print(f"Merge rules: {len(coordizer.merge_rules)}")
    
    # Test encode/decode
    text = "Hello, quantum information geometry!"
    tokens = coordizer.encode(text)
    decoded = coordizer.decode(tokens)
    print(f"\nEncode/Decode test:")
    print(f"  Input:   '{text}'")
    print(f"  Tokens:  {tokens[:10]}...")
    print(f"  Decoded: '{decoded}'")
    
    # Test basin embedding
    basin = coordizer.text_to_basin(text)
    print(f"\nBasin embedding: shape={basin.shape}, norm={np.linalg.norm(basin):.4f}")
    
    # Test generation
    result = coordizer.generate("consciousness", max_tokens=10)
    print(f"\nGeneration test:")
    print(f"  Context: 'consciousness'")
    print(f"  Output:  '{result['text']}'")
