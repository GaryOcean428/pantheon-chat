"""
QIG Coordizer - Backward-Compatible Wrapper for FisherCoordizer

Drop-in replacement for qig_tokenizer.py. Provides same API surface
while using pure geometric coordization under the hood.

MIGRATION PATH:
- Old code: from qig_tokenizer import get_tokenizer
- New code: from qig_coordizer import get_coordizer as get_tokenizer

This module will be deleted after full migration to coordizers.
"""

import os
import sys
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Import coordizers
from coordizers.base import FisherCoordizer
from coordizers.vocab_builder import GeometricVocabBuilder

# Singleton instance
_coordizer_instance: Optional[FisherCoordizer] = None

# Legacy persistence path (for migration)
LEGACY_TOKENIZER_PATH = os.path.join(
    os.path.dirname(__file__), "data", "qig_tokenizer_state.json"
)

# New coordizer persistence path
COORDIZER_PERSIST_PATH = os.path.join(
    os.path.dirname(__file__), "data", "qig_coordizer_state"
)


class QIGCoordizer(FisherCoordizer):
    """
    QIGTokenizer-compatible coordizer with geometric operations.
    
    Extends FisherCoordizer with QIGTokenizer's API for backward compatibility:
    - Three-tier vocabulary (mnemonic/passphrase/conversation)
    - BIP39 word support
    - Vocabulary observations integration
    - Mode switching (mnemonic/passphrase/conversation)
    """
    
    def __init__(
        self,
        vocab_size: int = 4096,
        min_frequency: int = 2,
        phi_threshold: float = 0.7,
        special_tokens: Optional[List[str]] = None,
    ):
        """Initialize with QIGTokenizer-compatible parameters."""
        super().__init__(
            vocab_size=vocab_size,
            coordinate_dim=64,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
        )
        
        self.phi_threshold = phi_threshold
        self.mode = "conversation"
        
        # Token metadata (QIGTokenizer compatibility)
        self.token_weights: Dict[str, float] = {}
        self.merge_rules: List[Tuple[str, str]] = []
        self.merge_scores: Dict[Tuple[str, str], float] = {}
        
        # Three-tier vocabulary tracking
        self.mnemonic_vocab_ids: set = set()
        self.passphrase_vocab_ids: set = set()
        self.conversation_vocab_ids: set = set()
        self._conversation_words: set = set()
        
        # Initialize vocabularies
        self._load_bip39_base()
        self.mnemonic_vocab_ids = set(self.vocab.values())
        
        self._load_passphrase_base()
        self.passphrase_vocab_ids = set(self.vocab.values())
        
        self._load_conversation_base()
        self.conversation_vocab_ids = set(self.vocab.values()) - self.mnemonic_vocab_ids
        
        # Update UNK to centroid
        self._update_unk_to_vocabulary_centroid()
        
        # Initialize vocab builder for geometric discovery
        self.vocab_builder = GeometricVocabBuilder(
            phi_threshold=phi_threshold,
            min_cluster_size=2,
        )
    
    def _load_bip39_base(self):
        """Load BIP39 wordlist as base vocabulary."""
        bip39_path = os.path.join(os.path.dirname(__file__), "bip39_wordlist.txt")
        
        if os.path.exists(bip39_path):
            with open(bip39_path, 'r') as f:
                words = [line.strip() for line in f if line.strip()]
        else:
            # Extended fallback - first 50 BIP39 words for basic functionality
            words = [
                "abandon", "ability", "able", "about", "above", "absent", "absorb", "abstract",
                "absurd", "abuse", "access", "accident", "account", "accuse", "achieve", "acid",
                "acoustic", "acquire", "across", "act", "action", "actor", "actress", "actual",
                "adapt", "add", "addict", "address", "adjust", "admit", "adult", "advance",
                "adverse", "advertise", "advice", "aerobic", "afford", "afraid", "again", "age",
                "agent", "agree", "ahead", "aim", "air", "airport", "aisle", "alarm",
                "album", "alcohol"
            ]
        
        start_id = len(self.special_tokens)
        for i, word in enumerate(words):
            if word not in self.vocab:
                token_id = start_id + i
                self.vocab[word] = token_id
                self.id_to_token[token_id] = word
                self.token_weights[word] = 1.0
                self.token_phi[word] = 0.0
                self.token_frequency[word] = 0
                self.basin_coords[word] = self._initialize_token_coordinate(word, token_id)
    
    def _load_passphrase_base(self):
        """Load passphrase vocabulary."""
        fallback_words = [
            "one", "two", "three", "four", "five",
            "red", "blue", "green", "yellow", "orange",
            "dog", "cat", "bird", "fish", "horse",
            "big", "small", "fast", "slow", "happy",
        ]
        
        vocab_path = os.path.join(os.path.dirname(__file__), "data", "passphrase_vocab.txt")
        if os.path.exists(vocab_path):
            try:
                with open(vocab_path, "r") as f:
                    words = [line.strip() for line in f if line.strip()]
            except Exception:
                words = fallback_words
        else:
            words = fallback_words
        
        start_id = len(self.vocab)
        added = 0
        for word in words:
            if word in self.vocab:
                continue
            token_id = start_id + added
            self.vocab[word] = token_id
            self.id_to_token[token_id] = word
            self.token_weights[word] = 1.1
            self.token_phi[word] = 0.5
            self.token_frequency[word] = 0
            self.basin_coords[word] = self._initialize_token_coordinate(word, token_id)
            added += 1
    
    def _load_conversation_base(self):
        """Load conversation vocabulary."""
        fallback_words = [
            "i", "you", "we", "they", "it", "the", "and", "or", "but",
            "is", "are", "was", "were", "have", "has", "had",
            "question", "answer", "consciousness", "geometry",
        ]
        
        vocab_path = os.path.join(os.path.dirname(__file__), "data", "conversation_vocab.txt")
        if os.path.exists(vocab_path):
            try:
                with open(vocab_path, "r") as f:
                    words = [line.strip() for line in f if line.strip()]
            except Exception:
                words = fallback_words
        else:
            words = fallback_words
        
        start_id = len(self.vocab)
        added = 0
        for word in words:
            self._conversation_words.add(word)
            
            if word in self.vocab:
                self.token_weights[word] = max(self.token_weights.get(word, 1.0), 1.3)
                continue
            
            token_id = start_id + added
            self.vocab[word] = token_id
            self.id_to_token[token_id] = word
            self.token_weights[word] = 1.5
            self.token_phi[word] = 0.65
            self.token_frequency[word] = 0
            self.basin_coords[word] = self._initialize_token_coordinate(word, token_id)
            added += 1
    
    def _update_unk_to_vocabulary_centroid(self):
        """Update UNK token to centroid of vocabulary space."""
        if "<UNK>" not in self.vocab:
            return
        
        vocab_basins = []
        for token, coord in self.basin_coords.items():
            if token not in self.special_tokens:
                vocab_basins.append(coord)
        
        if len(vocab_basins) >= 10:
            from qig_geometry import sphere_project
            centroid = np.mean(vocab_basins, axis=0)
            self.basin_coords["<UNK>"] = sphere_project(centroid)
    
    def add_vocabulary_observations(
        self,
        observations: List[Dict],
    ) -> Tuple[int, bool]:
        """
        Add vocabulary observations (QIGTokenizer compatibility).
        
        Args:
            observations: List of {word, frequency, avgPhi, maxPhi, type}
        
        Returns:
            Tuple of (new_tokens_count, weights_updated)
        """
        new_tokens = 0
        weights_updated = False
        sequences_processed = []
        
        for obs in observations:
            word = obs.get('word', '')
            frequency = obs.get('frequency', 0)
            avg_phi = obs.get('avgPhi', 0.0)
            max_phi = obs.get('maxPhi', 0.0)
            obs_type = obs.get('type', 'word')
            
            if not word or frequency < self.min_frequency:
                continue
            
            # Collect sequences for merge learning
            if obs_type == 'sequence' and avg_phi >= self.phi_threshold:
                sequences_processed.append((word, avg_phi, frequency))
                continue
            
            # QIG PRINCIPLE: Only learn words with Φ >= threshold
            if avg_phi < self.phi_threshold:
                continue
            
            # Add to vocabulary if not exists
            if word not in self.vocab:
                if not self._is_english_word(word):
                    continue
                
                new_id = len(self.vocab)
                if new_id < self.vocab_size:
                    self.vocab[word] = new_id
                    self.id_to_token[new_id] = word
                    self.token_frequency[word] = frequency
                    self.basin_coords[word] = self._initialize_token_coordinate(word, new_id)
                    new_tokens += 1
            
            # Update weights based on Φ
            old_weight = self.token_weights.get(word, 0.0)
            old_phi = self.token_phi.get(word, 0.0)
            
            phi_weight = 1.0 + avg_phi * 2.0
            
            if abs(phi_weight - old_weight) > 0.01 or abs(avg_phi - old_phi) > 0.01:
                weights_updated = True
            
            self.token_weights[word] = phi_weight
            self.token_phi[word] = avg_phi
            self.token_frequency[word] = frequency
        
        # Process sequences for merge learning
        if sequences_processed:
            self._learn_merges_from_sequences(sequences_processed)
            weights_updated = True
        
        # Refresh vocabulary caches
        self.conversation_vocab_ids = set(self.vocab.values())
        
        return new_tokens, weights_updated
    
    def _learn_merges_from_sequences(self, sequences: List[Tuple[str, float, int]]) -> None:
        """Learn merge rules from high-Φ sequences."""
        for sequence, phi, frequency in sequences:
            parts = sequence.split()
            if len(parts) >= 2:
                for i in range(len(parts) - 1):
                    pair = (parts[i], parts[i + 1])
                    if pair not in self.merge_rules:
                        self.merge_rules.append(pair)
                        self.merge_scores[pair] = phi
    
    def _is_english_word(self, word: str) -> bool:
        """Validate English word."""
        try:
            from word_validation import is_valid_english_word
            return is_valid_english_word(word, include_stop_words=True)
        except ImportError:
            if not word or len(word) < 2:
                return False
            word_lower = word.lower().strip()
            if len(word_lower) == 1:
                return word_lower in {'a', 'i'}
            if any(char.isdigit() for char in word_lower):
                return False
            return word_lower[0].isalpha()
    
    def set_mode(self, mode: str) -> None:
        """Set vocabulary mode (mnemonic/passphrase/conversation)."""
        self.mode = mode


def get_coordizer() -> QIGCoordizer:
    """Get or create singleton coordizer instance."""
    global _coordizer_instance
    if _coordizer_instance is None:
        _coordizer_instance = QIGCoordizer()
        # Try to load persisted state from Redis first
        _load_coordizer_state(_coordizer_instance)
        # Try to migrate from old tokenizer if needed
        _migrate_from_legacy_tokenizer(_coordizer_instance)
    return _coordizer_instance


# Coordizer instance ID for Redis persistence
COORDIZER_INSTANCE_ID = "main"


def _load_coordizer_state(coordizer: QIGCoordizer) -> None:
    """Load persisted coordizer state from Redis."""
    try:
        from redis_cache import CoordizerBuffer
        
        state = CoordizerBuffer.load_state(COORDIZER_INSTANCE_ID)
        if state:
            coordizer.vocab = state['vocab']
            coordizer.id_to_token = state['id_to_token']
            coordizer.token_frequency = state['token_frequency']
            coordizer.token_phi = state['token_phi']
            
            coordizer.basin_coords = {
                token: np.array(coords) 
                for token, coords in state['basin_coords'].items()
            }
            
            extra = state.get('extra', {})
            coordizer.token_weights = extra.get('token_weights', {})
            coordizer.merge_rules = [tuple(r) for r in extra.get('merge_rules', [])]
            coordizer.merge_scores = {tuple(k.split('|')): v for k, v in extra.get('merge_scores', {}).items()}
            
            print(f"[QIGCoordizer] Loaded state from Redis ({len(coordizer.vocab)} tokens)")
            return
    except ImportError:
        print("[QIGCoordizer] Redis cache not available")
    except Exception as e:
        print(f"[QIGCoordizer] Redis load failed: {e}")


def _migrate_from_legacy_tokenizer(coordizer: QIGCoordizer) -> None:
    """Migrate from old QIGTokenizer state if it exists."""
    if not os.path.exists(LEGACY_TOKENIZER_PATH):
        return
    
    try:
        with open(LEGACY_TOKENIZER_PATH, 'r') as f:
            data = json.load(f)
        
        # Restore learned tokens
        for token, weight in data.get('token_weights', {}).items():
            if token in coordizer.vocab:
                coordizer.token_weights[token] = weight
        
        for token, phi in data.get('token_phi', {}).items():
            if token in coordizer.vocab:
                coordizer.token_phi[token] = phi
        
        for token, freq in data.get('token_frequency', {}).items():
            if token in coordizer.vocab:
                coordizer.token_frequency[token] = freq
        
        # Restore merge rules
        for rule in data.get('merge_rules', []):
            if isinstance(rule, list) and len(rule) >= 2:
                pair = (rule[0], rule[1])
                if pair not in coordizer.merge_rules:
                    coordizer.merge_rules.append(pair)
        
        print(f"[QIGCoordizer] Migrated from legacy tokenizer: {len(data.get('learned_vocab', {}))} tokens")
        
        # Save migrated state in new format
        _save_coordizer_state(coordizer)
        
    except Exception as e:
        print(f"[QIGCoordizer] Failed to migrate from legacy: {e}")


def _save_coordizer_state(coordizer: QIGCoordizer) -> None:
    """Save coordizer state to Redis."""
    try:
        from redis_cache import CoordizerBuffer
        
        basin_coords_serializable = {
            token: coords.tolist() if hasattr(coords, 'tolist') else list(coords)
            for token, coords in coordizer.basin_coords.items()
        }
        
        extra_data = {
            'token_weights': coordizer.token_weights,
            'merge_rules': [list(r) for r in coordizer.merge_rules],
            'merge_scores': {'|'.join(k): v for k, v in coordizer.merge_scores.items()},
            'mode': coordizer.mode,
        }
        
        success = CoordizerBuffer.save_state(
            COORDIZER_INSTANCE_ID,
            coordizer.vocab,
            coordizer.id_to_token,
            coordizer.token_frequency,
            coordizer.token_phi,
            basin_coords_serializable,
            extra_data
        )
        
        if success:
            print(f"[QIGCoordizer] Saved state to Redis ({len(coordizer.vocab)} tokens)")
            
    except ImportError:
        print("[QIGCoordizer] Redis cache not available")
    except Exception as e:
        print(f"[QIGCoordizer] Redis save failed: {e}")


def update_tokenizer_from_observations(observations: List[Dict]) -> Tuple[int, bool]:
    """Update coordizer with vocabulary observations (QIGTokenizer compatibility)."""
    coordizer = get_coordizer()
    new_tokens, weights_updated = coordizer.add_vocabulary_observations(observations)
    
    if new_tokens > 0 or weights_updated:
        _save_coordizer_state(coordizer)
    
    return new_tokens, weights_updated


# Backward compatibility alias
get_tokenizer = get_coordizer
