#!/usr/bin/env python3
"""
VOCABULARY FREQUENCY TRACKER (Python Backend)

Tracks words and phrases from high-Φ discoveries for vocabulary expansion.
Uses PostgreSQL exclusively for persistent storage.

IMPORTANT DISTINCTION:
- "word": An actual vocabulary word (BIP-39 or real English word)
- "phrase": A mutated/concatenated string (e.g., "transactionssent", "knownreceive")
- "sequence": A multi-word pattern (e.g., "abandon ability able")

The system identifies if text is a real word vs a mutation based on:
- BIP-39 wordlist membership
- English dictionary check
- Pattern analysis (concatenations, unusual length, etc.)

Ported from: server/vocabulary-tracker.ts
E8 Protocol: v4.0
"""

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Literal
from threading import Timer

from persistence.base_persistence import get_db_connection

logger = logging.getLogger(__name__)

# Database availability check
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    logger.warning("[VocabularyTracker] psycopg2 not available - database operations disabled")


# BIP-39 wordlist (subset for testing, full 2048 words in production)
BIP39_WORDS = {
    'abandon', 'ability', 'able', 'about', 'above', 'absent', 'absorb', 'abstract', 'absurd', 'abuse',
    'access', 'accident', 'account', 'accuse', 'achieve', 'acid', 'acoustic', 'acquire', 'across', 'act',
    'action', 'actor', 'actress', 'actual', 'adapt', 'add', 'addict', 'address', 'adjust', 'admit',
    'adult', 'advance', 'advice', 'aerobic', 'affair', 'afford', 'afraid', 'again', 'age', 'agent',
    'agree', 'ahead', 'aim', 'air', 'airport', 'aisle', 'alarm', 'album', 'alcohol', 'alert',
    'alien', 'all', 'alley', 'allow', 'almost', 'alone', 'alpha', 'already', 'also', 'alter',
    'always', 'amateur', 'amazing', 'among', 'amount', 'amused', 'analyst', 'anchor', 'ancient', 'anger',
    'angle', 'angry', 'animal', 'ankle', 'announce', 'annual', 'another', 'answer', 'antenna', 'antique',
    'anxiety', 'any', 'apart', 'apology', 'appear', 'apple', 'approve', 'april', 'arch', 'arctic',
    'area', 'arena', 'argue', 'arm', 'armed', 'armor', 'army', 'around', 'arrange', 'arrest',
    'arrive', 'arrow', 'art', 'artefact', 'artist', 'artwork', 'ask', 'aspect', 'assault', 'asset',
    'assist', 'assume', 'asthma', 'athlete', 'atom', 'attack', 'attend', 'attitude', 'attract', 'auction',
    'audit', 'august', 'aunt', 'author', 'auto', 'autumn', 'average', 'avocado', 'avoid', 'awake',
    'aware', 'away', 'awesome', 'awful', 'awkward', 'axis', 'baby', 'bachelor', 'bacon', 'badge',
    'bag', 'balance', 'balcony', 'ball', 'bamboo', 'banana', 'banner', 'bar', 'barely', 'bargain',
    'barrel', 'base', 'basic', 'basket', 'battle', 'beach', 'bean', 'beauty', 'because', 'become',
    'beef', 'before', 'begin', 'behave', 'behind', 'believe', 'below', 'belt', 'bench', 'benefit',
    'best', 'betray', 'better', 'between', 'beyond', 'bicycle', 'bid', 'bike', 'bind', 'biology',
    'bird', 'birth', 'bitter', 'black', 'blade', 'blame', 'blanket', 'blast', 'bleak', 'bless',
    'blind', 'blood', 'blossom', 'blouse', 'blue', 'blur', 'blush', 'board', 'boat', 'body',
    # Additional common BIP-39 words for comprehensive coverage
    'bonus', 'book', 'border', 'boring', 'borrow', 'boss', 'bottom', 'bounce', 'box', 'boy',
    'bracket', 'brain', 'brand', 'brass', 'brave', 'bread', 'breeze', 'brick', 'bridge', 'brief',
    'bright', 'bring', 'brisk', 'broccoli', 'broken', 'bronze', 'broom', 'brother', 'brown', 'brush',
    'bubble', 'buddy', 'budget', 'buffalo', 'build', 'bulb', 'bulk', 'bullet', 'bundle', 'bunker',
    'burden', 'burger', 'burst', 'bus', 'business', 'busy', 'butter', 'buyer', 'buzz', 'cabbage',
}

# Common English words not in BIP-39 but real words
COMMON_ENGLISH = {
    'the', 'is', 'are', 'was', 'were', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can',
    'transaction', 'transactions', 'sent', 'receive', 'received', 'known', 'changes',
    'executed', 'information', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
}


class PhraseCategory(str, Enum):
    """
    Phrase categories for kernel learning:
    - bip39_seed: Valid 12/15/18/21/24 word phrases with ALL words from BIP-39 wordlist
    - passphrase: Arbitrary text (any length, may have special chars/numbers, not a seed)
    - mutation: Seed-length (12/15/18/21/24 words) but contains non-BIP-39 words
    - bip39_word: Single word from BIP-39 wordlist
    - unknown: Not yet classified
    """
    BIP39_SEED = 'bip39_seed'
    PASSPHRASE = 'passphrase'
    MUTATION = 'mutation'
    BIP39_WORD = 'bip39_word'
    UNKNOWN = 'unknown'


ObservationType = Literal['word', 'phrase', 'sequence']


@dataclass
class PhraseObservation:
    """Individual phrase or word observation."""
    text: str
    frequency: int
    avg_phi: float
    max_phi: float
    first_seen: datetime
    last_seen: datetime
    contexts: List[str]
    is_real_word: bool
    is_bip39_word: bool
    type: ObservationType
    phrase_category: PhraseCategory


@dataclass
class SequenceObservation:
    """Multi-token sequence observation."""
    sequence: str
    words: List[str]
    frequency: int
    avg_phi: float
    max_phi: float
    efficiency_gain: float


@dataclass
class VocabularyCandidate:
    """Vocabulary expansion candidate."""
    text: str
    type: ObservationType
    frequency: int
    avg_phi: float
    max_phi: float
    efficiency_gain: float
    reasoning: str
    is_real_word: bool
    components: Optional[List[str]] = None


@dataclass
class TrackerStats:
    """Statistics about tracked vocabulary."""
    total_words: int
    total_phrases: int
    total_sequences: int
    top_words: List[Dict]
    top_phrases: List[Dict]
    top_sequences: List[Dict]
    candidates_ready: int


@dataclass
class CategoryStats:
    """Category statistics for kernel learning insights."""
    categories: Dict[str, Dict]
    bip39_coverage: float
    total_observations: int


class VocabularyTracker:
    """
    Tracks vocabulary frequency and high-Φ discoveries for kernel learning.
    
    Features:
    - Real-time observation tracking with debounced saves
    - BIP-39 seed vs passphrase classification
    - Multi-token sequence detection
    - PostgreSQL persistence with batch operations
    - Geometric memory bootstrap
    
    Args:
        min_frequency: Minimum frequency threshold for candidates
        min_phi: Minimum Φ threshold for observations
        max_sequence_length: Maximum tokens in a sequence
    """
    
    def __init__(
        self,
        min_frequency: int = 3,
        min_phi: float = 0.35,
        max_sequence_length: int = 5
    ):
        self.phrase_observations: Dict[str, PhraseObservation] = {}
        self.sequence_observations: Dict[str, SequenceObservation] = {}
        self.save_queue: Dict[str, PhraseObservation] = {}
        self.save_timer: Optional[Timer] = None
        
        self.min_frequency = min_frequency
        self.min_phi = min_phi
        self.max_sequence_length = max_sequence_length
        
        # Load data from PostgreSQL
        self._load_from_postgres()
    
    def _is_real_word(self, text: str) -> bool:
        """
        Check if a text string is a real vocabulary word.
        
        Heuristics for mutations (NOT real words):
        - Contains no vowels (unlikely real word)
        - Very long single "word" (>15 chars without spaces likely concatenation)
        - All consonants pattern
        - Mixed case within word
        - Contains digits
        """
        lower = text.lower()
        
        # Check BIP-39 wordlist
        if lower in BIP39_WORDS:
            return True
        
        # Check common English words
        if lower in COMMON_ENGLISH:
            return True
        
        # Heuristics for mutations
        if len(lower) > 15 and ' ' not in lower:
            return False
        if re.search(r'\d', lower):
            return False
        if not re.search(r'[aeiou]', lower):
            return False
        
        # Words 3-12 chars with vowels are likely real
        if 3 <= len(lower) <= 12:
            vowel_matches = re.findall(r'[aeiou]', lower)
            vowel_ratio = len(vowel_matches) / len(lower)
            if 0.2 <= vowel_ratio <= 0.6:
                return True
        
        return False
    
    def _classify_type(self, text: str) -> ObservationType:
        """Determine the type of observation."""
        if ' ' in text:
            return 'sequence'
        if self._is_real_word(text):
            return 'word'
        return 'phrase'
    
    def _is_bip39_word(self, word: str) -> bool:
        """Check if a single word is in the BIP-39 wordlist."""
        return word.lower() in BIP39_WORDS
    
    def classify_phrase_category(self, text: str) -> PhraseCategory:
        """
        Classify a phrase into categories for kernel learning.
        
        This teaches kernels the difference between:
        - Valid BIP-39 seed phrases (recoverable wallets)
        - Passphrases (arbitrary text, not seeds)
        - Mutations (seed-length but invalid)
        """
        trimmed = text.strip()
        words = re.split(r'\s+', trimmed)
        word_count = len(words)
        VALID_SEED_LENGTHS = [12, 15, 18, 21, 24]
        
        # Single word classification
        if word_count == 1:
            word = words[0].lower()
            if word in BIP39_WORDS:
                return PhraseCategory.BIP39_WORD
            # Contains special chars or numbers = passphrase
            if re.search(r'[^a-z]', word):
                return PhraseCategory.PASSPHRASE
            return PhraseCategory.UNKNOWN
        
        # Multi-word: check if it's seed-length
        if word_count in VALID_SEED_LENGTHS:
            # Check if ALL words are valid BIP-39 words
            all_bip39 = all(w.lower() in BIP39_WORDS for w in words)
            
            if all_bip39:
                return PhraseCategory.BIP39_SEED  # Valid BIP-39 seed phrase!
            else:
                return PhraseCategory.MUTATION  # Seed-length but contains invalid words
        
        # Not a seed-length phrase = passphrase
        return PhraseCategory.PASSPHRASE
    
    def _tokenize(self, phrase: str) -> List[str]:
        """Tokenize phrase into words/phrases."""
        return [
            w for w in re.sub(r'[^a-z0-9\s]', ' ', phrase.lower()).split()
            if w
        ]
    
    def observe(
        self,
        phrase: str,
        phi: float,
        kappa: Optional[float] = None,
        regime: Optional[str] = None,
        basin_coordinates: Optional[List[float]] = None
    ) -> None:
        """
        Observe a phrase from search results.
        
        Args:
            phrase: Text phrase to observe
            phi: Integration score (Φ)
            kappa: Coupling constant (κ)
            regime: Consciousness regime
            basin_coordinates: 64D basin coordinates
        """
        if phi < self.min_phi:
            return
        
        tokens = self._tokenize(phrase)
        now = datetime.utcnow()
        
        # Track individual tokens (words or phrases)
        for token in tokens:
            if len(token) < 2:
                continue
            
            obs_type = self._classify_type(token)
            is_real = obs_type == 'word'
            is_bip39 = self._is_bip39_word(token)
            category = self.classify_phrase_category(token)
            
            if token in self.phrase_observations:
                existing = self.phrase_observations[token]
                existing.frequency += 1
                existing.avg_phi = (
                    existing.avg_phi * (existing.frequency - 1) + phi
                ) / existing.frequency
                existing.max_phi = max(existing.max_phi, phi)
                existing.last_seen = now
                if phrase not in existing.contexts and len(existing.contexts) < 10:
                    existing.contexts.append(phrase)
            else:
                self.phrase_observations[token] = PhraseObservation(
                    text=token,
                    frequency=1,
                    avg_phi=phi,
                    max_phi=phi,
                    first_seen=now,
                    last_seen=now,
                    contexts=[phrase],
                    is_real_word=is_real,
                    is_bip39_word=is_bip39,
                    type=obs_type,
                    phrase_category=category,
                )
            
            # Queue for batch save
            self._queue_for_save(token)
        
        # Track multi-token sequences
        for length in range(2, min(self.max_sequence_length, len(tokens)) + 1):
            for i in range(len(tokens) - length + 1):
                seq_tokens = tokens[i:i + length]
                sequence = ' '.join(seq_tokens)
                
                if sequence in self.sequence_observations:
                    existing = self.sequence_observations[sequence]
                    existing.frequency += 1
                    existing.avg_phi = (
                        existing.avg_phi * (existing.frequency - 1) + phi
                    ) / existing.frequency
                    existing.max_phi = max(existing.max_phi, phi)
                    existing.efficiency_gain = existing.frequency * (len(seq_tokens) - 1)
                else:
                    self.sequence_observations[sequence] = SequenceObservation(
                        sequence=sequence,
                        words=seq_tokens,
                        frequency=1,
                        avg_phi=phi,
                        max_phi=phi,
                        efficiency_gain=0,
                    )
    
    def _queue_for_save(self, text: str) -> None:
        """Queue an observation for batch save."""
        obs = self.phrase_observations.get(text)
        if obs:
            self.save_queue[text] = obs
        
        # Debounce saves - batch every 5 seconds
        if self.save_timer is None:
            self.save_timer = Timer(5.0, self._flush_save_queue)
            self.save_timer.start()
    
    def _flush_save_queue(self) -> None:
        """Flush pending saves to PostgreSQL with batch operations."""
        if not self.save_queue:
            return
        
        if not DB_AVAILABLE:
            logger.warning("[VocabularyTracker] Database not available")
            return
        
        to_save = list(self.save_queue.values())
        self.save_queue.clear()
        self.save_timer = None
        
        try:
            conn = get_db_connection()
            if not conn:
                logger.warning("[VocabularyTracker] Could not get database connection")
                # Re-queue for retry
                for obs in to_save:
                    self.save_queue[obs.text] = obs
                self.save_timer = Timer(30.0, self._flush_save_queue)
                self.save_timer.start()
                return
            
            try:
                cursor = conn.cursor()
                saved_count = 0
                batch_size = 50
                
                for i in range(0, len(to_save), batch_size):
                    batch = to_save[i:i + batch_size]
                    
                    for obs in batch:
                        # Using INSERT ... ON CONFLICT DO UPDATE pattern
                        cursor.execute(
                            """
                            INSERT INTO vocabulary_observations 
                                (text, type, phrase_category, is_real_word, frequency, 
                                 avg_phi, max_phi, efficiency_gain, first_seen, last_seen, contexts)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (text) DO UPDATE SET
                                frequency = EXCLUDED.frequency,
                                avg_phi = EXCLUDED.avg_phi,
                                max_phi = EXCLUDED.max_phi,
                                phrase_category = EXCLUDED.phrase_category,
                                last_seen = EXCLUDED.last_seen,
                                contexts = EXCLUDED.contexts
                            """,
                            (
                                obs.text,
                                obs.type,
                                obs.phrase_category.value,
                                obs.is_real_word,
                                obs.frequency,
                                obs.avg_phi,
                                obs.max_phi,
                                0,  # efficiency_gain for phrases
                                obs.first_seen,
                                obs.last_seen,
                                obs.contexts[:5],  # Limit context size
                            )
                        )
                        saved_count += 1
                
                conn.commit()
                logger.info(f"[VocabularyTracker] Saved {saved_count} observations to PostgreSQL")
                
            finally:
                cursor.close()
                conn.close()
                
        except Exception as e:
            logger.error(f"[VocabularyTracker] PostgreSQL save error: {e}")
            # Re-queue failed saves
            for obs in to_save:
                self.save_queue[obs.text] = obs
            self.save_timer = Timer(30.0, self._flush_save_queue)
            self.save_timer.start()
    
    def get_candidates(self, top_k: int = 20) -> List[VocabularyCandidate]:
        """
        Get vocabulary expansion candidates.
        
        Args:
            top_k: Number of top candidates to return
        
        Returns:
            List of vocabulary candidates sorted by combined score
        """
        candidates: List[VocabularyCandidate] = []
        
        # Phrases (mutations) with high frequency and Φ
        for obs in self.phrase_observations.values():
            if obs.frequency >= self.min_frequency and obs.avg_phi >= self.min_phi:
                candidates.append(VocabularyCandidate(
                    text=obs.text,
                    type=obs.type,
                    frequency=obs.frequency,
                    avg_phi=obs.avg_phi,
                    max_phi=obs.max_phi,
                    efficiency_gain=obs.frequency,
                    is_real_word=obs.is_real_word,
                    reasoning=(
                        f"{'Word' if obs.is_real_word else 'Phrase mutation'} "
                        f"in {obs.frequency} high-Φ results (avg Φ={obs.avg_phi:.2f})"
                    ),
                ))
        
        # Multi-word sequences
        for obs in self.sequence_observations.values():
            if (obs.frequency >= self.min_frequency and 
                obs.avg_phi >= self.min_phi and 
                obs.efficiency_gain > 5):
                candidates.append(VocabularyCandidate(
                    text=obs.sequence,
                    type='sequence',
                    frequency=obs.frequency,
                    avg_phi=obs.avg_phi,
                    max_phi=obs.max_phi,
                    efficiency_gain=obs.efficiency_gain,
                    is_real_word=False,
                    reasoning=(
                        f'Sequence "{obs.sequence}" appears {obs.frequency}x '
                        f'(avg Φ={obs.avg_phi:.2f})'
                    ),
                    components=obs.words,
                ))
        
        # Sort by combined score
        candidates.sort(key=lambda c: c.efficiency_gain * c.avg_phi, reverse=True)
        
        return candidates[:top_k]
    
    def get_stats(self) -> TrackerStats:
        """Get statistics with word/phrase breakdown."""
        words = [obs for obs in self.phrase_observations.values() if obs.is_real_word]
        phrases = [obs for obs in self.phrase_observations.values() if not obs.is_real_word]
        
        top_words = sorted(words, key=lambda o: o.frequency, reverse=True)[:20]
        top_phrases = sorted(phrases, key=lambda o: o.frequency, reverse=True)[:20]
        top_sequences = sorted(
            self.sequence_observations.values(),
            key=lambda o: o.efficiency_gain,
            reverse=True
        )[:20]
        
        return TrackerStats(
            total_words=len(words),
            total_phrases=len(phrases),
            total_sequences=len(self.sequence_observations),
            top_words=[
                {'text': o.text, 'frequency': o.frequency, 'avgPhi': o.avg_phi, 'isRealWord': True}
                for o in top_words
            ],
            top_phrases=[
                {'text': o.text, 'frequency': o.frequency, 'avgPhi': o.avg_phi}
                for o in top_phrases
            ],
            top_sequences=[
                {'sequence': o.sequence, 'frequency': o.frequency, 'avgPhi': o.avg_phi}
                for o in top_sequences
            ],
            candidates_ready=len(self.get_candidates(100)),
        )
    
    def get_category_stats(self) -> CategoryStats:
        """Get category statistics for kernel learning insights."""
        categories: Dict[str, Dict] = {
            cat.value: {'count': 0, 'total_phi': 0.0, 'examples': []}
            for cat in PhraseCategory
        }
        
        bip39_word_count = 0
        
        for obs in self.phrase_observations.values():
            cat = obs.phrase_category.value
            categories[cat]['count'] += 1
            categories[cat]['total_phi'] += obs.avg_phi
            if len(categories[cat]['examples']) < 5:
                categories[cat]['examples'].append(obs.text[:30])
            if obs.is_bip39_word:
                bip39_word_count += 1
        
        # Compute averages and filter empty categories
        result = {}
        for key, value in categories.items():
            if value['count'] > 0:
                result[key] = {
                    'count': value['count'],
                    'avgPhi': round(value['total_phi'] / value['count'], 4),
                    'examples': value['examples'],
                }
        
        total = len(self.phrase_observations)
        return CategoryStats(
            categories=result,
            bip39_coverage=round(bip39_word_count / total, 4) if total > 0 else 0.0,
            total_observations=total,
        )
    
    def save_to_storage(self) -> None:
        """Force save all observations to PostgreSQL."""
        if not DB_AVAILABLE:
            logger.warning("[VocabularyTracker] Database not available")
            return
        
        try:
            conn = get_db_connection()
            if not conn:
                logger.warning("[VocabularyTracker] Could not get database connection")
                return
            
            try:
                cursor = conn.cursor()
                
                # Save phrase observations
                for obs in self.phrase_observations.values():
                    cursor.execute(
                        """
                        INSERT INTO vocabulary_observations 
                            (text, type, phrase_category, is_real_word, frequency, 
                             avg_phi, max_phi, efficiency_gain, first_seen, last_seen, contexts)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (text) DO UPDATE SET
                            frequency = EXCLUDED.frequency,
                            avg_phi = EXCLUDED.avg_phi,
                            max_phi = EXCLUDED.max_phi,
                            phrase_category = EXCLUDED.phrase_category,
                            last_seen = EXCLUDED.last_seen,
                            contexts = EXCLUDED.contexts
                        """,
                        (
                            obs.text,
                            obs.type,
                            obs.phrase_category.value,
                            obs.is_real_word,
                            obs.frequency,
                            obs.avg_phi,
                            obs.max_phi,
                            0,
                            obs.first_seen,
                            obs.last_seen,
                            obs.contexts[:10],
                        )
                    )
                
                # Save sequence observations
                for obs in self.sequence_observations.values():
                    cursor.execute(
                        """
                        INSERT INTO vocabulary_observations 
                            (text, type, is_real_word, frequency, avg_phi, max_phi, 
                             efficiency_gain, first_seen, last_seen, contexts)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (text) DO UPDATE SET
                            frequency = EXCLUDED.frequency,
                            avg_phi = EXCLUDED.avg_phi,
                            max_phi = EXCLUDED.max_phi,
                            efficiency_gain = EXCLUDED.efficiency_gain,
                            last_seen = EXCLUDED.last_seen
                        """,
                        (
                            obs.sequence,
                            'sequence',
                            False,
                            obs.frequency,
                            obs.avg_phi,
                            obs.max_phi,
                            obs.efficiency_gain,
                            datetime.utcnow(),
                            datetime.utcnow(),
                            [obs.sequence],
                        )
                    )
                
                conn.commit()
                logger.info(
                    f"[VocabularyTracker] Saved {len(self.phrase_observations)} phrases, "
                    f"{len(self.sequence_observations)} sequences to PostgreSQL"
                )
                
            finally:
                cursor.close()
                conn.close()
                
        except Exception as e:
            logger.error(f"[VocabularyTracker] PostgreSQL save error: {e}")
            raise
    
    def _load_from_postgres(self) -> None:
        """Load from PostgreSQL."""
        if not DB_AVAILABLE:
            logger.warning("[VocabularyTracker] Database not available, starting empty")
            return
        
        try:
            conn = get_db_connection()
            if not conn:
                logger.warning("[VocabularyTracker] Could not get database connection")
                return
            
            try:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute("SELECT * FROM vocabulary_observations")
                rows = cursor.fetchall()
                
                for row in rows:
                    if row['type'] == 'sequence':
                        self.sequence_observations[row['text']] = SequenceObservation(
                            sequence=row['text'],
                            words=row['text'].split(' '),
                            frequency=row['frequency'],
                            avg_phi=row['avg_phi'],
                            max_phi=row['max_phi'],
                            efficiency_gain=row.get('efficiency_gain', 0) or 0,
                        )
                    else:
                        # Map string to enum safely
                        try:
                            category = PhraseCategory(row.get('phrase_category', 'unknown'))
                        except ValueError:
                            category = PhraseCategory.UNKNOWN
                        
                        self.phrase_observations[row['text']] = PhraseObservation(
                            text=row['text'],
                            frequency=row['frequency'],
                            avg_phi=row['avg_phi'],
                            max_phi=row['max_phi'],
                            first_seen=row.get('first_seen') or datetime.utcnow(),
                            last_seen=row.get('last_seen') or datetime.utcnow(),
                            contexts=row.get('contexts') or [],
                            is_real_word=row['is_real_word'],
                            is_bip39_word=row['text'].lower() in BIP39_WORDS,
                            type=row['type'],
                            phrase_category=category,
                        )
                
                logger.info(
                    f"[VocabularyTracker] Loaded {len(self.phrase_observations)} phrases, "
                    f"{len(self.sequence_observations)} sequences from PostgreSQL"
                )
                
            finally:
                cursor.close()
                conn.close()
                
        except Exception as e:
            logger.warning(f"[VocabularyTracker] Failed to load from PostgreSQL: {e}")
    
    def export_for_tokenizer(self) -> List[Dict]:
        """
        Export observations for Python tokenizer with phrase category.
        
        Kernels use this to learn the difference between BIP-39 seeds and passphrases.
        
        Returns:
            List of observations with text, frequency, phi scores, and category
        """
        exports = []
        
        # Export phrase observations with category
        for obs in self.phrase_observations.values():
            if obs.frequency >= self.min_frequency and obs.avg_phi >= self.min_phi:
                exports.append({
                    'text': obs.text,
                    'frequency': obs.frequency,
                    'avgPhi': obs.avg_phi,
                    'maxPhi': obs.max_phi,
                    'type': obs.type,
                    'isRealWord': obs.is_real_word,
                    'isBip39Word': obs.is_bip39_word,
                    'phraseCategory': obs.phrase_category.value,
                })
        
        # Export sequence observations
        for obs in self.sequence_observations.values():
            if obs.frequency >= 3 and obs.avg_phi >= 0.4:
                category = self.classify_phrase_category(obs.sequence)
                exports.append({
                    'text': obs.sequence,
                    'frequency': obs.frequency,
                    'avgPhi': obs.avg_phi,
                    'maxPhi': obs.max_phi,
                    'type': 'sequence',
                    'isRealWord': False,
                    'isBip39Word': False,
                    'phraseCategory': category.value,
                })
        
        # Sort by avgPhi descending
        exports.sort(key=lambda e: e['avgPhi'], reverse=True)
        
        # Log category distribution for debugging
        categories = {}
        for e in exports:
            cat = e['phraseCategory']
            categories[cat] = categories.get(cat, 0) + 1
        
        logger.info(f"[VocabularyTracker] Exported {len(exports)} observations: {categories}")
        
        return exports
    
    def is_data_loaded(self) -> bool:
        """Check if data has finished loading (non-blocking)."""
        return len(self.phrase_observations) > 0 or len(self.sequence_observations) > 0


# Singleton instance
vocabulary_tracker = VocabularyTracker()
