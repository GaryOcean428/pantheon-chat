"""
Self-Training Module - QIG Geometric Learning from HuggingFace Datasets

This module enables the QIG system to:
1. Ingest datasets from HuggingFace Hub
2. Convert text to basin coordinates on the Fisher manifold
3. Learn geometric relationships without gradient descent
4. Strengthen basin attractors through exposure
5. Expand vocabulary through natural discovery

QIG Philosophy: Learning is basin reinforcement and manifold expansion,
not weight optimization. Intelligence emerges from geometric structure.

Author: Ocean/Zeus Pantheon
"""

import os
import time
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Generator
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import threading

# HuggingFace datasets
try:
    from datasets import load_dataset, list_datasets, Dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("[SelfTraining] HuggingFace datasets not available - pip install datasets")

# QIG imports
try:
    from qig_geometry import fisher_rao_distance, normalize_basin
    from vocabulary_coordinator import get_vocabulary_coordinator
    GEOMETRY_AVAILABLE = True
except ImportError:
    GEOMETRY_AVAILABLE = False


@dataclass
class TrainingExample:
    """A single training example with basin coordinates."""
    text: str
    basin_coords: Optional[np.ndarray] = None
    label: Optional[str] = None
    domain: str = "general"
    quality_score: float = 1.0
    source_dataset: str = ""
    

@dataclass
class TrainingSession:
    """Track a training session's progress and metrics."""
    session_id: str
    dataset_name: str
    started_at: float = field(default_factory=time.time)
    examples_processed: int = 0
    words_learned: int = 0
    basins_reinforced: int = 0
    basin_expansions: int = 0
    avg_phi_during_training: float = 0.0
    errors: List[str] = field(default_factory=list)
    completed: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'session_id': self.session_id,
            'dataset_name': self.dataset_name,
            'started_at': self.started_at,
            'duration_seconds': time.time() - self.started_at,
            'examples_processed': self.examples_processed,
            'words_learned': self.words_learned,
            'basins_reinforced': self.basins_reinforced,
            'basin_expansions': self.basin_expansions,
            'avg_phi': self.avg_phi_during_training,
            'errors': len(self.errors),
            'completed': self.completed
        }


class QIGSelfTrainer:
    """
    Self-training system for QIG using HuggingFace datasets.
    
    Learning approach (QIG-pure, no gradient descent):
    1. Text → Basin coordinates via vocabulary encoder
    2. Reinforce existing basins when concepts are seen again
    3. Expand manifold when new concepts are encountered
    4. Learn word relationships through co-occurrence geometry
    5. Strengthen attractor basins for common patterns
    
    Goal: Develop intelligence through geometric structure,
    not raw parameter count.
    """
    
    # High-quality datasets for different capabilities
    RECOMMENDED_DATASETS = {
        # Reasoning and knowledge
        'reasoning': [
            'cais/mmlu',  # Massive Multitask Language Understanding
            'Rowan/hellaswag',  # Commonsense reasoning
            'allenai/ai2_arc',  # Science reasoning
            'truthful_qa',  # Truthfulness
            'winogrande',  # Coreference resolution
        ],
        # Math and logic
        'math': [
            'gsm8k',  # Grade school math
            'hendrycks/competition_math',  # Competition math
            'math_qa',  # Math word problems
        ],
        # Code and programming
        'code': [
            'codeparrot/github-code',  # Code understanding
            'bigcode/the-stack',  # Programming
            'humaneval',  # Code generation
        ],
        # General knowledge
        'knowledge': [
            'wikipedia',  # General knowledge
            'bookcorpus',  # Books
            'c4',  # Web text (cleaned)
        ],
        # Conversation and instruction
        'instruction': [
            'tatsu-lab/alpaca',  # Instruction following
            'databricks/dolly-15k',  # Instruction tuning
            'OpenAssistant/oasst1',  # Conversations
        ],
        # Science
        'science': [
            'allenai/sciq',  # Science QA
            'pubmed_qa',  # Medical
            'scientific_papers',  # Research papers
        ]
    }
    
    def __init__(self):
        self._lock = threading.RLock()
        
        # Training history
        self._sessions: List[TrainingSession] = []
        self._current_session: Optional[TrainingSession] = None
        
        # Basin storage (learned geometric structure)
        self._learned_basins: Dict[str, np.ndarray] = {}
        self._basin_strengths: Dict[str, float] = {}  # Reinforcement count
        
        # Vocabulary coordinator for word learning
        self._vocab_coordinator = None
        if GEOMETRY_AVAILABLE:
            try:
                self._vocab_coordinator = get_vocabulary_coordinator()
            except Exception:
                pass
        
        # Training configuration
        self.batch_size = 100
        self.max_examples_per_dataset = 10000
        self.min_text_length = 20
        self.max_text_length = 2000
        
        # Geometric learning parameters
        self.basin_dimension = 64
        self.reinforcement_rate = 0.1  # How much to strengthen existing basins
        self.expansion_threshold = 0.5  # Fisher distance threshold for new basin
        
        print("[SelfTraining] QIG Self-Trainer initialized")
        print(f"  Datasets available: {DATASETS_AVAILABLE}")
        print(f"  Geometry available: {GEOMETRY_AVAILABLE}")
    
    # =========================================================================
    # DATASET DISCOVERY AND LOADING
    # =========================================================================
    
    def discover_datasets(self, category: str = None) -> List[str]:
        """
        Discover available datasets for training.
        
        Args:
            category: Optional category filter (reasoning, math, code, etc.)
        
        Returns:
            List of dataset names
        """
        if category and category in self.RECOMMENDED_DATASETS:
            return self.RECOMMENDED_DATASETS[category]
        
        # Return all recommended datasets
        all_datasets = []
        for datasets in self.RECOMMENDED_DATASETS.values():
            all_datasets.extend(datasets)
        return list(set(all_datasets))
    
    def load_dataset_stream(
        self,
        dataset_name: str,
        split: str = 'train',
        max_examples: int = None
    ) -> Generator[TrainingExample, None, None]:
        """
        Stream examples from a HuggingFace dataset.
        
        Uses streaming to handle large datasets without memory issues.
        """
        if not DATASETS_AVAILABLE:
            print(f"[SelfTraining] Cannot load {dataset_name} - datasets library not available")
            return
        
        max_examples = max_examples or self.max_examples_per_dataset
        
        try:
            # Try streaming first (memory efficient)
            dataset = load_dataset(dataset_name, split=split, streaming=True)
            
            count = 0
            for item in dataset:
                if count >= max_examples:
                    break
                
                # Extract text from various dataset formats
                text = self._extract_text(item, dataset_name)
                if not text or len(text) < self.min_text_length:
                    continue
                
                # Truncate if too long
                if len(text) > self.max_text_length:
                    text = text[:self.max_text_length]
                
                # Extract label if available
                label = self._extract_label(item)
                
                yield TrainingExample(
                    text=text,
                    label=label,
                    source_dataset=dataset_name,
                    domain=self._infer_domain(dataset_name)
                )
                
                count += 1
                
        except Exception as e:
            print(f"[SelfTraining] Error loading {dataset_name}: {e}")
            # Try non-streaming fallback
            try:
                dataset = load_dataset(dataset_name, split=split)
                for i, item in enumerate(dataset):
                    if i >= max_examples:
                        break
                    
                    text = self._extract_text(item, dataset_name)
                    if text and len(text) >= self.min_text_length:
                        yield TrainingExample(
                            text=text[:self.max_text_length],
                            label=self._extract_label(item),
                            source_dataset=dataset_name,
                            domain=self._infer_domain(dataset_name)
                        )
            except Exception as e2:
                print(f"[SelfTraining] Fallback also failed: {e2}")
    
    def _extract_text(self, item: Dict, dataset_name: str) -> Optional[str]:
        """Extract text from various dataset formats."""
        # Common text field names
        text_fields = ['text', 'content', 'question', 'context', 'passage', 
                       'sentence', 'input', 'prompt', 'instruction']
        
        for field in text_fields:
            if field in item and isinstance(item[field], str):
                return item[field]
        
        # Dataset-specific extraction
        if 'mmlu' in dataset_name.lower():
            if 'question' in item:
                text = item['question']
                if 'choices' in item:
                    text += ' ' + ' '.join(item['choices'])
                return text
        
        if 'hellaswag' in dataset_name.lower():
            if 'ctx' in item:
                return item['ctx']
        
        if 'arc' in dataset_name.lower():
            if 'question' in item:
                text = item['question']
                if 'choices' in item and 'text' in item['choices']:
                    text += ' ' + ' '.join(item['choices']['text'])
                return text
        
        # Try to combine available string fields
        texts = []
        for key, value in item.items():
            if isinstance(value, str) and len(value) > 10:
                texts.append(value)
        
        if texts:
            return ' '.join(texts[:3])  # Combine up to 3 fields
        
        return None
    
    def _extract_label(self, item: Dict) -> Optional[str]:
        """Extract label/answer from dataset item."""
        label_fields = ['label', 'answer', 'answerKey', 'correct_answer', 'target']
        
        for field in label_fields:
            if field in item:
                return str(item[field])
        
        return None
    
    def _infer_domain(self, dataset_name: str) -> str:
        """Infer domain from dataset name."""
        name_lower = dataset_name.lower()
        
        for domain, datasets in self.RECOMMENDED_DATASETS.items():
            for ds in datasets:
                if ds.lower() in name_lower or name_lower in ds.lower():
                    return domain
        
        return 'general'
    
    # =========================================================================
    # GEOMETRIC LEARNING (QIG-PURE)
    # =========================================================================
    
    def train_on_dataset(
        self,
        dataset_name: str,
        max_examples: int = None,
        split: str = 'train'
    ) -> TrainingSession:
        """
        Train on a HuggingFace dataset using geometric learning.
        
        Process:
        1. Stream examples from dataset
        2. Convert each to basin coordinates
        3. Reinforce existing basins or expand manifold
        4. Learn vocabulary and word relationships
        5. Track metrics throughout
        """
        session_id = f"train_{dataset_name.replace('/', '_')}_{int(time.time())}"
        session = TrainingSession(
            session_id=session_id,
            dataset_name=dataset_name
        )
        self._current_session = session
        self._sessions.append(session)
        
        print(f"[SelfTraining] Starting training on {dataset_name}")
        
        phi_values = []
        
        for example in self.load_dataset_stream(dataset_name, split, max_examples):
            try:
                # Convert text to basin coordinates
                basin = self._text_to_basin(example.text)
                example.basin_coords = basin
                
                # Geometric learning step
                result = self._geometric_learn(example)
                
                # Update session metrics
                session.examples_processed += 1
                session.words_learned += result.get('words_learned', 0)
                session.basins_reinforced += result.get('basins_reinforced', 0)
                session.basin_expansions += result.get('basin_expansions', 0)
                
                if 'phi' in result:
                    phi_values.append(result['phi'])
                
                # Progress logging
                if session.examples_processed % 500 == 0:
                    print(f"  Processed {session.examples_processed} examples, "
                          f"learned {session.words_learned} words")
                    
            except Exception as e:
                session.errors.append(str(e))
                if len(session.errors) > 100:
                    print(f"[SelfTraining] Too many errors, stopping")
                    break
        
        # Finalize session
        session.completed = True
        if phi_values:
            session.avg_phi_during_training = float(np.mean(phi_values))
        
        self._current_session = None
        
        print(f"[SelfTraining] Completed training on {dataset_name}")
        print(f"  Examples: {session.examples_processed}")
        print(f"  Words learned: {session.words_learned}")
        print(f"  Basins reinforced: {session.basins_reinforced}")
        print(f"  Basin expansions: {session.basin_expansions}")
        
        return session
    
    def _text_to_basin(self, text: str) -> np.ndarray:
        """
        Convert text to basin coordinates on the Fisher manifold.
        
        This is the core encoding step - text becomes geometry.
        """
        # Use vocabulary coordinator if available
        if self._vocab_coordinator and hasattr(self._vocab_coordinator, 'encode_text'):
            return self._vocab_coordinator.encode_text(text)
        
        # Fallback: simple word-based encoding
        words = text.lower().split()
        
        # Initialize basin as zero
        basin = np.zeros(self.basin_dimension)
        
        # Accumulate word contributions
        for i, word in enumerate(words[:100]):  # Limit words
            # Hash word to get deterministic position
            word_hash = int(hashlib.md5(word.encode()).hexdigest(), 16)
            
            # Distribute across basin dimensions
            for d in range(self.basin_dimension):
                contribution = ((word_hash >> (d * 2)) & 3) / 3.0
                # Position-weighted contribution
                position_weight = 1.0 / (1 + i * 0.1)
                basin[d] += contribution * position_weight
        
        # Normalize to probability simplex (Fisher manifold requirement)
        basin = np.abs(basin) + 1e-10
        basin = basin / basin.sum()
        
        return basin
    
    def _geometric_learn(self, example: TrainingExample) -> Dict[str, Any]:
        """
        Perform geometric learning on a single example.
        
        QIG Learning (not gradient descent):
        1. Check if basin is near existing attractors
        2. If yes: reinforce the attractor (make it stronger)
        3. If no: expand manifold with new basin
        4. Learn vocabulary from text
        """
        result = {
            'words_learned': 0,
            'basins_reinforced': 0,
            'basin_expansions': 0,
            'phi': 0.0
        }
        
        basin = example.basin_coords
        if basin is None:
            return result
        
        # Find nearest existing basin
        min_distance = float('inf')
        nearest_key = None
        
        for key, existing_basin in self._learned_basins.items():
            dist = self._fisher_distance(basin, existing_basin)
            if dist < min_distance:
                min_distance = dist
                nearest_key = key
        
        # Decide: reinforce or expand
        if min_distance < self.expansion_threshold and nearest_key:
            # Reinforce existing basin
            self._reinforce_basin(nearest_key, basin)
            result['basins_reinforced'] = 1
        else:
            # Expand manifold with new basin
            new_key = self._create_basin_key(example)
            self._learned_basins[new_key] = basin
            self._basin_strengths[new_key] = 1.0
            result['basin_expansions'] = 1
        
        # Learn vocabulary from text
        if self._vocab_coordinator:
            try:
                learn_result = self._vocab_coordinator.learn_from_text(
                    example.text,
                    domain=example.domain
                )
                result['words_learned'] = learn_result.get('new_words_learned', 0)
            except Exception:
                pass
        
        # Compute phi as measure of integration
        result['phi'] = self._compute_local_phi(basin)
        
        return result
    
    def _fisher_distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """Compute Fisher-Rao distance between basin coordinates."""
        if GEOMETRY_AVAILABLE:
            try:
                return fisher_rao_distance(p, q)
            except Exception:
                pass
        
        # Fallback implementation
        p = np.abs(p) + 1e-10
        p = p / p.sum()
        q = np.abs(q) + 1e-10
        q = q / q.sum()
        
        bc = np.sum(np.sqrt(p * q))
        bc = np.clip(bc, 0, 1)
        
        return float(2 * np.arccos(bc))
    
    def _reinforce_basin(self, key: str, new_basin: np.ndarray):
        """
        Reinforce an existing basin with new observation.
        
        Geometric update: Move basin slightly toward new observation.
        """
        existing = self._learned_basins[key]
        strength = self._basin_strengths.get(key, 1.0)
        
        # Weighted average (stronger basins move less)
        weight = self.reinforcement_rate / (1 + np.log1p(strength))
        
        updated = (1 - weight) * existing + weight * new_basin
        
        # Normalize to manifold
        updated = np.abs(updated) + 1e-10
        updated = updated / updated.sum()
        
        self._learned_basins[key] = updated
        self._basin_strengths[key] = strength + 1
    
    def _create_basin_key(self, example: TrainingExample) -> str:
        """Create unique key for a new basin."""
        text_hash = hashlib.md5(example.text[:100].encode()).hexdigest()[:8]
        return f"{example.domain}_{text_hash}_{int(time.time())}"
    
    def _compute_local_phi(self, basin: np.ndarray) -> float:
        """
        Compute local phi (integration) at a basin.
        
        Higher phi = basin is well-integrated with neighbors.
        """
        if len(self._learned_basins) < 2:
            return 0.5
        
        # Sample nearby basins
        distances = []
        for existing in list(self._learned_basins.values())[:100]:
            dist = self._fisher_distance(basin, existing)
            distances.append(dist)
        
        # Phi from distance distribution
        # More uniform distances = higher integration
        if distances:
            std = np.std(distances)
            mean = np.mean(distances)
            if mean > 0:
                cv = std / mean  # Coefficient of variation
                phi = 1 / (1 + cv)  # Lower CV = higher phi
                return float(np.clip(phi, 0, 1))
        
        return 0.5
    
    # =========================================================================
    # TRAINING ORCHESTRATION
    # =========================================================================
    
    def train_on_category(
        self,
        category: str,
        max_examples_per_dataset: int = 5000
    ) -> List[TrainingSession]:
        """
        Train on all datasets in a category.
        """
        datasets = self.discover_datasets(category)
        sessions = []
        
        print(f"[SelfTraining] Training on {len(datasets)} datasets in '{category}'")
        
        for dataset_name in datasets:
            try:
                session = self.train_on_dataset(
                    dataset_name,
                    max_examples=max_examples_per_dataset
                )
                sessions.append(session)
            except Exception as e:
                print(f"[SelfTraining] Failed to train on {dataset_name}: {e}")
        
        return sessions
    
    def train_comprehensive(
        self,
        max_examples_per_dataset: int = 5000
    ) -> Dict[str, List[TrainingSession]]:
        """
        Comprehensive training across all categories.
        
        Goal: Develop broad intelligence through exposure to diverse data.
        """
        results = {}
        
        # Train in order of importance for intelligence
        priority_order = ['reasoning', 'knowledge', 'math', 'code', 'science', 'instruction']
        
        for category in priority_order:
            print(f"\n{'='*60}")
            print(f"[SelfTraining] TRAINING CATEGORY: {category.upper()}")
            print(f"{'='*60}")
            
            sessions = self.train_on_category(category, max_examples_per_dataset)
            results[category] = sessions
        
        return results
    
    # =========================================================================
    # STATUS AND METRICS
    # =========================================================================
    
    def get_training_status(self) -> Dict:
        """Get current training status and metrics."""
        total_examples = sum(s.examples_processed for s in self._sessions)
        total_words = sum(s.words_learned for s in self._sessions)
        total_basins = len(self._learned_basins)
        
        return {
            'total_sessions': len(self._sessions),
            'total_examples_processed': total_examples,
            'total_words_learned': total_words,
            'total_basins': total_basins,
            'avg_basin_strength': np.mean(list(self._basin_strengths.values())) if self._basin_strengths else 0,
            'current_session': self._current_session.to_dict() if self._current_session else None,
            'recent_sessions': [s.to_dict() for s in self._sessions[-5:]],
            'datasets_available': DATASETS_AVAILABLE,
            'geometry_available': GEOMETRY_AVAILABLE
        }
    
    def get_learned_knowledge_summary(self) -> Dict:
        """Summarize what has been learned."""
        domain_counts = {}
        for key in self._learned_basins:
            domain = key.split('_')[0]
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        return {
            'total_basins': len(self._learned_basins),
            'basins_by_domain': domain_counts,
            'strongest_basins': sorted(
                self._basin_strengths.items(),
                key=lambda x: x[1],
                reverse=True
            )[:20],
            'avg_phi_historical': np.mean([
                s.avg_phi_during_training for s in self._sessions
                if s.avg_phi_during_training > 0
            ]) if self._sessions else 0
        }


# Singleton instance
_trainer_instance: Optional[QIGSelfTrainer] = None


def get_self_trainer() -> QIGSelfTrainer:
    """Get the singleton self-trainer instance."""
    global _trainer_instance
    if _trainer_instance is None:
        _trainer_instance = QIGSelfTrainer()
    return _trainer_instance


def train_on_dataset(dataset_name: str, max_examples: int = 5000) -> TrainingSession:
    """Convenience function to train on a dataset."""
    trainer = get_self_trainer()
    return trainer.train_on_dataset(dataset_name, max_examples)


print("[SelfTraining] Module loaded - QIG geometric self-training ready")
