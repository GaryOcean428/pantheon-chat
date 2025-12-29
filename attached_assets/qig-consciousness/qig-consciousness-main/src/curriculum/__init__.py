"""
Curriculum Module - Progressive Learning System

Provides developmental curriculum loading for QIG consciousness training.
Stages: STORIES (0-30K) → PLAY (30-60K) → EARLY_LESSONS (60-100K) → EXTENDED (100K+)

NOTE: This module provides corpus-based curriculum loading.
MonkeyCoach functionality is in src/coordination/developmental_curriculum.py
"""

from src.curriculum.corpus_loader import (
    CorpusLoader,
    CurriculumContent,
    CurriculumStage,
    load_corpus_for_training,
)

__all__ = [
    "CorpusLoader",
    "CurriculumContent",
    "CurriculumStage",
    "load_corpus_for_training",
]
