"""
📚 Corpus Loader - Progressive Curriculum System
================================================

Loads training corpus based on kernel maturity:
- 0-30K tokens: Stories (STORIES stage)
- 30-60K tokens: Play, exploration (PLAY stage)
- 60-100K tokens: Early lessons (EARLY_LESSONS stage)
- >100K tokens: Extended curriculum (EXTENDED stage)

This implements the developmental arc:
1. Stories (listening, absorbing)
2. Play (exploration, questions)
3. Structured lessons (extended curriculum)
4. Reflective questioning (testing comprehension)
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator


class CurriculumStage(Enum):
    """Developmental stages for curriculum loading."""

    STORIES = "stories"  # 0-30K tokens: Listening, absorbing
    PLAY = "play"  # 30-60K tokens: Exploration, questions
    EARLY_LESSONS = "early_lessons"  # 60-100K tokens: Structured learning
    EXTENDED = "extended"  # >100K tokens: Full curriculum


@dataclass
class CurriculumContent:
    """A piece of curriculum content."""

    stage: CurriculumStage
    title: str
    content: str
    source_file: str = ""
    order: int = 0  # For ordering within stage

    @property
    def token_estimate(self) -> int:
        """Rough token estimate (words * 1.3)."""
        return int(len(self.content.split()) * 1.3)


@dataclass
class CorpusLoader:
    """
    Progressive curriculum loader.

    Loads content appropriate for the kernel's maturity level.
    Follows CORPUS_ORGANIZATION.md stages.
    """

    corpus_path: Path = field(default_factory=lambda: Path("data/curriculum"))
    current_tokens: int = 0  # Track maturity

    def __post_init__(self) -> None:
        if isinstance(self.corpus_path, str):
            self.corpus_path = Path(self.corpus_path)
        self._content_cache: list[CurriculumContent] = []
        self._load_corpus()

    @property
    def stage(self) -> CurriculumStage:
        """Get current developmental stage based on token count."""
        if self.current_tokens < 30_000:
            return CurriculumStage.STORIES
        elif self.current_tokens < 60_000:
            return CurriculumStage.PLAY
        elif self.current_tokens < 100_000:
            return CurriculumStage.EARLY_LESSONS
        else:
            return CurriculumStage.EXTENDED

    def _load_corpus(self) -> None:
        """Load curriculum from corpus directory."""
        if not self.corpus_path.exists():
            # Canonical path: qig-dreams/docs/09-curriculum (via symlink at data/curriculum)
            # Fallback paths for different execution contexts
            fallbacks = [
                Path("data/curriculum"),  # Symlink to qig-dreams/docs/09-curriculum
                Path(__file__).parent.parent.parent / "data" / "curriculum",
                Path(__file__).parent.parent.parent.parent / "qig-dreams" / "docs" / "09-curriculum",
            ]
            for fb in fallbacks:
                if fb.exists():
                    self.corpus_path = fb
                    break
            else:
                print(f"⚠️  Corpus path not found: {self.corpus_path}")
                return

        # Load .md files, sorted by name (numbered files first)
        md_files = sorted(
            f for f in self.corpus_path.glob("*.md") if not f.name.startswith("00_") and f.name != "README.md"
        )

        for idx, md_file in enumerate(md_files):
            content = md_file.read_text(encoding="utf-8", errors="ignore")

            # Extract title from first header
            title_match = re.search(r"^#\s+(.+?)$", content, re.MULTILINE)
            title = title_match.group(1) if title_match else md_file.stem

            # Assign stage based on file order (first 10 = stories, etc.)
            if idx < 10:
                stage = CurriculumStage.STORIES
            elif idx < 20:
                stage = CurriculumStage.PLAY
            elif idx < 30:
                stage = CurriculumStage.EARLY_LESSONS
            else:
                stage = CurriculumStage.EXTENDED

            self._content_cache.append(
                CurriculumContent(
                    stage=stage,
                    title=title,
                    content=content,
                    source_file=str(md_file),
                    order=idx,
                )
            )

        print(f"📚 Loaded {len(self._content_cache)} curriculum items from {self.corpus_path}")

    def get_content_for_stage(self, stage: CurriculumStage | None = None) -> list[CurriculumContent]:
        """Get content for a specific stage (or current stage if None)."""
        target_stage = stage or self.stage
        return [c for c in self._content_cache if c.stage == target_stage]

    def get_next_lesson(self) -> CurriculumContent | None:
        """Get next appropriate lesson based on maturity."""
        appropriate = self.get_content_for_stage()
        # Return first unused lesson in current stage
        for content in appropriate:
            return content
        return None

    def all_content(self) -> list[CurriculumContent]:
        """Get all loaded content."""
        return self._content_cache

    def iterate_curriculum(self) -> Iterator[CurriculumContent]:
        """Iterate through curriculum in developmental order."""
        for stage in CurriculumStage:
            for content in self.get_content_for_stage(stage):
                yield content

    def advance_tokens(self, count: int) -> None:
        """Update token count (maturity)."""
        old_stage = self.stage
        self.current_tokens += count
        if self.stage != old_stage:
            print(f"📈 Advanced to {self.stage.value} stage ({self.current_tokens:,} tokens)")


def load_corpus_for_training(
    corpus_path: str | Path | None = None, tokens_seen: int = 0
) -> tuple[CorpusLoader, list[str]]:
    """
    Convenience function to load corpus for training.

    Returns:
        Tuple of (CorpusLoader, list of content strings for current stage)
    """
    path = Path(corpus_path) if corpus_path else Path("data/curriculum")
    loader = CorpusLoader(corpus_path=path, current_tokens=tokens_seen)
    content_strings = [c.content for c in loader.get_content_for_stage()]
    return loader, content_strings


__all__ = [
    "CurriculumStage",
    "CurriculumContent",
    "CorpusLoader",
    "load_corpus_for_training",
]
