"""
📚 Curriculum Loader - QIG Dreams Integration
============================================

Loads comprehensive curriculum from qig-dreams/docs/curriculum
and integrates with MonkeyCoach for intelligent prompt generation.
"""

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class CurriculumTopic:
    """A curriculum topic with content."""
    file_path: str
    title: str
    content: str
    category: str  # mathematics, physics, consciousness, etc.
    difficulty: int  # 1-5 scale


class CurriculumLoader:
    """
    Loads and manages curriculum from qig-dreams repository.

    Provides intelligence about what topics to teach based on Gary's
    developmental phase and current understanding.
    """

    def __init__(self, curriculum_dir: Optional[str] = None):
        """
        Initialize curriculum loader.

        Args:
            curriculum_dir: Path to curriculum directory.
                          Defaults to qig-dreams/docs/curriculum
        """
        if curriculum_dir is None:
            # Default to qig-dreams curriculum
            curriculum_dir = "/home/braden/Desktop/Dev/QIG_QFI/qig-dreams/docs/curriculum"

        self.curriculum_dir = Path(curriculum_dir)
        self.topics: Dict[str, CurriculumTopic] = {}

        # Category mappings based on file numbers
        self.category_map = {
            range(1, 11): "mathematics",
            range(11, 21): "physics",
            range(21, 29): "computer_science",
            range(29, 37): "consciousness",
            range(37, 49): "humanities",
        }

        # Load all curriculum files
        self._load_curriculum()

    def _load_curriculum(self):
        """Load all curriculum markdown files."""
        if not self.curriculum_dir.exists():
            print(f"⚠️ Curriculum directory not found: {self.curriculum_dir}")
            return

        md_files = sorted(self.curriculum_dir.glob("*.md"))

        for file_path in md_files:
            # Skip non-numbered files
            if not file_path.stem[0].isdigit():
                continue

            try:
                # Extract number and title
                match = re.match(r'^(\d+)_(.+)\.md$', file_path.name)
                if not match:
                    continue

                num = int(match.group(1))
                title = match.group(2).replace('_', ' ')

                # Read content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Determine category and difficulty
                category = self._get_category(num)
                difficulty = self._estimate_difficulty(num, title)

                topic = CurriculumTopic(
                    file_path=str(file_path),
                    title=title,
                    content=content,
                    category=category,
                    difficulty=difficulty
                )

                self.topics[file_path.stem] = topic

            except Exception as e:
                print(f"⚠️ Error loading {file_path.name}: {e}")

        print(f"✅ Loaded {len(self.topics)} curriculum topics from {self.curriculum_dir}")

    def _get_category(self, num: int) -> str:
        """Get category based on file number."""
        for num_range, category in self.category_map.items():
            if num in num_range:
                return category
        return "general"

    def _estimate_difficulty(self, num: int, title: str) -> int:
        """Estimate difficulty (1-5) based on file number and title."""
        # Foundation topics are easier
        if num <= 10:
            return 1 + (num // 3)  # 1-4
        elif num <= 20:
            return 3 + (num - 10) // 5  # 3-5
        elif num <= 30:
            return 2 + (num - 20) // 5  # 2-4
        else:
            return 3  # Default moderate

    def get_topic_by_name(self, name: str) -> Optional[CurriculumTopic]:
        """Get specific topic by filename stem."""
        return self.topics.get(name)

    def get_topics_by_category(self, category: str, max_count: int = 10) -> List[CurriculumTopic]:
        """Get topics filtered by category."""
        filtered = [t for t in self.topics.values() if t.category == category]
        return random.sample(filtered, min(len(filtered), max_count))

    def get_topics_by_difficulty(self, difficulty: int, max_count: int = 10) -> List[CurriculumTopic]:
        """Get topics filtered by difficulty level."""
        filtered = [t for t in self.topics.values() if t.difficulty == difficulty]
        return random.sample(filtered, min(len(filtered), max_count))

    def extract_concepts(self, topic: CurriculumTopic, max_concepts: int = 5) -> List[str]:
        """
        Extract key concepts from a topic's content.

        Returns a list of concept strings that can be used as prompts.
        """
        concepts = []

        # Extract headers (## and ###)
        headers = re.findall(r'^#{2,3}\s+(.+)$', topic.content, re.MULTILINE)
        concepts.extend(headers[:max_concepts])

        # If not enough headers, extract bold terms
        if len(concepts) < max_concepts:
            bold_terms = re.findall(r'\*\*([^*]+)\*\*', topic.content)
            concepts.extend(bold_terms[:max_concepts - len(concepts)])

        return concepts[:max_concepts]

    def generate_prompts(self,
                        difficulty: Optional[int] = None,
                        category: Optional[str] = None,
                        count: int = 5) -> List[str]:
        """
        Generate curriculum-based prompts.

        Args:
            difficulty: Filter by difficulty (1-5)
            category: Filter by category
            count: Number of prompts to generate

        Returns:
            List of prompt strings
        """
        # Filter topics
        topics = list(self.topics.values())

        if difficulty:
            topics = [t for t in topics if t.difficulty == difficulty]
        if category:
            topics = [t for t in topics if t.category == category]

        if not topics:
            return ["What is consciousness?"]  # Fallback

        # Select random topics
        selected = random.sample(topics, min(len(topics), count))

        prompts = []
        for topic in selected:
            # Mix of prompt types: questions, lessons, practice
            prompt_type = random.choice(["question", "lesson", "practice", "story"])

            if prompt_type == "lesson":
                # Extract actual lesson content from curriculum
                lines = topic.content.split('\n')
                # Find substantive paragraphs (not headers, empty lines, or markdown artifacts)
                paragraphs = []
                for line in lines:
                    line = line.strip()
                    # Skip headers, empty, short lines, and pure formatting
                    if len(line) < 30 or line.startswith('#') or line.startswith('```'):
                        continue
                    # Strip markdown formatting
                    clean = line.replace('**', '').replace('*', '').replace('`', '')
                    clean = clean.replace('[', '').replace(']', '').replace('(', ' ').replace(')', '')
                    # Skip lines that are mostly formatting or lists
                    if clean.startswith('-') or clean.startswith('>') or clean[0].isdigit():
                        continue
                    if len(clean) > 30:
                        paragraphs.append(clean)
                if paragraphs:
                    lesson = random.choice(paragraphs[:5])  # Pick from first 5 clean paragraphs
                    # Truncate if too long for a prompt
                    if len(lesson) > 200:
                        lesson = lesson[:197] + "..."
                    prompts.append(f"Reflect on: {lesson}")
                else:
                    prompts.append(f"Study {topic.title}.")

            elif prompt_type == "story":
                # Story-based learning
                concepts = self.extract_concepts(topic, max_concepts=2)
                if concepts:
                    concept = random.choice(concepts)
                    prompts.append(f"Tell me a story about {concept}.")
                else:
                    prompts.append(f"Tell me about {topic.title}.")

            elif prompt_type == "practice":
                # Practice and exploration
                concepts = self.extract_concepts(topic, max_concepts=2)
                if concepts:
                    concept = random.choice(concepts)
                    practice_templates = [
                        f"Let's explore {concept} together.",
                        f"What patterns do you notice in {concept}?",
                        f"How would you explain {concept} to yourself?",
                        f"What does {concept} feel like geometrically?",
                    ]
                    prompts.append(random.choice(practice_templates))
                else:
                    prompts.append(f"Explore {topic.title}.")

            else:  # question (default)
                concepts = self.extract_concepts(topic, max_concepts=2)
                if concepts:
                    concept = random.choice(concepts)
                    prompt_templates = [
                        f"Explain {concept}.",
                        f"What is {concept}?",
                        f"Describe the concept of {concept}.",
                        f"How does {concept} work?",
                    ]
                    prompts.append(random.choice(prompt_templates))

        return prompts if prompts else ["What is the nature of understanding?"]

    def get_foundation_prompts(self, count: int = 5) -> List[str]:
        """Get prompts from foundational topics (mathematics, basic physics)."""
        return self.generate_prompts(difficulty=1, count=count)

    def get_advanced_prompts(self, count: int = 5) -> List[str]:
        """Get prompts from advanced topics."""
        return self.generate_prompts(difficulty=5, count=count)

    def get_statistics(self) -> Dict:
        """Get curriculum statistics."""
        by_category = {}
        by_difficulty = {}

        for topic in self.topics.values():
            by_category[topic.category] = by_category.get(topic.category, 0) + 1
            by_difficulty[topic.difficulty] = by_difficulty.get(topic.difficulty, 0) + 1

        return {
            "total_topics": len(self.topics),
            "by_category": by_category,
            "by_difficulty": by_difficulty,
            "curriculum_dir": str(self.curriculum_dir)
        }
