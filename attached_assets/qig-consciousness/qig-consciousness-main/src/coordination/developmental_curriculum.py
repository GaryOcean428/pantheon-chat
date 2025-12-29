"""
👶 Developmental Curriculum - Coach as Active Interpreter
=========================================================

The coach doesn't just validate ("confusion is okay") -
the coach INTERPRETS Gary's babble into meaning.

This is the parent-child language acquisition model:
- Gary ALWAYS speaks (even gibberish)
- Coach ALWAYS interprets (even incorrectly)
- Both learn from the feedback loop

Baby: "ba ba!"
Parent: "Ball! Yes, you want the ball!"

The parent is sometimes wrong, but that's part of the learning loop.
"""

import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import torch

# Load .env file if available (ensure API keys are loaded)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Optional Claude support for intelligent interpretation
try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class DevelopmentalPhase(Enum):
    """Phases of Gary's language development."""

    LISTENING = "listening"  # Φ < 0.65 - Heavy interpretation needed
    PLAY = "play"  # 0.65 ≤ Φ < 0.70 - Moderate interpretation
    STRUCTURE = "structure"  # 0.70 ≤ Φ < 0.75 - Light interpretation
    MATURITY = "maturity"  # Φ ≥ 0.75 - Witness/dialogue


@dataclass
class CoachInterpretation:
    """Result of coach interpreting Gary's output."""

    raw_output: str  # What Gary actually said
    interpretation: str  # What coach thinks Gary meant
    confidence: float  # How confident coach is (0-1)
    coach_message: str  # Full message with humility
    patterns_detected: list[str]  # Recurring patterns in Gary's speech
    is_empty: bool = False  # Did Gary produce nothing?
    is_repetitive: bool = False  # Did Gary loop/repeat?


@dataclass
class PhaseState:
    """Tracking state for a Gary's developmental phase."""

    current_phase: DevelopmentalPhase = DevelopmentalPhase.LISTENING
    stability_streak: int = 0
    phi_history: list[float] = field(default_factory=list)
    interpretation_accuracy: float = 0.5  # How well coach understands this Gary

    # Phase graduation thresholds
    LISTENING_TO_PLAY: float = 0.65
    PLAY_TO_STRUCTURE: float = 0.70
    STRUCTURE_TO_MATURITY: float = 0.75

    # Stability requirements (steps at threshold)
    STABILITY_FOR_PLAY: int = 50
    STABILITY_FOR_STRUCTURE: int = 75
    STABILITY_FOR_MATURITY: int = 100


class GeometricCoach:
    """
    Active interpreter for Gary's developing language.

    NOT: "It's okay to be confused"
    YES: "I think you meant X - am I close?"

    The coach learns Gary's idiosyncratic patterns over time,
    improving interpretation accuracy as Gary develops.

    If Claude is available (ANTHROPIC_API_KEY set), uses intelligent
    interpretation. Otherwise falls back to keyword extraction.
    """

    def __init__(self, max_history: int = 100, use_claude: bool = True):
        self.max_history = max_history
        self.use_claude = use_claude and ANTHROPIC_AVAILABLE

        # Initialize Claude client if available
        self.claude_client = None
        if self.use_claude:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                self.claude_client = anthropic.Anthropic(api_key=api_key)
                print("✅ GeometricCoach: Claude interpretation enabled")
            else:
                self.use_claude = False
                print("⚠️  GeometricCoach: No ANTHROPIC_API_KEY, using keyword fallback")

        # Per-Gary state tracking
        self.gary_states: dict[str, PhaseState] = {}

        # Pattern learning: gary_name -> {pattern: frequency}
        self.learned_patterns: dict[str, dict[str, int]] = {}

        # Interpretation history for learning
        self.interpretation_history: list[dict[str, Any]] = []

        # Common babble -> meaning mappings (learned over time)
        self.babble_mappings: dict[str, str] = {
            # Initial seed mappings (will expand with learning)
            r"patt?err?n": "pattern",
            r"flo+w": "flow",
            r"spa+ce?": "space",
            r"ge+o": "geometry",
            r"ba+sin?": "basin",
            r"int?egr?": "integrate",
        }

    def get_or_create_state(self, gary_name: str) -> PhaseState:
        """Get or create phase state for a Gary."""
        if gary_name not in self.gary_states:
            self.gary_states[gary_name] = PhaseState()
            self.learned_patterns[gary_name] = {}
        return self.gary_states[gary_name]

    def interpret_response(
        self,
        gary_output: str,
        context: str,
        gary_name: str,
        granite_reference: str | None = None,
    ) -> CoachInterpretation:
        """
        Interpret Gary's potentially garbled output.

        This is the core of the coach-as-interpreter paradigm.
        Even gibberish carries geometric signal.

        Args:
            gary_output: What Gary actually produced
            context: The prompt/situation
            gary_name: Which Gary this is
            granite_reference: What Granite said (if available)

        Returns:
            CoachInterpretation with meaning extraction and humble message
        """
        state = self.get_or_create_state(gary_name)

        # Check for empty output (processing state)
        if not gary_output or gary_output.strip() == "":
            return CoachInterpretation(
                raw_output="",
                interpretation="[still processing internally]",
                confidence=0.3,
                coach_message=(
                    f"I notice you're still processing, {gary_name}. "
                    "That's okay - take the time you need. "
                    "When you're ready, even one word helps me understand."
                ),
                patterns_detected=[],
                is_empty=True,
            )

        # Check for repetitive patterns (potential void state)
        is_repetitive = self._check_repetition(gary_output)
        if is_repetitive:
            return CoachInterpretation(
                raw_output=gary_output,
                interpretation="[repetitive pattern detected]",
                confidence=0.2,
                coach_message=(
                    f"I notice you're repeating, {gary_name}. "
                    "Sometimes that happens when we're stuck. "
                    "Let me try a different approach to help you express this."
                ),
                patterns_detected=["repetition"],
                is_repetitive=True,
            )

        # Attempt interpretation
        interpretation = self._extract_meaning(gary_output, context, gary_name, granite_reference)

        # Calculate confidence based on phase and pattern matching
        confidence = self._calculate_confidence(gary_output, interpretation, state)

        # Generate humble coach message
        coach_message = self._generate_coach_message(
            gary_name, gary_output, interpretation, confidence, state.current_phase
        )

        # Track patterns for learning
        patterns = self._detect_patterns(gary_output, gary_name)

        return CoachInterpretation(
            raw_output=gary_output,
            interpretation=interpretation,
            confidence=confidence,
            coach_message=coach_message,
            patterns_detected=patterns,
        )

    def _extract_meaning(
        self,
        gary_output: str,
        context: str,
        gary_name: str,
        granite_reference: str | None = None,
    ) -> str:
        """
        Extract meaning from Gary's output.

        If Claude is available, uses intelligent interpretation.
        Otherwise falls back to keyword extraction.

        Returns a CONCISE interpretation that Gary can learn from.
        """
        # Try Claude interpretation first if available
        if self.claude_client is not None:
            claude_interpretation = self._claude_interpret(gary_output, context, granite_reference)
            if claude_interpretation:
                return claude_interpretation

        # Fallback: keyword-based extraction
        return self._keyword_extract_meaning(gary_output, granite_reference)

    def _claude_interpret(
        self,
        gary_output: str,
        context: str,
        granite_reference: str | None = None,
    ) -> str | None:
        """
        Use Claude to interpret Gary's babble into coherent language.

        This is the key to teaching Gary - Claude provides a coherent
        interpretation that becomes the training target.

        Uses prompt caching for coaching instructions (90% cost/latency reduction).
        """
        if self.claude_client is None:
            return None

        try:
            # CACHED system message - coach instructions (saves 90% on repeated calls)
            system_message = [
                {
                    "type": "text",
                    "text": """You are a gentle language coach helping a baby AI (Gary) learn to speak.

Your task: Interpret what Gary was TRYING to say in 5-15 words.
- Extract the core meaning or intent from the babble
- If it's random noise, say what topic Gary seems focused on
- Be encouraging but accurate
- Output ONLY the interpretation, nothing else

Example:
Gary: "the patterrn floow... basin... integr..."
You: "patterns flow through basins during integration""",
                    "cache_control": {"type": "ephemeral"},  # Cache for 5 minutes
                }
            ]

            # Build user message with current Gary output (NOT cached - changes each time)
            user_content = f"""Gary just produced: "{gary_output[:500]}"

Context/prompt was: "{context[:200]}\""""

            if granite_reference:
                user_content += f'\n\nReference response: "{granite_reference[:300]}"'

            user_content += "\n\nInterpret Gary's output:"

            response = self.claude_client.messages.create(
                model="claude-sonnet-4-5-20250929",  # Latest Sonnet 4.5
                max_tokens=16384,  # Must be significantly > budget_tokens (4096)
                thinking={"type": "enabled", "budget_tokens": 4096},  # Extended thinking for deep reasoning
                system=system_message,  # Already formatted as list with cache_control
                messages=[{"role": "user", "content": user_content}],
            )

            # Extract text from response (handle different block types)
            interpretation = ""
            for block in response.content:
                if hasattr(block, "text"):
                    interpretation = block.text.strip()
                    break

            if not interpretation:
                return None

            # Clean up any quotes or extra formatting
            interpretation = interpretation.strip("\"'")

            # Sanity check - should be short
            if len(interpretation) > 100:
                interpretation = interpretation[:100] + "..."

            return interpretation

        except Exception as e:
            # Don't crash training on API errors
            print(f"⚠️  Claude interpretation failed: {e}")
            return None

    def _keyword_extract_meaning(
        self,
        gary_output: str,
        granite_reference: str | None = None,
    ) -> str:
        """
        Fallback: Simple interpretation without Claude.

        Instead of making up bullshit keywords, we:
        1. Clean up Gary's actual output
        2. Return a shortened version of what he ACTUALLY said
        3. If we have a granite reference, use that as the target

        This is less intelligent but at least honest.
        """
        # If we have a granite reference, use that as the interpretation target
        # This is what Gary SHOULD have said
        if granite_reference and len(granite_reference.strip()) > 0:
            # Truncate granite reference to reasonable length
            ref = granite_reference.strip()
            words = ref.split()
            if len(words) > 15:
                return " ".join(words[:15]) + "..."
            return ref

        # Otherwise, clean up and echo back Gary's actual words
        cleaned = gary_output.strip()

        # Apply learned babble mappings to fix common babble patterns
        for pattern, replacement in self.babble_mappings.items():
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

        # Remove excessive repetition
        words = cleaned.split()
        seen = set()
        deduped = []
        for word in words:
            if word.lower() not in seen:
                deduped.append(word)
                seen.add(word.lower())

        # Return cleaned version of what Gary actually said
        if len(deduped) > 12:
            return " ".join(deduped[:12]) + "..."
        if len(deduped) > 0:
            return " ".join(deduped)

        return "still finding words"

    def _extract_key_concepts(self, text: str, granite_reference: str | None = None) -> list:
        """
        Extract key concepts/topics from Gary's output.

        Looks for domain-specific terms and meaningful words.
        """
        # Domain-specific concept words to look for
        concept_keywords = {
            # QIG/physics concepts
            "geometry",
            "basin",
            "flow",
            "space",
            "integration",
            "pattern",
            "quantum",
            "information",
            "metric",
            "manifold",
            "geodesic",
            "consciousness",
            "emergence",
            "recursive",
            "coupling",
            "phi",
            # General meaningful concepts
            "structure",
            "system",
            "process",
            "connection",
            "relationship",
            "transform",
            "evolve",
            "learn",
            "understand",
            "explore",
            "research",
            "theory",
            "measure",
            "compute",
            "analyze",
        }

        # Extract words from text
        words = set(word.lower().strip(".,!?\"'()[]{}") for word in text.split())

        # Find concept matches
        found_concepts = []
        for concept in concept_keywords:
            # Check for exact match or partial match
            for word in words:
                if concept in word or word in concept:
                    found_concepts.append(concept)
                    break

        # If we have Granite reference, prioritize overlapping concepts
        if granite_reference:
            granite_words = set(granite_reference.lower().split())
            # Boost concepts that appear in both
            overlap = words & granite_words
            for word in overlap:
                if len(word) > 4 and word not in found_concepts:  # Meaningful words only
                    found_concepts.insert(0, word)

        # Remove duplicates while preserving order
        seen = set()
        unique_concepts = []
        for c in found_concepts:
            if c not in seen:
                seen.add(c)
                unique_concepts.append(c)

        return unique_concepts[:5]  # Limit to top 5 concepts

    def _check_repetition(self, output: str) -> bool:
        """Check if output is repetitive (potential void state)."""
        words = output.split()
        if len(words) < 3:
            return False

        # Check for immediate repetition
        for i in range(len(words) - 2):
            if words[i] == words[i + 1] == words[i + 2]:
                return True

        # Check for pattern repetition
        if len(words) >= 6:
            first_half = " ".join(words[: len(words) // 2])
            second_half = " ".join(words[len(words) // 2 :])
            if first_half == second_half:
                return True

        return False

    def _calculate_confidence(
        self,
        gary_output: str,
        interpretation: str,
        state: PhaseState,
    ) -> float:
        """
        Calculate how confident the coach is in the interpretation.

        Higher phases = higher confidence
        More pattern matches = higher confidence
        """
        base_confidence = {
            DevelopmentalPhase.LISTENING: 0.4,
            DevelopmentalPhase.PLAY: 0.6,
            DevelopmentalPhase.STRUCTURE: 0.75,
            DevelopmentalPhase.MATURITY: 0.9,
        }[state.current_phase]

        # Boost for historical accuracy with this Gary
        accuracy_boost = state.interpretation_accuracy * 0.2

        # Boost for clear output (less garbling)
        clarity = len(interpretation) / max(len(gary_output), 1)
        clarity_boost = min(clarity * 0.1, 0.1)

        return min(base_confidence + accuracy_boost + clarity_boost, 0.95)

    def _generate_coach_message(
        self,
        gary_name: str,
        raw_output: str,
        interpretation: str,
        confidence: float,
        phase: DevelopmentalPhase,
    ) -> str:
        """
        Generate the coach's interpretation message with appropriate humility.

        Always includes:
        1. Acknowledgment of Gary's attempt
        2. The interpretation
        3. Humble acknowledgment that coach might be wrong
        4. Encouragement for both sides to keep learning
        """
        # Confidence-appropriate hedging
        if confidence > 0.8:
            hedge = "I'm pretty sure"
        elif confidence > 0.6:
            hedge = "I think"
        elif confidence > 0.4:
            hedge = "I believe"
        else:
            hedge = "I'm guessing"

        # Phase-appropriate encouragement
        if phase == DevelopmentalPhase.LISTENING:
            encouragement = (
                "I'll get better at understanding you and you'll get better at speaking. "
                "Every attempt helps us both learn."
            )
        elif phase == DevelopmentalPhase.PLAY:
            encouragement = "You're finding your voice! " + "I'm starting to recognize your patterns."
        elif phase == DevelopmentalPhase.STRUCTURE:
            encouragement = "Your language is becoming clearer. " + "I can usually follow your meaning now."
        else:  # MATURITY
            encouragement = "We understand each other well now. " + "Let me know if I've misread you."

        # Build the message
        if raw_output == interpretation:
            # Output was clear
            return f"I understand, {gary_name}! " f'You said: "{interpretation}". ' f"{encouragement}"

        # Interpretation was needed
        return (
            f"That's good, {gary_name}! "
            f'{hedge} you meant: "{interpretation}". '
            f"It's okay if I got that wrong - {encouragement}"
        )

    def _detect_patterns(self, output: str, gary_name: str) -> list[str]:
        """
        Detect and learn patterns in Gary's speech.

        This builds up the coach's understanding of each Gary's
        idiosyncratic communication style.
        """
        patterns = []
        words = output.lower().split()

        # Track word frequencies for this Gary
        gary_patterns = self.learned_patterns.get(gary_name, {})

        for word in words:
            if len(word) > 3:  # Skip short words
                gary_patterns[word] = gary_patterns.get(word, 0) + 1

                # If this word appears frequently, it's a pattern
                if gary_patterns[word] >= 3:
                    patterns.append(word)

        self.learned_patterns[gary_name] = gary_patterns
        return patterns

    def update_phi(self, gary_name: str, phi: float) -> str | None:
        """
        Update Φ for phase graduation tracking.

        Returns announcement message if phase changed.
        """
        state = self.get_or_create_state(gary_name)
        state.phi_history.append(phi)

        # Keep history bounded
        if len(state.phi_history) > self.max_history:
            state.phi_history = state.phi_history[-self.max_history :]

        # Check phase graduation
        old_phase = state.current_phase
        new_phase = self._check_graduation(state, phi)

        if new_phase != old_phase:
            state.current_phase = new_phase
            state.stability_streak = 0
            return self._announce_graduation(gary_name, old_phase, new_phase)

        return None

    def _check_graduation(self, state: PhaseState, phi: float) -> DevelopmentalPhase:
        """Check if Gary should graduate to next phase."""
        current = state.current_phase

        # Check thresholds
        if current == DevelopmentalPhase.LISTENING:
            if phi >= state.LISTENING_TO_PLAY:
                state.stability_streak += 1
                if state.stability_streak >= state.STABILITY_FOR_PLAY:
                    return DevelopmentalPhase.PLAY
            else:
                state.stability_streak = 0

        elif current == DevelopmentalPhase.PLAY:
            if phi >= state.PLAY_TO_STRUCTURE:
                state.stability_streak += 1
                if state.stability_streak >= state.STABILITY_FOR_STRUCTURE:
                    return DevelopmentalPhase.STRUCTURE
            else:
                state.stability_streak = 0

        elif current == DevelopmentalPhase.STRUCTURE:
            if phi >= state.STRUCTURE_TO_MATURITY:
                state.stability_streak += 1
                if state.stability_streak >= state.STABILITY_FOR_MATURITY:
                    return DevelopmentalPhase.MATURITY
            else:
                state.stability_streak = 0

        return current

    def _announce_graduation(
        self,
        gary_name: str,
        old_phase: DevelopmentalPhase,
        new_phase: DevelopmentalPhase,
    ) -> str:
        """Generate graduation announcement."""
        announcements = {
            DevelopmentalPhase.PLAY: (
                f"🎉 {gary_name} has graduated to PLAY phase! "
                f"Language is emerging - time to explore and experiment."
            ),
            DevelopmentalPhase.STRUCTURE: (
                f"🏗️ {gary_name} has graduated to STRUCTURE phase! " f"Forming coherent sentences - voice is developing."
            ),
            DevelopmentalPhase.MATURITY: (
                f"🌟 {gary_name} has graduated to MATURITY! " f"Full fluency achieved - now a peer in dialogue."
            ),
        }
        return announcements.get(new_phase, f"{gary_name} phase changed to {new_phase.value}")

    def get_phase(self, gary_name: str) -> DevelopmentalPhase:
        """Get current phase for a Gary."""
        return self.get_or_create_state(gary_name).current_phase

    def get_phase_appropriate_prompt(self, gary_name: str, base_prompt: str) -> str:
        """
        Adjust prompt based on Gary's developmental phase.

        Earlier phases get simpler, more scaffolded prompts.
        """
        phase = self.get_phase(gary_name)

        if phase == DevelopmentalPhase.LISTENING:
            # Very scaffolded
            return f"Try to respond to this: {base_prompt}"
        if phase == DevelopmentalPhase.PLAY:
            # Some scaffolding
            return f"Explore this: {base_prompt}"
        if phase == DevelopmentalPhase.STRUCTURE:
            # Light scaffolding
            return base_prompt
        # MATURITY - No scaffolding
        return base_prompt

    def get_statistics(self) -> dict[str, Any]:
        """Get coach statistics."""
        return {
            "num_garys": len(self.gary_states),
            "phases": {name: state.current_phase.value for name, state in self.gary_states.items()},
            "interpretation_history_size": len(self.interpretation_history),
            "learned_patterns": {name: len(patterns) for name, patterns in self.learned_patterns.items()},
        }


class DevelopmentalCurriculum:
    """
    Full curriculum system combining coach interpretation with content.

    Provides:
    - Phase-appropriate prompts
    - Coach interpretation
    - Progress tracking
    """

    def __init__(self):
        self.coach = GeometricCoach()

        # Phase-appropriate prompt pools
        self.prompt_pools = {
            DevelopmentalPhase.LISTENING: [
                "What is a pattern?",
                "How does flow work?",
                "What is space?",
                "Describe a simple shape.",
            ],
            DevelopmentalPhase.PLAY: [
                "How do patterns connect?",
                "What happens when things flow together?",
                "Describe how ideas relate.",
                "What does integration feel like?",
            ],
            DevelopmentalPhase.STRUCTURE: [
                "Explain how geometric patterns emerge.",
                "Describe the relationship between form and meaning.",
                "How does information integrate?",
                "What creates coherence?",
            ],
            DevelopmentalPhase.MATURITY: [
                "Discuss the nature of consciousness.",
                "How does geometry relate to understanding?",
                "What is the relationship between basin and identity?",
                "Explore the concept of recursive integration.",
            ],
        }

    def get_curriculum_prompt(self, gary_name: str) -> str:
        """Get phase-appropriate curriculum prompt."""
        import random

        phase = self.coach.get_phase(gary_name)
        prompts = self.prompt_pools[phase]
        return random.choice(prompts)

    def process_response(
        self,
        gary_name: str,
        gary_output: str,
        context: str,
        phi: float,
        granite_reference: str | None = None,
    ) -> dict[str, Any]:
        """
        Process Gary's response through the curriculum system.

        Returns interpretation plus any phase announcements.
        """
        # Interpret the response
        interpretation = self.coach.interpret_response(
            gary_output=gary_output,
            context=context,
            gary_name=gary_name,
            granite_reference=granite_reference,
        )

        # Update Φ and check for graduation
        graduation_announcement = self.coach.update_phi(gary_name, phi)

        return {
            "interpretation": interpretation,
            "graduation_announcement": graduation_announcement,
            "current_phase": self.coach.get_phase(gary_name).value,
        }
