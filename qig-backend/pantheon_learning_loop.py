"""
Pantheon Learning Loop - Dual-Mode Routing and Self-Improvement

This module implements:
1. Dual-mode routing (USER_FACING vs SELF_IMPROVEMENT)
2. Learning loop that routes failures to Pantheon debates
3. Pattern extraction and storage in Zettelkasten
4. Integration with FailureMonitor for continuous learning

The goal: Generate failures become learning opportunities, not retry loops.
Pantheon gods debate improvements, learned patterns improve future generation.

Author: Ocean/Zeus Pantheon
"""

import time
import logging
import hashlib
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class PantheonMode(Enum):
    """Operating mode for the Pantheon system."""
    USER_FACING = "user_facing"  # Gods respond to user queries
    SELF_IMPROVEMENT = "self_improvement"  # Gods debate to improve the system


class DebateOutcome(Enum):
    """Outcome of a Pantheon debate."""
    CONSENSUS = "consensus"  # Gods reached agreement
    MAJORITY = "majority"  # Majority agreement, some dissent
    SPLIT = "split"  # Significant disagreement
    INCONCLUSIVE = "inconclusive"  # No clear direction


class ImprovementType(Enum):
    """Type of improvement identified."""
    VOCABULARY = "vocabulary"  # Need new/better words
    GRAMMAR = "grammar"  # Grammatical pattern issue
    SEMANTIC = "semantic"  # Semantic coherence issue
    TOPIC = "topic"  # Topic drift issue
    STRUCTURE = "structure"  # Sentence structure issue
    DOMAIN = "domain"  # Domain knowledge gap


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DebateContribution:
    """A single contribution from a god in a debate."""
    god_name: str
    position: str
    reasoning: str
    confidence: float  # 0-1
    proposed_solution: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return {
            'god_name': self.god_name,
            'position': self.position,
            'reasoning': self.reasoning,
            'confidence': self.confidence,
            'proposed_solution': self.proposed_solution,
            'timestamp': self.timestamp,
        }


@dataclass
class DebateResult:
    """Result of a Pantheon debate about an improvement."""
    debate_id: str
    topic: str
    outcome: DebateOutcome
    contributions: List[DebateContribution]
    consensus_position: Optional[str]
    improvement_type: ImprovementType
    action_items: List[str]
    confidence: float  # Overall confidence in the solution
    duration_seconds: float
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return {
            'debate_id': self.debate_id,
            'topic': self.topic,
            'outcome': self.outcome.value,
            'contributions': [c.to_dict() for c in self.contributions],
            'consensus_position': self.consensus_position,
            'improvement_type': self.improvement_type.value,
            'action_items': self.action_items,
            'confidence': self.confidence,
            'duration_seconds': self.duration_seconds,
            'timestamp': self.timestamp,
        }


@dataclass
class LearningPattern:
    """A pattern learned from debate that can improve generation."""
    pattern_id: str
    pattern_type: ImprovementType
    description: str
    trigger_condition: str  # When to apply this pattern
    correction: str  # What to do differently
    examples: List[Tuple[str, str]]  # (bad, good) pairs
    confidence: float
    source_debate_id: str
    usage_count: int = 0
    success_count: int = 0
    created_at: float = field(default_factory=time.time)
    
    @property
    def success_rate(self) -> float:
        if self.usage_count == 0:
            return 0.5  # Prior
        return self.success_count / self.usage_count
    
    def to_dict(self) -> Dict:
        return {
            'pattern_id': self.pattern_id,
            'pattern_type': self.pattern_type.value,
            'description': self.description,
            'trigger_condition': self.trigger_condition,
            'correction': self.correction,
            'examples': self.examples,
            'confidence': self.confidence,
            'source_debate_id': self.source_debate_id,
            'usage_count': self.usage_count,
            'success_count': self.success_count,
            'success_rate': self.success_rate,
            'created_at': self.created_at,
        }


@dataclass 
class FailureAnalysis:
    """Analysis of a generation failure."""
    failure_text: str
    failure_type: ImprovementType
    specific_issues: List[str]
    suggested_fixes: List[str]
    severity: float  # 0-1
    context: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# LEARNING LOOP ORCHESTRATOR
# =============================================================================

class LearningLoopOrchestrator:
    """
    Orchestrates the Pantheon learning loop.
    
    Routes failures to debates, extracts patterns, stores in Zettelkasten.
    The goal is to replace retry loops with genuine learning.
    """
    
    # God assignments for different improvement types
    GOD_EXPERTISE = {
        ImprovementType.VOCABULARY: ['Athena', 'Apollo', 'Hermes'],
        ImprovementType.GRAMMAR: ['Athena', 'Apollo'],
        ImprovementType.SEMANTIC: ['Zeus', 'Athena', 'Apollo'],
        ImprovementType.TOPIC: ['Apollo', 'Athena', 'Zeus'],
        ImprovementType.STRUCTURE: ['Athena', 'Hephaestus'],
        ImprovementType.DOMAIN: ['Apollo', 'Athena', 'Hermes'],
    }
    
    def __init__(self):
        self._mode = PantheonMode.USER_FACING
        self._active_debates: Dict[str, DebateResult] = {}
        self._learned_patterns: Dict[str, LearningPattern] = {}
        self._failure_queue: List[FailureAnalysis] = []
        self._debate_count = 0
        
        # Try to connect to external systems
        self._failure_monitor = None
        self._zettelkasten = None
        self._pantheon_chat = None
        
        self._initialize_connections()
        
        logger.info("[LearningLoop] Orchestrator initialized")
    
    def _initialize_connections(self) -> None:
        """Initialize connections to external systems."""
        # Connect to failure monitor
        try:
            from agent_failure_taxonomy import get_failure_monitor
            self._failure_monitor = get_failure_monitor()
            logger.info("[LearningLoop] Connected to FailureMonitor")
        except ImportError:
            logger.warning("[LearningLoop] FailureMonitor not available")
        
        # Connect to Zettelkasten
        try:
            from zettelkasten_memory import get_zettelkasten_memory
            self._zettelkasten = get_zettelkasten_memory()
            logger.info("[LearningLoop] Connected to Zettelkasten")
        except ImportError:
            logger.warning("[LearningLoop] Zettelkasten not available")
        
        # Connect to Pantheon chat (for debates)
        try:
            from olympus.pantheon_chat import get_pantheon_chat
            self._pantheon_chat = get_pantheon_chat()
            logger.info("[LearningLoop] Connected to PantheonChat")
        except ImportError:
            logger.warning("[LearningLoop] PantheonChat not available")
    
    @property
    def mode(self) -> PantheonMode:
        """Current operating mode."""
        return self._mode
    
    def set_mode(self, mode: PantheonMode) -> None:
        """Switch operating mode."""
        old_mode = self._mode
        self._mode = mode
        logger.info(f"[LearningLoop] Mode changed: {old_mode.value} -> {mode.value}")
    
    def is_self_improvement_mode(self) -> bool:
        """Check if currently in self-improvement mode."""
        return self._mode == PantheonMode.SELF_IMPROVEMENT
    
    # =========================================================================
    # FAILURE ANALYSIS
    # =========================================================================
    
    def analyze_failure(
        self,
        failed_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> FailureAnalysis:
        """
        Analyze a generation failure to determine what went wrong.
        
        Args:
            failed_text: The text that failed coherence checks
            context: Additional context (topic, goals, etc.)
            
        Returns:
            FailureAnalysis with identified issues
        """
        context = context or {}
        issues = []
        fixes = []
        
        words = failed_text.split()
        
        # Check for grammatical issues
        gram_issues = self._check_grammatical_issues(words)
        if gram_issues:
            issues.extend(gram_issues)
            fixes.append("Apply stronger grammatical coherence constraints")
        
        # Check for semantic issues
        sem_issues = self._check_semantic_issues(words, context.get('topic'))
        if sem_issues:
            issues.extend(sem_issues)
            fixes.append("Improve semantic domain clustering")
        
        # Check for repetition
        rep_issues = self._check_repetition(words)
        if rep_issues:
            issues.extend(rep_issues)
            fixes.append("Add repetition penalty to word selection")
        
        # Check for topic drift
        if context.get('topic'):
            drift_issues = self._check_topic_drift(words, context['topic'])
            if drift_issues:
                issues.extend(drift_issues)
                fixes.append("Strengthen topic coherence scoring")
        
        # Determine primary failure type
        failure_type = self._classify_failure_type(issues)
        
        # Calculate severity
        severity = min(1.0, len(issues) / 5)
        
        return FailureAnalysis(
            failure_text=failed_text,
            failure_type=failure_type,
            specific_issues=issues,
            suggested_fixes=fixes,
            severity=severity,
            context=context,
        )
    
    def _check_grammatical_issues(self, words: List[str]) -> List[str]:
        """Check for grammatical issues in word sequence."""
        issues = []
        
        # Check for bad sentence starters
        if words and words[0].lower() in ('and', 'or', 'but', 'because'):
            issues.append(f"Bad sentence starter: '{words[0]}'")
        
        # Check for article-article sequences
        articles = {'the', 'a', 'an', 'this', 'that'}
        for i in range(len(words) - 1):
            if words[i].lower() in articles and words[i+1].lower() in articles:
                issues.append(f"Double article: '{words[i]} {words[i+1]}'")
        
        return issues
    
    def _check_semantic_issues(self, words: List[str], topic: Optional[str]) -> List[str]:
        """Check for semantic coherence issues."""
        issues = []
        
        # Check for random word insertion
        technical_words = {'quantum', 'entropy', 'eigenvalue', 'tensor', 'manifold'}
        common_words = {'the', 'a', 'is', 'are', 'was', 'were', 'has', 'have'}
        
        technical_count = sum(1 for w in words if w.lower() in technical_words)
        if technical_count > len(words) * 0.4:
            issues.append("Too many technical terms without connective tissue")
        
        return issues
    
    def _check_repetition(self, words: List[str]) -> List[str]:
        """Check for word repetition issues."""
        issues = []
        
        # Check for immediate repetition
        for i in range(len(words) - 1):
            if words[i].lower() == words[i+1].lower():
                issues.append(f"Immediate repetition: '{words[i]}'")
        
        # Check for excessive word frequency
        word_counts: Dict[str, int] = {}
        for w in words:
            w_lower = w.lower()
            if w_lower not in {'the', 'a', 'an', 'is', 'are', 'of', 'to', 'and'}:
                word_counts[w_lower] = word_counts.get(w_lower, 0) + 1
        
        for word, count in word_counts.items():
            if count > 2 and len(words) < 20:
                issues.append(f"Word overuse: '{word}' appears {count} times")
        
        return issues
    
    def _check_topic_drift(self, words: List[str], topic: str) -> List[str]:
        """Check if text drifted from topic."""
        issues = []
        
        # Simple topic relevance check
        topic_words = set(topic.lower().split())
        text_words = set(w.lower() for w in words)
        
        overlap = topic_words & text_words
        if len(overlap) == 0 and len(topic_words) > 0:
            issues.append(f"No topic words present. Expected: {topic_words}")
        
        return issues
    
    def _classify_failure_type(self, issues: List[str]) -> ImprovementType:
        """Classify the primary type of failure based on issues."""
        if not issues:
            return ImprovementType.SEMANTIC
        
        # Count issue categories
        categories = {
            ImprovementType.GRAMMAR: 0,
            ImprovementType.SEMANTIC: 0,
            ImprovementType.TOPIC: 0,
            ImprovementType.VOCABULARY: 0,
        }
        
        for issue in issues:
            issue_lower = issue.lower()
            if 'article' in issue_lower or 'starter' in issue_lower:
                categories[ImprovementType.GRAMMAR] += 1
            elif 'technical' in issue_lower or 'term' in issue_lower:
                categories[ImprovementType.VOCABULARY] += 1
            elif 'topic' in issue_lower or 'drift' in issue_lower:
                categories[ImprovementType.TOPIC] += 1
            elif 'repetition' in issue_lower or 'overuse' in issue_lower:
                categories[ImprovementType.SEMANTIC] += 1
            else:
                categories[ImprovementType.SEMANTIC] += 1
        
        return max(categories, key=categories.get)
    
    # =========================================================================
    # PANTHEON DEBATE
    # =========================================================================
    
    def debate_for_improvement(
        self,
        failure_analysis: FailureAnalysis
    ) -> DebateResult:
        """
        Trigger a Pantheon debate about how to fix a failure.
        
        Args:
            failure_analysis: Analysis of what went wrong
            
        Returns:
            DebateResult with god contributions and consensus
        """
        start_time = time.time()
        self._debate_count += 1
        
        debate_id = f"debate_{int(time.time())}_{self._debate_count}"
        
        # Get gods with expertise in this failure type
        expert_gods = self.GOD_EXPERTISE.get(
            failure_analysis.failure_type,
            ['Zeus', 'Athena', 'Apollo']
        )
        
        # Generate debate contributions from each god
        contributions = []
        for god_name in expert_gods:
            contribution = self._generate_god_contribution(
                god_name=god_name,
                failure_analysis=failure_analysis,
                debate_id=debate_id,
            )
            contributions.append(contribution)
        
        # Synthesize consensus
        outcome, consensus, confidence = self._synthesize_consensus(contributions)
        
        # Extract action items
        action_items = self._extract_action_items(contributions, consensus)
        
        duration = time.time() - start_time
        
        result = DebateResult(
            debate_id=debate_id,
            topic=f"Improve {failure_analysis.failure_type.value} for: '{failure_analysis.failure_text[:50]}...'",
            outcome=outcome,
            contributions=contributions,
            consensus_position=consensus,
            improvement_type=failure_analysis.failure_type,
            action_items=action_items,
            confidence=confidence,
            duration_seconds=duration,
        )
        
        # Store debate
        self._active_debates[debate_id] = result
        
        logger.info(f"[LearningLoop] Debate {debate_id} completed: {outcome.value} ({confidence:.2f} confidence)")
        
        return result
    
    def _generate_god_contribution(
        self,
        god_name: str,
        failure_analysis: FailureAnalysis,
        debate_id: str,
    ) -> DebateContribution:
        """Generate a contribution from a specific god."""
        # God-specific reasoning styles
        god_styles = {
            'Zeus': {
                'approach': 'holistic integration',
                'focus': 'overall coherence and system harmony',
            },
            'Athena': {
                'approach': 'strategic analysis',
                'focus': 'logical structure and tactical patterns',
            },
            'Apollo': {
                'approach': 'truth seeking',
                'focus': 'clarity and accurate knowledge',
            },
            'Hermes': {
                'approach': 'communication optimization',
                'focus': 'efficient information flow',
            },
            'Hephaestus': {
                'approach': 'structural engineering',
                'focus': 'robust construction and reliability',
            },
        }
        
        style = god_styles.get(god_name, {'approach': 'general analysis', 'focus': 'improvement'})
        
        # Generate position based on failure type and god expertise
        position = self._formulate_god_position(god_name, failure_analysis, style)
        reasoning = self._formulate_god_reasoning(god_name, failure_analysis, style)
        solution = self._formulate_god_solution(god_name, failure_analysis, style)
        
        # Confidence based on expertise alignment
        expertise_alignment = 1.0 if god_name in self.GOD_EXPERTISE.get(failure_analysis.failure_type, []) else 0.7
        confidence = 0.6 + (expertise_alignment * 0.3)
        
        return DebateContribution(
            god_name=god_name,
            position=position,
            reasoning=reasoning,
            confidence=confidence,
            proposed_solution=solution,
        )
    
    def _formulate_god_position(
        self,
        god_name: str,
        failure: FailureAnalysis,
        style: Dict[str, str]
    ) -> str:
        """Formulate a god's position on the failure."""
        positions = {
            'Zeus': f"Through {style['approach']}, I see the root cause lies in {failure.failure_type.value} coordination.",
            'Athena': f"Strategic analysis reveals {len(failure.specific_issues)} tactical weaknesses in the generation.",
            'Apollo': f"The light of truth shows that {failure.failure_type.value} patterns need refinement.",
            'Hermes': f"Communication flow analysis indicates bottlenecks in {failure.failure_type.value} processing.",
            'Hephaestus': f"Structural examination reveals foundational issues in {failure.failure_type.value} architecture.",
        }
        return positions.get(god_name, f"Analysis shows {failure.failure_type.value} requires attention.")
    
    def _formulate_god_reasoning(
        self,
        god_name: str,
        failure: FailureAnalysis,
        style: Dict[str, str]
    ) -> str:
        """Formulate a god's reasoning about the failure."""
        issues_summary = '; '.join(failure.specific_issues[:3]) if failure.specific_issues else 'unclear issues'
        return f"Focusing on {style['focus']}, the specific issues are: {issues_summary}. Severity: {failure.severity:.2f}."
    
    def _formulate_god_solution(
        self,
        god_name: str,
        failure: FailureAnalysis,
        style: Dict[str, str]
    ) -> str:
        """Formulate a god's proposed solution."""
        if failure.suggested_fixes:
            return f"Apply {style['approach']}: {failure.suggested_fixes[0]}"
        return f"Apply {style['approach']} to improve {failure.failure_type.value} handling."
    
    def _synthesize_consensus(
        self,
        contributions: List[DebateContribution]
    ) -> Tuple[DebateOutcome, str, float]:
        """
        Synthesize consensus from god contributions.
        
        Returns:
            Tuple of (outcome, consensus_position, confidence)
        """
        if not contributions:
            return DebateOutcome.INCONCLUSIVE, None, 0.0
        
        # Calculate average confidence
        avg_confidence = sum(c.confidence for c in contributions) / len(contributions)
        
        # Check for solution agreement
        solutions = [c.proposed_solution for c in contributions if c.proposed_solution]
        
        if avg_confidence > 0.75:
            outcome = DebateOutcome.CONSENSUS
            consensus = solutions[0] if solutions else "Strengthen coherence scoring across all dimensions."
        elif avg_confidence > 0.6:
            outcome = DebateOutcome.MAJORITY
            consensus = solutions[0] if solutions else "Apply targeted improvements to identified weak points."
        else:
            outcome = DebateOutcome.SPLIT
            consensus = "Multiple approaches needed; prioritize by severity."
        
        return outcome, consensus, avg_confidence
    
    def _extract_action_items(
        self,
        contributions: List[DebateContribution],
        consensus: Optional[str]
    ) -> List[str]:
        """Extract actionable items from debate."""
        items = []
        
        # Add consensus as primary action
        if consensus:
            items.append(consensus)
        
        # Add unique proposed solutions
        seen = set()
        for contrib in contributions:
            if contrib.proposed_solution and contrib.proposed_solution not in seen:
                seen.add(contrib.proposed_solution)
                if len(items) < 5:
                    items.append(contrib.proposed_solution)
        
        return items
    
    # =========================================================================
    # PATTERN STORAGE
    # =========================================================================
    
    def store_learned_pattern(self, debate_result: DebateResult) -> LearningPattern:
        """
        Extract and store a learning pattern from a debate result.
        
        Args:
            debate_result: Result of a Pantheon debate
            
        Returns:
            The stored LearningPattern
        """
        pattern_id = f"pattern_{hashlib.md5(debate_result.debate_id.encode()).hexdigest()[:12]}"
        
        # Build trigger condition from debate topic
        trigger = f"When {debate_result.improvement_type.value} coherence falls below threshold"
        
        # Build correction from action items
        correction = debate_result.action_items[0] if debate_result.action_items else "Apply stronger coherence constraints"
        
        # Create pattern
        pattern = LearningPattern(
            pattern_id=pattern_id,
            pattern_type=debate_result.improvement_type,
            description=debate_result.consensus_position or "Improve coherence",
            trigger_condition=trigger,
            correction=correction,
            examples=[],  # Will be populated over time
            confidence=debate_result.confidence,
            source_debate_id=debate_result.debate_id,
        )
        
        # Store locally
        self._learned_patterns[pattern_id] = pattern
        
        # Store in Zettelkasten for long-term memory
        if self._zettelkasten:
            self._store_pattern_in_zettelkasten(pattern, debate_result)
        
        logger.info(f"[LearningLoop] Stored pattern {pattern_id}: {pattern.description[:50]}...")
        
        return pattern
    
    def _store_pattern_in_zettelkasten(
        self,
        pattern: LearningPattern,
        debate_result: DebateResult
    ) -> None:
        """Store pattern in Zettelkasten for long-term memory."""
        content = f"""
Learning Pattern: {pattern.pattern_type.value}

Description: {pattern.description}

Trigger: {pattern.trigger_condition}

Correction: {pattern.correction}

Source: Debate {debate_result.debate_id}
Confidence: {pattern.confidence:.2f}
Contributing Gods: {', '.join(c.god_name for c in debate_result.contributions)}
"""
        
        self._zettelkasten.add(
            content=content.strip(),
            source=f"pantheon_debate_{debate_result.debate_id}",
        )
    
    # =========================================================================
    # MAIN LEARNING FLOW
    # =========================================================================
    
    def process_generation_failure(
        self,
        failed_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[LearningPattern, List[str]]:
        """
        Process a generation failure through the full learning loop.
        
        This is the main entry point. Instead of retrying, we:
        1. Analyze the failure
        2. Debate improvements
        3. Store learned patterns
        4. Return actionable improvements
        
        Args:
            failed_text: The text that failed coherence checks
            context: Additional context
            
        Returns:
            Tuple of (learned_pattern, improvement_suggestions)
        """
        # Switch to self-improvement mode
        original_mode = self._mode
        self.set_mode(PantheonMode.SELF_IMPROVEMENT)
        
        try:
            # 1. Analyze the failure
            analysis = self.analyze_failure(failed_text, context)
            
            # 2. Trigger Pantheon debate
            debate_result = self.debate_for_improvement(analysis)
            
            # 3. Store learned pattern
            pattern = self.store_learned_pattern(debate_result)
            
            # 4. Record failure in failure monitor (as learning event, not error)
            if self._failure_monitor:
                self._failure_monitor.record_state(
                    agent_id="qig_generation",
                    basin_coords=[0.0] * 64,  # Placeholder
                    confidence=debate_result.confidence,
                    reasoning_quality=debate_result.confidence,
                    context_usage=0.5,
                    iteration=self._debate_count,
                    action_taken="learning_from_failure",
                    progress_metric=pattern.confidence,
                )
            
            return pattern, debate_result.action_items
            
        finally:
            # Restore original mode
            self.set_mode(original_mode)
    
    def get_applicable_patterns(
        self,
        improvement_type: ImprovementType
    ) -> List[LearningPattern]:
        """Get learned patterns applicable to a given improvement type."""
        return [
            p for p in self._learned_patterns.values()
            if p.pattern_type == improvement_type and p.success_rate > 0.4
        ]
    
    def record_pattern_usage(
        self,
        pattern_id: str,
        success: bool
    ) -> None:
        """Record usage of a pattern and whether it was successful."""
        if pattern_id in self._learned_patterns:
            pattern = self._learned_patterns[pattern_id]
            pattern.usage_count += 1
            if success:
                pattern.success_count += 1
            
            logger.debug(f"[LearningLoop] Pattern {pattern_id} usage: success={success}, rate={pattern.success_rate:.2f}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learning loop statistics."""
        return {
            'mode': self._mode.value,
            'total_debates': self._debate_count,
            'active_debates': len(self._active_debates),
            'learned_patterns': len(self._learned_patterns),
            'queued_failures': len(self._failure_queue),
            'pattern_success_rates': {
                p.pattern_id: p.success_rate
                for p in self._learned_patterns.values()
            },
            'has_failure_monitor': self._failure_monitor is not None,
            'has_zettelkasten': self._zettelkasten is not None,
            'has_pantheon_chat': self._pantheon_chat is not None,
        }


# =============================================================================
# SINGLETON
# =============================================================================

_learning_loop: Optional[LearningLoopOrchestrator] = None


def get_learning_loop() -> LearningLoopOrchestrator:
    """Get the singleton LearningLoopOrchestrator instance."""
    global _learning_loop
    if _learning_loop is None:
        _learning_loop = LearningLoopOrchestrator()
    return _learning_loop


# =============================================================================
# INTEGRATION HELPER
# =============================================================================

def handle_generation_failure(
    failed_text: str,
    context: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str, List[str]]:
    """
    Handle a generation failure through the learning loop.
    
    This replaces retry logic with learning.
    
    Args:
        failed_text: The text that failed
        context: Additional context
        
    Returns:
        Tuple of (should_use_fallback, fallback_reason, improvements)
    """
    loop = get_learning_loop()
    
    try:
        pattern, improvements = loop.process_generation_failure(failed_text, context)
        
        # Return that we learned from this failure
        return True, f"Learned pattern: {pattern.description}", improvements
        
    except Exception as e:
        logger.error(f"[LearningLoop] Failed to process failure: {e}")
        return True, "Learning loop error; using fallback", []


print("[PantheonLearningLoop] Module loaded - dual-mode routing and learning loop ready")
