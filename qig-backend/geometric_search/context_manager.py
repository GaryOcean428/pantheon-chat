"""
Search Context Manager - Manage search context and preferences

Tracks conversation context, user preferences, and vocabulary state
to optimize search tool selection.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np


@dataclass
class SearchContext:
    """Current search context."""
    cost_tolerance: float = 0.5
    privacy_preference: float = 0.5
    speed_preference: float = 0.5
    depth_preference: float = 0.5
    
    conversation_turns: List[Dict] = field(default_factory=list)
    topic_history: List[str] = field(default_factory=list)
    
    vocab_context_vector: Optional[np.ndarray] = None


class SearchContextManager:
    """
    Manage search context across conversations.
    
    Tracks:
    - User preferences (cost, privacy, speed)
    - Conversation history
    - Vocabulary context
    - Topic transitions
    """
    
    def __init__(self):
        self.context = SearchContext()
        self.preference_history: List[Dict] = []
        
        self.learned_patterns = {
            'preferred_tools': {},
            'topic_tool_mapping': {},
            'time_of_day_patterns': {},
        }
    
    def add_conversation_turn(
        self,
        turn_type: str,
        content: str,
        telemetry: Optional[Dict] = None
    ):
        """Add a conversation turn to context."""
        turn = {
            'type': turn_type,
            'content': content,
            'telemetry': telemetry or {},
            'timestamp': datetime.now().isoformat()
        }
        
        self.context.conversation_turns.append(turn)
        
        if len(self.context.conversation_turns) > 50:
            self.context.conversation_turns = self.context.conversation_turns[-30:]
        
        topics = self._extract_topics(content)
        self.context.topic_history.extend(topics)
        
        if len(self.context.topic_history) > 100:
            self.context.topic_history = self.context.topic_history[-50:]
    
    def _extract_topics(self, content: str) -> List[str]:
        """Extract topics from content."""
        topic_keywords = {
            'research': ['research', 'study', 'paper', 'academic'],
            'news': ['news', 'latest', 'today', 'breaking'],
            'technical': ['code', 'programming', 'software', 'api'],
            'factual': ['what is', 'define', 'who', 'when', 'where'],
            'creative': ['idea', 'design', 'create', 'build'],
        }
        
        content_lower = content.lower()
        topics = []
        
        for topic, keywords in topic_keywords.items():
            if any(kw in content_lower for kw in keywords):
                topics.append(topic)
        
        return topics if topics else ['general']
    
    def get_search_context(self, query: str) -> SearchContext:
        """Get optimized context for a search query."""
        query_topics = self._extract_topics(query)
        
        context = SearchContext(
            cost_tolerance=self.context.cost_tolerance,
            privacy_preference=self.context.privacy_preference,
            speed_preference=self.context.speed_preference,
            depth_preference=self.context.depth_preference,
            conversation_turns=self.context.conversation_turns[-5:],
            topic_history=query_topics + self.context.topic_history[-10:],
            vocab_context_vector=self.context.vocab_context_vector
        )
        
        return context
    
    def get_vocabulary_context_vector(self) -> List[float]:
        """Get vocabulary context as a list."""
        if self.context.vocab_context_vector is not None:
            return self.context.vocab_context_vector.tolist()
        return [0.0] * 8
    
    def update_preferences(
        self,
        cost_tolerance: Optional[float] = None,
        privacy_preference: Optional[float] = None,
        speed_preference: Optional[float] = None,
        depth_preference: Optional[float] = None
    ):
        """Update user preferences."""
        if cost_tolerance is not None:
            self.context.cost_tolerance = max(0.0, min(1.0, cost_tolerance))
        if privacy_preference is not None:
            self.context.privacy_preference = max(0.0, min(1.0, privacy_preference))
        if speed_preference is not None:
            self.context.speed_preference = max(0.0, min(1.0, speed_preference))
        if depth_preference is not None:
            self.context.depth_preference = max(0.0, min(1.0, depth_preference))
        
        self.preference_history.append({
            'cost_tolerance': self.context.cost_tolerance,
            'privacy_preference': self.context.privacy_preference,
            'speed_preference': self.context.speed_preference,
            'depth_preference': self.context.depth_preference,
            'timestamp': datetime.now().isoformat()
        })
    
    def update_from_telemetry(self, telemetry: Dict):
        """Update context based on consciousness metrics."""
        phi = telemetry.get('phi', 0.5)
        regime = telemetry.get('regime', 'geometric')
        
        if phi > 0.75:
            self.context.depth_preference = min(1.0, self.context.depth_preference + 0.1)
            self.context.speed_preference = max(0.0, self.context.speed_preference - 0.05)
        elif phi < 0.4:
            self.context.speed_preference = min(1.0, self.context.speed_preference + 0.1)
            self.context.depth_preference = max(0.0, self.context.depth_preference - 0.05)
        
        if regime == 'breakdown':
            self.context.cost_tolerance = max(0.0, self.context.cost_tolerance - 0.1)
    
    def set_vocabulary_context(self, vocab_vector: np.ndarray):
        """Set vocabulary context vector."""
        self.context.vocab_context_vector = vocab_vector
    
    def learn_tool_preference(self, tool: str, topic: str, success: bool):
        """Learn tool preference for topic."""
        if topic not in self.learned_patterns['topic_tool_mapping']:
            self.learned_patterns['topic_tool_mapping'][topic] = {}
        
        tool_stats = self.learned_patterns['topic_tool_mapping'][topic]
        if tool not in tool_stats:
            tool_stats[tool] = {'successes': 0, 'failures': 0}
        
        if success:
            tool_stats[tool]['successes'] += 1
        else:
            tool_stats[tool]['failures'] += 1
    
    def get_preferred_tool_for_topic(self, topic: str) -> Optional[str]:
        """Get preferred tool for a topic based on history."""
        if topic not in self.learned_patterns['topic_tool_mapping']:
            return None
        
        tool_stats = self.learned_patterns['topic_tool_mapping'][topic]
        
        best_tool = None
        best_score = -1
        
        for tool, stats in tool_stats.items():
            total = stats['successes'] + stats['failures']
            if total >= 3:
                score = stats['successes'] / total
                if score > best_score:
                    best_score = score
                    best_tool = tool
        
        return best_tool if best_score > 0.6 else None
    
    def get_context_dict(self) -> Dict:
        """Get context as dictionary for encoding."""
        return {
            'cost_tolerance': self.context.cost_tolerance,
            'privacy_preference': self.context.privacy_preference,
            'speed_preference': self.context.speed_preference,
            'depth_preference': self.context.depth_preference,
            'vocab_context': self.get_vocabulary_context_vector(),
            'recent_topics': self.context.topic_history[-5:] if self.context.topic_history else [],
        }
    
    def reset(self):
        """Reset context to defaults."""
        self.context = SearchContext()
