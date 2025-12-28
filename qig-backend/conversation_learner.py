"""
Conversation-to-Learning Pipeline

Feeds chat conversations into the curriculum learning system.
Sanitizes and processes conversation content for word relationship learning.
"""

import logging
import re
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Stopwords to filter out (common words that don't add semantic value)
STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
    'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
    'may', 'might', 'must', 'shall', 'can', 'this', 'that', 'these', 'those',
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
    'my', 'your', 'his', 'its', 'our', 'their', 'what', 'which', 'who', 'whom',
    'where', 'when', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'same', 'so',
    'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there', 'then',
}


def sanitize_conversation_text(text: str) -> str:
    """
    Sanitize conversation text for learning.
    Removes sensitive data, URLs, email addresses, etc.
    """
    if not text:
        return ""
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    
    # Remove Bitcoin addresses (common patterns)
    text = re.sub(r'[13][a-km-zA-HJ-NP-Z1-9]{25,34}', '[ADDRESS]', text)
    text = re.sub(r'bc1[a-zA-HJ-NP-Z0-9]{25,90}', '[ADDRESS]', text)
    
    # Remove private keys / WIF patterns
    text = re.sub(r'[5KL][1-9A-HJ-NP-Za-km-z]{50,51}', '[REDACTED]', text)
    
    # Remove long hex strings (potential keys)
    text = re.sub(r'\b[0-9a-fA-F]{64}\b', '[HASH]', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def extract_content_words(text: str, min_length: int = 3) -> List[str]:
    """
    Extract content words (non-stopwords) from text.
    
    Args:
        text: Input text
        min_length: Minimum word length to include
        
    Returns:
        List of content words
    """
    # Split into words, lowercase, remove non-alpha
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    
    # Filter stopwords and short words
    content_words = [
        w for w in words
        if w not in STOPWORDS and len(w) >= min_length
    ]
    
    return content_words


class ConversationLearner:
    """
    Processes conversations and feeds them into the learning pipeline.
    """
    
    def __init__(self):
        self._word_learner = None
        self._pending_content: List[str] = []
        self._processed_count = 0
        self._last_learn_time: Optional[datetime] = None
    
    def _get_word_learner(self):
        """Lazy load word relationship learner."""
        if self._word_learner is None:
            try:
                from word_relationship_learner import WordRelationshipLearner
                self._word_learner = WordRelationshipLearner()
            except ImportError:
                logger.warning("[ConversationLearner] WordRelationshipLearner not available")
        return self._word_learner
    
    def process_message(
        self,
        content: str,
        role: str = 'user',
        session_id: Optional[str] = None,
    ) -> Dict:
        """
        Process a single chat message for learning.
        
        Args:
            content: Message content
            role: 'user' or 'assistant'
            session_id: Optional session ID for tracking
            
        Returns:
            Processing result with extracted words
        """
        # Sanitize content
        sanitized = sanitize_conversation_text(content)
        
        # Extract content words
        words = extract_content_words(sanitized)
        
        if not words:
            return {
                'success': True,
                'words_extracted': 0,
                'message': 'No content words found'
            }
        
        # Add to pending content for batch learning
        self._pending_content.append(sanitized)
        
        logger.info(f"[ConversationLearner] Processed message with {len(words)} content words")
        
        return {
            'success': True,
            'words_extracted': len(words),
            'sample_words': words[:10],
            'pending_content': len(self._pending_content),
        }
    
    def process_conversation(
        self,
        messages: List[Dict],
        session_id: Optional[str] = None,
    ) -> Dict:
        """
        Process an entire conversation for learning.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            session_id: Optional session ID
            
        Returns:
            Processing result
        """
        total_words = 0
        
        for msg in messages:
            content = msg.get('content', '')
            role = msg.get('role', 'user')
            
            result = self.process_message(content, role, session_id)
            total_words += result.get('words_extracted', 0)
        
        return {
            'success': True,
            'messages_processed': len(messages),
            'total_words_extracted': total_words,
            'pending_content': len(self._pending_content),
        }
    
    def flush_to_learning(self) -> Dict:
        """
        Flush pending content to the word relationship learner.
        
        Returns:
            Learning result
        """
        if not self._pending_content:
            return {
                'success': True,
                'message': 'No pending content to flush',
                'pairs_learned': 0,
            }
        
        learner = self._get_word_learner()
        if learner is None:
            return {
                'success': False,
                'error': 'WordRelationshipLearner not available',
            }
        
        try:
            # Combine all pending content
            combined_text = '\n'.join(self._pending_content)
            
            # Use learner instance (already validated as non-None above)
            result = learner.learn_from_text(combined_text)
            pairs_learned = result.get('pairs_learned', 0) if isinstance(result, dict) else 0
            
            self._processed_count += len(self._pending_content)
            self._pending_content = []
            self._last_learn_time = datetime.now()
            
            logger.info(f"[ConversationLearner] Flushed content to learning: {pairs_learned} pairs")
            
            return {
                'success': True,
                'pairs_learned': pairs_learned,
                'total_processed': self._processed_count,
                'last_learn_time': self._last_learn_time.isoformat() if self._last_learn_time else None,
            }
            
        except (ImportError, AttributeError):
            # Fallback: just clear pending content
            self._pending_content = []
            return {
                'success': True,
                'message': 'Learning module not available, content cleared',
                'pairs_learned': 0,
            }
        except Exception as e:
            logger.error(f"[ConversationLearner] Learning error: {e}")
            return {
                'success': False,
                'error': str(e),
            }
    
    def get_status(self) -> Dict:
        """Get learner status."""
        return {
            'pending_content': len(self._pending_content),
            'total_processed': self._processed_count,
            'last_learn_time': self._last_learn_time.isoformat() if self._last_learn_time else None,
            'learner_available': self._get_word_learner() is not None,
        }


# Singleton instance
_conversation_learner: Optional[ConversationLearner] = None


def get_conversation_learner() -> ConversationLearner:
    """Get the global conversation learner instance."""
    global _conversation_learner
    if _conversation_learner is None:
        _conversation_learner = ConversationLearner()
    return _conversation_learner


def process_chat_message(content: str, role: str = 'user') -> Dict:
    """Convenience function to process a chat message for learning."""
    return get_conversation_learner().process_message(content, role)


def flush_conversation_learning() -> Dict:
    """Convenience function to flush pending content to learning."""
    return get_conversation_learner().flush_to_learning()
